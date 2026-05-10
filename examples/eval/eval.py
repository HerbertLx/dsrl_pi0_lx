"""Evaluation helpers: load SAC (PixelSAC) checkpoint trained with ``launch_train_sim`` / smoke env."""

import dataclasses
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import PIL.Image

import numpy as np
import torch
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import AttrDict, add_batch_dim

import openpi.transforms as openpi_transforms
from openpi.policies import policy_config as openpi_policy_config
from openpi.training import config as openpi_train_config

from examples.train_sim import SmokeEnv
from examples.train_utils_offline import eval_agent

# 默认与当前实验目录一致；可通过环境变量 SAC_CHECKPOINT_DIR 或 init_agent(checkpoint_dir=...) 覆盖
_DEFAULT_CHECKPOINT_DIR = (
    "/root/storage/CODE/lx/dsrl_pi0_lx/logs/DSRL_pi0_Aloha/"
    "dsrl_pi0_aloha_2026_05_05_12_23_05_0000--s-0/checkpoint2950000"
)


def init_agent(checkpoint_dir=None, seed=0):
    """
    构建与训练时一致的 PixelSACLearner（SmokeEnv 推断观测/动作维度），并从 Orbax/Flax checkpoint 恢复权重。

    超参数必须与训练一致（``examples/launch_train_sim.py`` 默认值 + ``scripts/run_offline.sh`` 中的覆盖项）。
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get("SAC_CHECKPOINT_DIR", _DEFAULT_CHECKPOINT_DIR)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    variant = AttrDict(
        seed=seed,
        resize_image=64,
        add_states=1,
    )
    dummy_env = SmokeEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())

    # 与 examples/launch_train_sim.train_args_dict + run_offline.sh 对齐；main 里会把 num_cameras 设为 variant.num_cameras（SmokeEnv 设为 3）
    kwargs = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding="VALID",
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction="mean",
        dropout_rate=0.0,
        aug_next=True,
        use_bottleneck=True,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy=0.0,
        num_qs=10,
        action_magnitude=2.0,
        num_cameras=variant.num_cameras,
        color_jitter=True,
    )

    agent = PixelSACLearner(seed, sample_obs, sample_action, **kwargs)
    agent.restore_checkpoint(checkpoint_dir)
    return agent

def init_agent_dp(
    ckpt_path="/root/storage/CODE/lx/dsrl_pi0_lx/openpi/checkpoints/pi0_airbot_local/lx_experiment/99999",
    config_name="pi0_airbot_local",
    task_description=(
        "use the left arm to pick up the yellow cup, then use the right arm to pour the coke"
    ),
    asset_id="0407",
):
    """与 ``openpi/scripts/remote.py`` 的 ``load_model`` 一致，返回 OpenPI ``Policy``。"""
    cfg = openpi_train_config.get_config(config_name)
    assets_base_dir = str(Path(ckpt_path).resolve() / "assets")
    train_cfg = dataclasses.replace(
        cfg,
        assets_base_dir=assets_base_dir,
        data=dataclasses.replace(
            cfg.data,
            assets=dataclasses.replace(cfg.data.assets, asset_id=asset_id),
        ),
    )
    return openpi_policy_config.create_trained_policy(
        train_cfg,
        ckpt_path,
        repack_transforms=openpi_transforms.Group(
            inputs=[
                openpi_transforms.RepackTransform(
                    {
                        "images": {
                            "cam_left_wrist": "observation.images.cam_left",
                            "cam_high": "observation.images.cam_head",
                            "cam_right_wrist": "observation.images.cam_right",
                        },
                        "state": "observation.state",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        default_prompt=task_description,
    )

def infer(agent, agent_dp, obs):
    obs_dict = oepnpi_obs2dict(obs)
    noise = np.asarray(agent.sample_actions(obs_dict))
    noise = np.pad(noise, [(0, 0)] * (noise.ndim - 1) + [(0, max(0, 32 - noise.shape[-1]))], mode="constant")
    noise = np.repeat(noise[:, np.newaxis, :], 50, axis=1)  # Pi0 action_horizon=50：同一噪声复制 50 份

    actions = agent_dp.infer(obs=obs, noise=noise)["actions"]
    return actions

def generate_rand_obs(variant=None, rng=None):
    """
    构造与 ``collect_traj`` 中写入 ``obs_list`` 的单步 ``obs_dict`` 相同布局的随机观测，供 ``agent.sample_actions`` 等测试使用。

    形状对齐逻辑（见 ``train_utils_sim.collect_traj``）::

        curr_image: (H, W, C)，C = 3 * num_cameras（smoke 下 SmokeEnv 设为 3 相机 → C=9）
        obs_dict['pixels']: (1, H, W, C, 1)
        obs_dict['state']: (1, state_dim, 1)  当 ``variant.add_states`` 为真时

    与 ``train_utils_offline.generate_traj`` 中 ``_obs_dict()`` 的键与维度一致；此处 ``pixels`` 使用 ``uint8``，与仿真里 ``obs_to_img`` 输出一致（合成 traj 里有时用 ``float32``，仅 dtype 可能不同）。
    """
    if variant is None:
        variant = AttrDict(resize_image=64, add_states=1, env="smoke")
    # 与训练时一致：SmokeEnv 会写入 num_cameras（一般为 3）
    if not getattr(variant, "num_cameras", None):
        SmokeEnv(variant)

    rng = np.random.default_rng(rng)
    h = w = int(variant.resize_image)
    c = int(3 * variant.num_cameras)
    state_dim = 14

    curr_image = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
    qpos = rng.standard_normal(state_dim).astype(np.float32)

    if variant.add_states:
        return {
            "pixels": curr_image[np.newaxis, ..., np.newaxis],
            "state": qpos[np.newaxis, ..., np.newaxis],
        }
    return {
        "pixels": curr_image[np.newaxis, ..., np.newaxis],
    }

def generate_local_obs(obs_path):
    """加载 ``remote`` 存下来的 obs pickle（``observation.images.*`` / ``observation.state`` / ``prompt``）。"""
    with open(obs_path, "rb") as f:
        data = pickle.load(f)

    def to_torch(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        if hasattr(v, "__array__"):
            return torch.from_numpy(np.asarray(v))
        return v

    return {k: to_torch(v) for k, v in data.items()}

def _remote_image_to_hwc_uint8(img):
    """Remote/DALI 图像常为 ``torch.Tensor`` / ``numpy``，形状 CHW、float32、取值约 ``[0,1]``。"""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected image with ndim 3, got shape {arr.shape}")
    # CHW → HWC
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr_f = np.clip(arr.astype(np.float32), 0.0, 1.0)
        arr = (arr_f * 255.0).round().astype(np.uint8)
    return arr

def oepnpi_obs2dict(obs, variant=None, resize_image=None):
    """
    将 ``generate_local_obs`` / remote 风格的 OpenPI 观测转为 PixelSAC 用的字典（与 ``generate_rand_obs`` 一致）::

        pixels: (1, H, W, 3 * num_cameras, 1)，uint8
        state: (1, 14, 1)，float32（当 ``variant.add_states`` 为真）

    三相机沿通道拼接顺序与 ``train_utils_offline._fuse_three_cams_rgb`` 一致：left → right → head，
    对应键 ``observation.images.cam_left`` / ``cam_right`` / ``cam_head``。
    """
    if variant is None:
        variant = AttrDict(resize_image=64, add_states=1, num_cameras=3)
    h = w = int(resize_image if resize_image is not None else variant.resize_image)

    cam_keys = ("cam_left", "cam_right", "cam_head")
    planes = []
    for ck in cam_keys:
        key = f"observation.images.{ck}"
        if key not in obs:
            raise KeyError(f"missing key {key}; got keys {list(obs.keys())}")
        hwc = _remote_image_to_hwc_uint8(obs[key])
        planes.append(np.asarray(PIL.Image.fromarray(hwc).resize((w, h))))

    curr_image = np.concatenate(planes, axis=2)

    if not getattr(variant, "add_states", True):
        return {"pixels": curr_image[np.newaxis, ..., np.newaxis]}

    st_key = "observation.state"
    if st_key not in obs:
        raise KeyError(f"missing key {st_key}")
    st = obs[st_key]
    if isinstance(st, torch.Tensor):
        st = st.detach().cpu().numpy()
    qpos = np.asarray(st, dtype=np.float32).reshape(-1)
    if qpos.size < 14:
        raise ValueError(f"observation.state length {qpos.size} < 14")
    qpos = qpos[:14]

    return {
        "pixels": curr_image[np.newaxis, ..., np.newaxis],
        "state": qpos[np.newaxis, ..., np.newaxis],
    }

def _checkpoint_step(path):
    """从目录名解析训练步数，例如 ``checkpoint_170000`` → 170000。"""
    name = os.path.basename(path.rstrip(os.sep))
    m = re.search(r"(?:checkpoint_?)(\d+)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0

def _list_checkpoint_dirs(exp_dir):
    """列出实验目录下所有 Flax ``checkpoint*`` 子目录（与 ``save_checkpoint(..., prefix='checkpoint')`` 一致）。"""
    if not os.path.isdir(exp_dir):
        return []
    out = []
    for name in os.listdir(exp_dir):
        if not name.startswith("checkpoint"):
            continue
        full = os.path.join(exp_dir, name)
        if os.path.isdir(full):
            out.append(full)
    return sorted(out, key=_checkpoint_step)

def eval_experiment(
    log_dir,
    output_dir="/root/storage/CODE/lx/dsrl_pi0_lx/examples/eval/output/eval_experiment",
    *,
    seed=0,
):
    """
    对 ``log_dir`` 下所有 checkpoint 依次做离线指标评估（``eval_agent`` 单次前向）。

    结果写入 ``output_dir/<时间戳>/``：每个 checkpoint 一份 ``.txt``、汇总 ``summary.json``、
    指标随步数曲线 ``metrics_curves.png``。
    """
    log_dir = os.path.abspath(log_dir)
    output_base = os.path.abspath(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, stamp)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_dirs = _list_checkpoint_dirs(log_dir)
    if not ckpt_dirs:
        raise FileNotFoundError(f"未在目录中找到 checkpoint 子目录: {log_dir}")

    summary_rows = []
    metric_keys_order = None

    for ckpt_path in ckpt_dirs:
        step = _checkpoint_step(ckpt_path)
        agent = init_agent(checkpoint_dir=ckpt_path, seed=seed)
        result = eval_agent(agent, seed=seed)
        meta_ev = result["meta"]
        metrics = result["metrics"]

        means = dict(metrics)
        stds = {k: 0.0 for k in metrics}
        if metric_keys_order is None:
            metric_keys_order = sorted(metrics.keys())

        row = {
            "step": step,
            "checkpoint_path": ckpt_path,
            "eval_meta": meta_ev,
            "metrics_mean": means,
            "metrics_std": stds,
        }
        summary_rows.append(row)

        txt_path = os.path.join(run_dir, f"checkpoint_{step}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"checkpoint_dir: {ckpt_path}\n")
            f.write(f"training_step: {step}\n\n")
            f.write("[meta]\n")
            for mk, mv in meta_ev.items():
                f.write(f"  {mk}: {mv}\n")
            f.write("\n[metrics]\n")
            for k in sorted(metrics.keys()):
                f.write(f"  {k}: {metrics[k]:.6f}\n")

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"log_dir": log_dir, "inference": "single_batch_per_checkpoint", "checkpoints": summary_rows},
            f,
            indent=2,
            ensure_ascii=False,
        )

    steps = np.array([r["step"] for r in summary_rows], dtype=np.float64)
    plot_keys = [
        k
        for k in ("critic_loss", "actor_loss", "temperature_loss", "entropy")
        if any(k in r["metrics_mean"] for r in summary_rows)
    ]
    if not plot_keys and metric_keys_order:
        plot_keys = metric_keys_order[: min(4, len(metric_keys_order))]

    if plot_keys:
        n = len(plot_keys)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), squeeze=False)
        for ax, key in zip(axes.flat, plot_keys):
            ys = [r["metrics_mean"].get(key, float("nan")) for r in summary_rows]
            ax.plot(steps, ys, marker="o")
            ax.set_ylabel(key)
            ax.set_xlabel("checkpoint step")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Offline eval metrics vs checkpoint step")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "metrics_curves.png"), dpi=150)
        plt.close(fig)

    print(f"[eval_experiment] 完成：共 {len(ckpt_dirs)} 个 checkpoint，输出目录:\n  {run_dir}")
    return run_dir

def analysis_actions(actions_dp, actions_sac):
    """
    对比两段动作序列，形状均为 ``(50, 14)``。

    指标：RMSE（整体幅度差异）、MAE（平均绝对误差）、mean_cos（逐步 14 维向量余弦相似度再对 50 步平均）。
    """
    a = np.asarray(actions_dp, dtype=np.float64)
    b = np.asarray(actions_sac, dtype=np.float64)
    if a.shape != (50, 14) or b.shape != (50, 14):
        raise ValueError(f"expected shape (50, 14), got {a.shape} and {b.shape}")

    diff = a - b
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))

    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    dots = np.sum(a * b, axis=1)
    cos_t = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 1e-12)
    both_small = (na < 1e-12) & (nb < 1e-12)
    cos_t = np.where(both_small, 1.0, cos_t)
    mean_cos = float(np.mean(cos_t))

    label_w, num_w = 18, 12
    line = " | ".join(
        [
            f"{'RMSE:':<{label_w}}{rmse:>{num_w}.6f}",
            f"{'MAE:':<{label_w}}{mae:>{num_w}.6f}",
            f"{'mean_cos:':<{label_w}}{mean_cos:>{num_w}.6f}",
        ]
    )
    print(line)
    return {"rmse": rmse, "mae": mae, "mean_cos": mean_cos}


# test functions
def test_agent():
    # eval_experiment("/root/storage/CODE/lx/dsrl_pi0_lx/logs/DSRL_pi0_Aloha/dsrl_pi0_aloha_2026_05_05_09_07_00_0000--s-0")
    agent = init_agent()
    obs = generate_rand_obs()
    noise = agent.sample_actions(obs) # nd.array, shape = [1, 14]
    print("noise", noise.shape, noise.dtype)
    print("noise", noise)
    return

def test_agent_dp():
    policy = init_agent_dp()
    obs_pkl = "/root/storage/CODE/lx/test/0509/pi0_obs/20260509_092433.pkl"
    obs = generate_local_obs(obs_pkl)
    out = policy.infer(obs=obs)
    print("actions", out["actions"].shape, out["actions"].dtype)
    print("policy_timing", out.get("policy_timing"))
    return out

def test_infer():
    obs_path = "/root/storage/CODE/lx/test/0509/pi0_obs/20260509_092433.pkl"
    obs = generate_local_obs(obs_path)
    agent = init_agent()
    agent_dp = init_agent_dp()
    
    actions = infer(agent, agent_dp, obs)
    print("actions", actions.shape, actions.dtype)
    print("actions", actions)
    return actions

def test_oepnpi_obs_to_dict():
    obs_path = "/root/storage/CODE/lx/test/0509/pi0_obs/20260509_092433.pkl"
    obs = generate_local_obs(obs_path)
    dict_obs = oepnpi_obs2dict(obs)
    ref = generate_rand_obs()
    print("dict_obs keys:", dict_obs.keys())
    print("pixels", dict_obs["pixels"].shape, dict_obs["pixels"].dtype)
    print("state", dict_obs["state"].shape, dict_obs["state"].dtype)
    assert dict_obs["pixels"].shape == ref["pixels"].shape
    assert dict_obs["state"].shape == ref["state"].shape
    return dict_obs

def test_analysis_actions():
    rng = np.random.default_rng(0)
    actions_dp = rng.standard_normal((50, 14)).astype(np.float32)
    actions_sac = rng.standard_normal((50, 14)).astype(np.float32)
    analysis_actions(actions_dp, actions_sac)


if __name__ == "__main__":
    test_agent()
    # main()

import functools
import json
import os

from tqdm import tqdm
import numpy as np
import wandb
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from openpi_client import image_tools
import math
import PIL

from jaxrl2.data.augmentations import batched_random_crop, color_transform

import cv2
import pyarrow.parquet as pq


def generate_traj(
    pixel_H=64,
    pixel_W=64,
    pixel_C=9,
    state_dim=14,
    action_dim=14,
    images=False,
    *,
    success_rate=0.5,
    query_frequency=50,
    max_timesteps=4000,
    env_max_reward=1,
    action_chunk_horizon=1,
    rng=None,
):
    """Synthetic rollout matching ``collect_traj`` keys/shapes for testing."""
    rng = np.random.default_rng(rng)
    env_steps = int(rng.integers(query_frequency, max_timesteps + 1))
    query_steps = (env_steps - 1) // query_frequency + 1

    raw_rewards = rng.standard_normal(env_steps).astype(np.float32)
    is_success = rng.random() < success_rate
    if is_success:
        raw_rewards[-1] = np.float32(env_max_reward)
    else:
        raw_rewards[-1] = np.float32(rng.choice([-1.0, 0.0]))
        if raw_rewards[-1] == env_max_reward:
            raw_rewards[-1] = np.float32(env_max_reward - 1e-3)
    episode_return = float(np.sum(raw_rewards))

    def _obs_dict():
        pix = rng.random((1, pixel_H, pixel_W, pixel_C, 1), dtype=np.float32)
        st = rng.standard_normal((1, state_dim, 1)).astype(np.float32)
        return {'pixels': pix, 'state': st}

    obs_list = [_obs_dict() for _ in range(query_steps)]
    obs_list.append(_obs_dict())
    action_list = [
        rng.standard_normal((action_chunk_horizon, action_dim)).astype(np.float32)
        for _ in range(query_steps)
    ]

    if is_success:
        rewards = np.concatenate([-np.ones(query_steps - 1, dtype=np.float32), [0.0]])
        masks = np.concatenate([np.ones(query_steps - 1, dtype=np.float32), [0.0]])
    else:
        rewards = -np.ones(query_steps, dtype=np.float32)
        masks = np.ones(query_steps, dtype=np.float32)

    out = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'env_steps': env_steps,
    }
    if images:
        c_vis = 3 if pixel_C >= 3 else pixel_C
        out['images'] = [
            (rng.random((pixel_H, pixel_W, c_vis), dtype=np.float32) * 255).astype(np.uint8)
            for _ in range(env_steps + 1)
        ]
    return out

def generate_trajs(num_trajs=100):
    """返回多条合成轨迹列表；``rng=episode_index`` 以便每条轨迹可复现（勿把 index 当作 ``pixel_H`` 传入）。"""
    trajs = []
    for episode_index in range(num_trajs):
        trajs.append(generate_traj(rng=episode_index))
    return trajs

def _lerobot_episode_parquet(data_root: str, episode_index: int) -> str:
    return os.path.join(data_root, "data", "chunk-000", f"episode_{episode_index:06d}.parquet")


def _lerobot_video_path(data_root: str, episode_index: int, camera: str) -> str:
    return os.path.join(
        data_root,
        "videos",
        "chunk-000",
        f"observation.images.{camera}",
        f"episode_{episode_index:06d}.mp4",
    )


def _decode_video_rgb_all_frames(video_path: str):
    """Decode all frames as RGB uint8 (H, W, 3). Same stack as analysis.py (PyAV for AV1)."""
    try:
        import av
    except ImportError as e:
        raise ImportError("Reading LeRobot videos requires PyAV: pip install av") from e

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = []
    container = av.open(video_path)
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _resize_rgb_cv2(img_rgb: np.ndarray, pixel_h: int, pixel_w: int) -> np.ndarray:
    if cv2 is None:
        raise ImportError("cv2 is required for resize: pip install opencv-python-headless")
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    return cv2.resize(img_rgb, (pixel_w, pixel_h), interpolation=cv2.INTER_AREA)


def _fuse_three_cams_rgb(
    left: np.ndarray,
    right: np.ndarray,
    head: np.ndarray,
    pixel_h: int,
    pixel_w: int,
) -> np.ndarray:
    """Concatenate left / right / head along channel (same idea as ``process_images`` in train_utils_real)."""
    im1 = _resize_rgb_cv2(left, pixel_h, pixel_w)
    im2 = _resize_rgb_cv2(right, pixel_h, pixel_w)
    im3 = _resize_rgb_cv2(head, pixel_h, pixel_w)
    return np.concatenate([im1, im2, im3], axis=2)


def _build_episode_traj(
    data_root: str,
    episode_index: int,
    *,
    pixel_h: int,
    pixel_w: int,
    state_key: str,
    action_key: str,
    cameras: tuple,
    env_max_reward: float,
    action_chunk_horizon: int,
):
    parquet_path = _lerobot_episode_parquet(data_root, episode_index)
    if pq is None:
        raise ImportError("Reading LeRobot parquet requires pyarrow: pip install pyarrow")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    df = pq.ParquetFile(parquet_path).read().to_pandas()
    env_steps = len(df)
    if env_steps == 0:
        return None

    video_paths = {cam: _lerobot_video_path(data_root, episode_index, cam) for cam in cameras}
    frames_per_cam = {cam: _decode_video_rgb_all_frames(video_paths[cam]) for cam in cameras}
    n_vid = min(len(frames_per_cam[c]) for c in cameras)
    if n_vid < env_steps:
        raise RuntimeError(
            f"Episode {episode_index}: video frames ({n_vid}) < parquet rows ({env_steps})"
        )

    raw_rewards = np.zeros(env_steps, dtype=np.float32)
    raw_rewards[-1] = np.float32(env_max_reward)
    episode_return = float(np.sum(raw_rewards))
    is_success = True

    # Real-robot LeRobot data: one (obs, action) per recorded frame — no query-frequency subsampling
    # (simulation ``collect_traj`` still uses ``variant.query_freq``).
    obs_list = []
    action_list = []

    state_dim = int(np.asarray(df[state_key].iloc[0]).reshape(-1).shape[0])
    action_dim = int(np.asarray(df[action_key].iloc[0]).reshape(-1).shape[0])

    def _obs_at_frame(t: int):
        fused = _fuse_three_cams_rgb(
            frames_per_cam[cameras[0]][t],
            frames_per_cam[cameras[1]][t],
            frames_per_cam[cameras[2]][t],
            pixel_h,
            pixel_w,
        )
        st = np.asarray(df[state_key].iloc[t], dtype=np.float32).reshape(-1)
        return {
            "pixels": fused[np.newaxis, ..., np.newaxis],
            "state": st.reshape(1, state_dim, 1).astype(np.float32),
        }

    for t in range(env_steps):
        obs_list.append(_obs_at_frame(t))
        a = np.asarray(df[action_key].iloc[t], dtype=np.float32).reshape(action_chunk_horizon, action_dim)
        action_list.append(a)

    obs_list.append(_obs_at_frame(env_steps - 1))

    # Dense-style sparse success: 0 on all steps except the last transition, which gets env_max_reward.
    # (``collect_traj`` for sim instead uses -1/0 as SAC shaping; real demos use 0/1 here.)
    rewards = np.concatenate(
        [
            np.zeros(env_steps - 1, dtype=np.float32),
            np.array([np.float32(env_max_reward)], dtype=np.float32),
        ]
    )
    # mask=1: non-terminal (bootstrap V(s') in Bellman); mask=0: terminal transition (no bootstrap).
    masks = np.concatenate([np.ones(env_steps - 1, dtype=np.float32), [0.0]])

    images = []
    for t in range(env_steps):
        images.append(
            _fuse_three_cams_rgb(
                frames_per_cam[cameras[0]][t],
                frames_per_cam[cameras[1]][t],
                frames_per_cam[cameras[2]][t],
                pixel_h,
                pixel_w,
            )
        )
    images.append(
        _fuse_three_cams_rgb(
            frames_per_cam[cameras[0]][env_steps - 1],
            frames_per_cam[cameras[1]][env_steps - 1],
            frames_per_cam[cameras[2]][env_steps - 1],
            pixel_h,
            pixel_w,
        )
    )

    return {
        "observations": obs_list,
        "actions": action_list,
        "rewards": rewards,
        "masks": masks,
        "is_success": is_success,
        "episode_return": episode_return,
        "env_steps": env_steps,
        "images": images,
    }


def read_local_trajs(
    data_path,
    *,
    pixel_h=64,
    pixel_w=64,
    state_key="observation.state",
    action_key="action",
    cameras=("cam_left", "cam_right", "cam_head"),
    env_max_reward=1.0,
    action_chunk_horizon=1,
    max_episodes=None,
):
    """
    Load LeRobot v2.x episodes from ``data_path`` (dataset root with ``data/``, ``videos/``, ``meta/``).

    Uses **every recorded frame** (no ``query_freq`` subsampling; that applies only to simulation
    ``collect_traj``). Trajectory layout matches the buffer contract: ``len(actions) == env_steps``,
    ``len(observations) == env_steps + 1`` (terminal observation appended like ``collect_traj``).

    Fused RGB cameras along channel (9 ch), resized with OpenCV.

    **Rewards:** per transition, ``0`` then ``env_max_reward`` (default 1) on the **last** transition
    only — not the simulation ``-1``/``0`` SAC shaping in ``collect_traj``.

    **Masks:** ``1`` for all but the last transition; the last is ``0`` (episode done → no bootstrapping
    from a fictitious next state in the Bellman target).
    """
    data_root = os.path.expanduser(str(data_path))
    meta_info = os.path.join(data_root, "meta", "info.json")
    if not os.path.isfile(meta_info):
        raise FileNotFoundError(f"Not a LeRobot dataset root (missing meta/info.json): {data_root}")

    with open(meta_info, "r") as f:
        info = json.load(f)
    total = int(info["total_episodes"])
    n_ep = total if max_episodes is None else min(total, int(max_episodes))

    trajs = []
    for episode_index in tqdm(range(n_ep), desc="read_local_trajs"):
        trajs.append(
            _build_episode_traj(
                data_root,
                episode_index,
                pixel_h=pixel_h,
                pixel_w=pixel_w,
                state_key=state_key,
                action_key=action_key,
                cameras=cameras,
                env_max_reward=env_max_reward,
                action_chunk_horizon=action_chunk_horizon,
            )
        )
    return trajs


@functools.partial(
    jax.jit,
    static_argnames=("critic_reduction", "color_jitter", "aug_next", "num_cameras"),
)
def offline_eval_metrics_jit(
    rng,
    actor,
    critic,
    target_critic_params,
    temp,
    batch,
    target_entropy,
    critic_reduction,
    color_jitter,
    aug_next,
    num_cameras,
):
    """
    Forward-only SAC diagnostics matching ``PixelSACLearner.update`` augmentations.
    Does **not** apply gradients — safe for checkpoint evaluation.
    """
    aug_pixels = batch["observations"]["pixels"]
    aug_next_pixels = batch["next_observations"]["pixels"]
    if batch["observations"]["pixels"].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:, :, :, i * 3 : (i + 1) * 3].set(
                        (
                            color_transform(
                                key,
                                aug_pixels[:, :, :, i * 3 : (i + 1) * 3].astype(jnp.float32) / 255.0,
                            )
                            * 255.0
                        ).astype(jnp.uint8)
                    )
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32) / 255.0) * 255.0).astype(
                    jnp.uint8
                )

    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch_aug = batch.copy(add_or_replace={"observations": observations})

    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:, :, :, i * 3 : (i + 1) * 3].set(
                        (
                            color_transform(
                                key,
                                aug_next_pixels[:, :, :, i * 3 : (i + 1) * 3].astype(jnp.float32) / 255.0,
                            )
                            * 255.0
                        ).astype(jnp.uint8)
                    )
            else:
                aug_next_pixels = (
                    color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.0) * 255.0
                ).astype(jnp.uint8)
        next_observations = batch["next_observations"].copy(add_or_replace={"pixels": aug_next_pixels})
        batch_aug = batch_aug.copy(add_or_replace={"next_observations": next_observations})

    target_critic = critic.replace(params=target_critic_params)

    rng, key_c = jax.random.split(rng)
    dist_next = actor.apply_fn({"params": actor.params}, batch_aug["next_observations"])
    next_actions, next_log_probs = dist_next.sample_and_log_prob(seed=key_c)
    next_qs = target_critic.apply_fn(
        {"params": target_critic.params}, batch_aug["next_observations"], next_actions
    )
    if critic_reduction == "min":
        next_q = next_qs.min(axis=0)
    elif critic_reduction == "mean":
        next_q = next_qs.mean(axis=0)
    else:
        raise ValueError(f"Invalid critic_reduction: {critic_reduction}")

    target_q = batch_aug["rewards"] + batch_aug["discount"] * batch_aug["masks"] * next_q

    # Match ``update_critic``: Q(s,a) uses params only (no critic batch_stats).
    qs = critic.apply_fn({"params": critic.params}, batch_aug["observations"], batch_aug["actions"])
    critic_loss = ((qs - target_q) ** 2).mean()
    abs_td = jnp.abs(qs - target_q)

    rng, key_a = jax.random.split(rng)
    if hasattr(actor, "batch_stats") and actor.batch_stats is not None:
        act_out = actor.apply_fn(
            {"params": actor.params, "batch_stats": actor.batch_stats},
            batch_aug["observations"],
            mutable=["batch_stats"],
        )
        dist = act_out[0] if isinstance(act_out, tuple) else act_out
    else:
        dist = actor.apply_fn({"params": actor.params}, batch_aug["observations"])
    actions_pi, log_probs = dist.sample_and_log_prob(seed=key_a)

    if hasattr(critic, "batch_stats") and critic.batch_stats is not None:
        qs_pi, _ = critic.apply_fn(
            {"params": critic.params, "batch_stats": critic.batch_stats},
            batch_aug["observations"],
            actions_pi,
            mutable=["batch_stats"],
        )
    else:
        qs_pi = critic.apply_fn(
            {"params": critic.params}, batch_aug["observations"], actions_pi
        )

    if critic_reduction == "min":
        q_pi = qs_pi.min(axis=0)
    elif critic_reduction == "mean":
        q_pi = qs_pi.mean(axis=0)
    else:
        raise ValueError(f"Invalid critic_reduction: {critic_reduction}")

    temperature = temp.apply_fn({"params": temp.params})
    entropy = -log_probs.mean()
    actor_loss = (log_probs * temperature - q_pi).mean()
    temp_loss = temperature * (entropy - target_entropy)

    return {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "temperature_loss": temp_loss,
        "temperature": temperature,
        "entropy": entropy,
        "q_at_actions_mean": qs.mean(),
        "q_pi_mean": q_pi.mean(),
        "target_q_mean": target_q.mean(),
        "abs_td_error_mean": abs_td.mean(),
        "next_q_mean": next_q.mean(),
    }


def _strip_leading_batch(obs):
    """``ReplayBuffer`` stores observations without batch dim (see ``add_online_data_to_buffer``)."""
    out = {}
    for k, v in obs.items():
        v = np.asarray(v)
        out[k] = v[0] if v.ndim >= 1 and v.shape[0] == 1 else v
    return out


def transitions_from_generate_traj(traj, *, discount_pow: float):
    """Expand one synthetic traj from ``generate_traj`` into transition dicts (replay layout)."""
    T = len(traj["actions"])
    out = []
    for t in range(T):
        obs = _strip_leading_batch(traj["observations"][t])
        next_obs = _strip_leading_batch(traj["observations"][t + 1])
        out.append(
            {
                "observations": obs,
                "next_observations": next_obs,
                "actions": np.asarray(traj["actions"][t], dtype=np.float32),
                "next_actions": np.asarray(
                    traj["actions"][t + 1] if t < T - 1 else traj["actions"][t], dtype=np.float32
                ),
                "rewards": float(traj["rewards"][t]),
                "masks": float(traj["masks"][t]),
                "discount": float(discount_pow),
            }
        )
    return out


def transitions_from_lerobot_traj(traj, *, discount_pow: float):
    """Same layout as ``transitions_from_generate_traj`` for ``read_local_trajs`` episodes."""
    T = len(traj["actions"])
    out = []
    for t in range(T):
        obs = _strip_leading_batch(traj["observations"][t])
        next_obs = _strip_leading_batch(traj["observations"][t + 1])
        out.append(
            {
                "observations": obs,
                "next_observations": next_obs,
                "actions": np.asarray(traj["actions"][t], dtype=np.float32),
                "next_actions": np.asarray(
                    traj["actions"][t + 1] if t < T - 1 else traj["actions"][t], dtype=np.float32
                ),
                "rewards": float(traj["rewards"][t]),
                "masks": float(traj["masks"][t]),
                "discount": float(discount_pow),
            }
        )
    return out


def stack_transitions(transitions):
    """Stack a list of single transitions into a batched numpy dict (replay ``sample`` layout)."""
    obs_keys = transitions[0]["observations"].keys()
    batched = {
        "observations": {
            k: np.stack([tr["observations"][k] for tr in transitions], axis=0) for k in obs_keys
        },
        "next_observations": {
            k: np.stack([tr["next_observations"][k] for tr in transitions], axis=0) for k in obs_keys
        },
        "actions": np.stack([tr["actions"] for tr in transitions], axis=0),
        "next_actions": np.stack([tr["next_actions"] for tr in transitions], axis=0),
        "rewards": np.array([tr["rewards"] for tr in transitions], dtype=np.float32),
        "masks": np.array([tr["masks"] for tr in transitions], dtype=np.float32),
        "discount": np.array([tr["discount"] for tr in transitions], dtype=np.float32),
    }
    return batched


def eval_agent(
    agent,
    *,
    batch_size=256,
    seed=0,
    query_freq=50,
    discount=0.999,
    data_root=None,
    max_episodes_for_dataset=4,
    pool_episodes=64,
):
    """
    单次 JIT 前向（仅 1 个 mini-batch、无梯度）：与训练一致的 critic/actor/temperature 标量指标。

    不在此函数内打印；返回 ``{"meta": ..., "metrics": ...}``，由调用方负责日志或落盘。
    可选环境变量：OFFLINE_EVAL_BATCH_SIZE、OFFLINE_EVAL_SEED、OFFLINE_EVAL_QUERY_FREQ、
    OFFLINE_EVAL_DISCOUNT、OFFLINE_EVAL_DATA_ROOT、OFFLINE_EVAL_MAX_EPISODES。
    """
    batch_size = int(os.environ.get("OFFLINE_EVAL_BATCH_SIZE", batch_size))
    seed = int(os.environ.get("OFFLINE_EVAL_SEED", seed))
    query_freq = int(os.environ.get("OFFLINE_EVAL_QUERY_FREQ", query_freq))
    discount = float(os.environ.get("OFFLINE_EVAL_DISCOUNT", discount))
    data_root = os.environ.get("OFFLINE_EVAL_DATA_ROOT", data_root)
    max_episodes_for_dataset = int(
        os.environ.get("OFFLINE_EVAL_MAX_EPISODES", max_episodes_for_dataset)
    )

    discount_pow = float(np.power(discount, query_freq))
    rng_py = np.random.default_rng(seed)

    transitions = []
    dataset_used = "synthetic (generate_traj)"

    root = data_root if data_root else os.environ.get("LEROBOT_DATA_ROOT")
    if root and os.path.isfile(os.path.join(os.path.expanduser(str(root)), "meta", "info.json")):
        trajs = read_local_trajs(
            root,
            max_episodes=max_episodes_for_dataset,
            pixel_h=64,
            pixel_w=64,
        )
        for traj in trajs:
            transitions.extend(transitions_from_lerobot_traj(traj, discount_pow=discount_pow))
        dataset_used = f"LeRobot: {root} (max_episodes={max_episodes_for_dataset})"
    else:
        for ep in range(pool_episodes):
            transitions.extend(
                transitions_from_generate_traj(
                    generate_traj(rng=seed + ep, query_frequency=query_freq),
                    discount_pow=discount_pow,
                )
            )

    if len(transitions) < batch_size:
        raise RuntimeError(
            f"Not enough transitions ({len(transitions)}) for batch_size={batch_size}. "
            "Increase pool_episodes or max_episodes_for_dataset."
        )

    rng_jax = jax.random.PRNGKey(seed)
    idx = rng_py.choice(len(transitions), size=batch_size, replace=True)
    batch_np = stack_transitions([transitions[i] for i in idx])
    batch_jax = frozen_dict.freeze(jax.tree_util.tree_map(jnp.asarray, batch_np))

    rng_jax, sub = jax.random.split(rng_jax)
    info = offline_eval_metrics_jit(
        sub,
        agent._actor,
        agent._critic,
        agent._target_critic_params,
        agent._temp,
        batch_jax,
        agent.target_entropy,
        agent.critic_reduction,
        agent.color_jitter,
        agent.aug_next,
        agent.num_cameras,
    )
    info = jax.device_get(info)

    metrics = {k: float(info[k]) for k in sorted(info.keys())}
    meta = {
        "dataset_used": dataset_used,
        "pool_size": len(transitions),
        "batch_size": batch_size,
        "discount_pow": discount_pow,
        "query_freq": query_freq,
        "discount": discount,
        "seed": seed,
        "inference_batches": 1,
    }
    return {"meta": meta, "metrics": metrics}


def test_read_local_trajs():
    # Usage: conda activate dsrl_lx && python examples/train_utils_sim.py
    # Optional: LEROBOT_DATA_ROOT=/path/to/dataset
    default_dataset = os.environ.get(
        "LEROBOT_DATA_ROOT",
        "/root/storage/CODE/txy/dsrl_pi0_lx/openpi/dataset/0316_pouring_water_easy",
    )

    print("=== generate_traj sanity check ===")
    traj_syn = generate_traj(rng=0)
    print(f"  len(observations)={len(traj_syn['observations'])}, len(actions)={len(traj_syn['actions'])}")
    print(f"  pixels shape {traj_syn['observations'][0]['pixels'].shape}")
    print(f"  state shape {traj_syn['observations'][0]['state'].shape}")
    print(f"  rewards {traj_syn['rewards']}, masks {traj_syn['masks']}")
    print(f"  episode_return={traj_syn['episode_return']:.4f}, is_success={traj_syn['is_success']}")

    print("\n=== read_local_trajs (max_episodes=1) ===")
    meta_path = os.path.join(default_dataset, "meta", "info.json")
    if os.path.isfile(meta_path):
        trajs = read_local_trajs(default_dataset, max_episodes=100, pixel_h=64, pixel_w=64)
        t0 = trajs[0]
        print(f"  dataset: {default_dataset}")
        print(f"  env_steps={t0['env_steps']}")
        print(f"  len(observations)={len(t0['observations'])}, len(actions)={len(t0['actions'])}")
        print(f"  pixels shape {t0['observations'][0]['pixels'].shape}")
        print(f"  state shape {t0['observations'][0]['state'].shape}")
        print(f"  action[0] shape {t0['actions'][0].shape}")
        print(f"  rewards {t0['rewards']}, masks {t0['masks']}")
        print(f"  episode_return={t0['episode_return']:.4f}, is_success={t0['is_success']}")
        print(f"  len(images)={len(t0['images'])}, fused frame shape {t0['images'][0].shape}")
    else:
        print(f"  skipped: no dataset at {default_dataset} (set LEROBOT_DATA_ROOT)")

if __name__ == "__main__":
    test_read_local_trajs()
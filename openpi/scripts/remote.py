import time
import numpy as np
import torch
from typing import Dict, List, Union
from jsonargparse import ArgumentParser
from termcolor import cprint
import uvicorn
import json
from pathlib import Path
import dataclasses
from dataclasses import dataclass
from torch import Tensor
from collections import deque

from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.transforms as _transforms

app = FastAPI()

from examples.eval.eval import init_agent, oepnpi_obs2dict, analysis_actions


@dataclass(frozen=True)
class EvaluateConfig:
    defalut_ckpt_path: str = "/data/ckpt/openpi/pretrain/pi0_base_jax"
    ckpt_path: str = "/data/ckpt/openpi/pi0_airbot_lora/stack_bowls_right_0922/2999"

    camera_names: Union[str, List[str]] = "cam_head cam_left cam_right"
    task_description: str = "something to do"

    config: Union[str, object] = "pi0_aloha"  # 这里保持 Union 以兼容解析前后的状态
    asset_id: Union[str, None] = None

    balancing_factor: float = None
    temporal_size: int = 20
    skip_frame: int = 0

    model_test: bool = False
    load_model: bool = True
    port: int = 6160

    def __post_init__(self):
        # 1. 处理 camera_names (只在它是字符串时才 split)
        if isinstance(self.camera_names, str):
            object.__setattr__(self, "camera_names", self.camera_names.split())

        # 定义 assets 的基础路径
        assets_base_path = Path(self.ckpt_path) / "assets"

        # 2. 自动搜索 asset_id
        # 如果主人您懒得输入 asset_id，我就去 assets_base_path 下面找第一个文件夹
        if self.asset_id is None:
            if assets_base_path.exists() and assets_base_path.is_dir():
                # 优先找文件夹
                subdirs = [
                    d.name
                    for d in assets_base_path.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
                if subdirs:
                    object.__setattr__(self, "asset_id", subdirs[0])
                else:
                    # 没文件夹就找文件
                    items = [
                        i.name
                        for i in assets_base_path.iterdir()
                        if not i.name.startswith(".")
                    ]
                    if items:
                        object.__setattr__(self, "asset_id", items[0])

            if self.asset_id is None:
                cprint(
                    f"[Warning] 找不到 asset 标识: {assets_base_path} 是空的？主人您在逗我吗？",
                    "yellow",
                )

        # 3. 解析 config 对象
        if isinstance(self.config, str):
            try:
                parsed_config = _config.get_config(self.config)
                object.__setattr__(self, "config", parsed_config)
            except NameError:
                cprint(
                    "[Error] 找不到 _config，主人您是不是忘了 import 那个解析工具？",
                    "red",
                )
                return

        # 4. 属性注入：将路径和 ID 优雅地塞进 config 对象
        if self.config and not isinstance(self.config, str):
            try:
                # 按照主人要求的，把 assets 的父目录存进去
                object.__setattr__(
                    self.config, "assets_base_dir", str(assets_base_path)
                )

                # asset_id 依然保持为具体的标识符（文件夹/文件名）
                if self.asset_id is not None:
                    # 遵循之前的结构设置到 data.assets 中
                    object.__setattr__(
                        self.config.data.assets, "asset_id", self.asset_id
                    )

            except AttributeError as e:
                cprint(
                    f"[Warning] 注入失败：{e}。主人的 config 对象结构可能又改了，真麻烦。",
                    "yellow",
                )


class TemporalEnsembler:
    def __init__(self, config: EvaluateConfig) -> None:
        self.use_temporal = True if config.balancing_factor is not None else False

        if config.balancing_factor is None:
            self._action_queue = deque([], maxlen=config.temporal_size)
            self.temporal_size = config.temporal_size
            self.skip_frame = config.skip_frame
        else:
            cprint("Now is using TemporalEnsembler", "cyan")
            self.temporal_size = config.temporal_size
            self.skip_frame = config.skip_frame

            self.temporal_mask = torch.flip(
                torch.triu(
                    torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)
                ),
                dims=[1],
            ).numpy()

            self.action_buffer = np.zeros(
                (
                    self.temporal_mask.shape[0],
                    self.temporal_mask.shape[0],
                    14,  # action_dim
                )
            )
            self.action_buffer_mask = np.zeros(
                (self.temporal_mask.shape[0], self.temporal_mask.shape[0]),
                dtype=np.bool_,
            )

            # Action chunking with temporal aggregation
            self.temporal_weights = np.array(
                [
                    np.exp(-1 * config.balancing_factor * i)
                    for i in range(self.temporal_size)
                ]
            )[:, None]

    def reset(self):
        if self.use_temporal is True:
            self.action_buffer = np.zeros(
                (
                    self.temporal_mask.shape[0],
                    self.temporal_mask.shape[0],
                    14,  # action_dim
                )
            )
            self.action_buffer_mask = np.zeros(
                (self.temporal_mask.shape[0], self.temporal_mask.shape[0]),
                dtype=np.bool_,
            )
        else:
            self._action_queue.clear()

    def update(self, actions: Union[Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(actions, Tensor):
            actions = actions.cpu().numpy()

        if self.use_temporal is True:
            pred_action = np.array(
                actions[
                    self.skip_frame : self.temporal_size + self.skip_frame, :
                ].tolist()
            )

            # 往后挪动一格
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

            # 添加新的动作
            self.action_buffer[0] = pred_action
            self.action_buffer_mask[0] = np.array(
                [True] * self.temporal_mask.shape[0], dtype=np.bool_
            )

            # Ensemble temporally to predict action
            action_prediction = np.sum(
                self.action_buffer[:, 0, :]
                * self.action_buffer_mask[:, 0:1]
                * self.temporal_weights,
                axis=0,
            ) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)

            return action_prediction
        else:
            return actions[self.skip_frame : self.temporal_size + self.skip_frame, :]


class ImageProcessingPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(ImageProcessingPipeline, self).__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=12,
            exec_pipelined=False,
            exec_async=False,
        )

        self.source = ops.ExternalSource(name="DALI_INPUT_JPEGS", device="cpu")
        self.decode = fn.decoders.image
        # --- 定义操作 ---
        self.source = ops.ExternalSource(name="DALI_INPUT_JPEGS", device="cpu")
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.cast = ops.Cast(device="gpu", dtype=types.FLOAT)
        self.normalize = ops.Normalize(
            device="gpu",
            mean=np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3),
            stddev=np.array([255.0, 255.0, 255.0], dtype=np.float32).reshape(1, 1, 3),
        )
        self.transpose = ops.Transpose(device="gpu", perm=[2, 0, 1])

    def define_graph(self):
        jpegs = self.source()
        images = self.decode(jpegs)
        images_casted = self.cast(images)
        images_normalized = self.normalize(images_casted)
        images_transposed = self.transpose(images_normalized)
        return images_transposed


def adjust_dict(app: FastAPI, qpos: str, image_bytes_list: List[bytes]) -> Dict:
    opt = app.state.opt
    dali_pipeline = app.state.dali_pipeline

    if len(image_bytes_list) != dali_pipeline.max_batch_size:
        return None

    numpy_images = [
        np.frombuffer(img_bytes, dtype=np.uint8) for img_bytes in image_bytes_list
    ]
    dali_pipeline.feed_input("DALI_INPUT_JPEGS", numpy_images)

    (processed_images_gpu,) = dali_pipeline.run()
    torch_tensors_cpu = []
    for dali_tensor_gpu in processed_images_gpu:
        dali_tensor_cpu = dali_tensor_gpu.as_cpu()
        numpy_array = np.array(dali_tensor_cpu)
        torch_tensor = torch.from_numpy(numpy_array)
        torch_tensors_cpu.append(torch_tensor)

    images_tensor_cpu = torch.stack(torch_tensors_cpu)

    obs = {}
    for i, cam_name in enumerate(opt.camera_names):
        obs[f"observation.images.{cam_name}"] = images_tensor_cpu[i]

    qpos_data = json.loads(qpos)
    obs["observation.state"] = torch.tensor(qpos_data, dtype=torch.float32)
    obs["prompt"] = opt.task_description

    return obs


def debug_image(images: List[UploadFile], image_bytes_list: List[bytes]):
    import os
    from datetime import datetime
    from PIL import Image
    import io

    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    DEBUG_DIR = os.path.join(script_dir, "..", "debug_images")

    os.makedirs(DEBUG_DIR, exist_ok=True)

    # ==================== 增强版 DEBUG 代码区 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cprint(f"[*] [DEBUG] Request Timestamp: {timestamp}", "yellow")

    # --- 1. 保存原始文件 (仍然保留，用于存档) ---
    cprint(f"[*] Saving raw image bytes...", "yellow")
    for i, img_bytes in enumerate(image_bytes_list):
        original_filename = images[i].filename
        debug_filename = os.path.join(
            DEBUG_DIR, f"{timestamp}_img_{i}_{original_filename}"
        )
        with open(debug_filename, "wb") as f:
            f.write(img_bytes)
        cprint(f"    - Saved raw: {debug_filename} ({len(img_bytes)} bytes)", "yellow")

    # --- 2. 使用 Pillow 进行内部属性“法医鉴定” ---
    cprint(f"[*] Inspecting image properties with Pillow...", "cyan")
    is_consistent = True
    first_image_props = None

    for i, img_bytes in enumerate(image_bytes_list):
        try:
            image = Image.open(io.BytesIO(img_bytes))
            props = {"mode": image.mode, "size": image.size}

            cprint(
                f"    - Image {i} ({images[i].filename}): Mode={props['mode']}, Size={props['size']}",
                "cyan",
            )

            if i == 0:
                first_image_props = props
            elif props != first_image_props:
                is_consistent = False
                cprint(f"    [!!!] Inconsistency FOUND!", "red", attrs=["bold"])
                cprint(f"          - Image 0 props: {first_image_props}", "red")
                cprint(f"          - Image {i} props: {props}", "red")

        except Exception as e:
            cprint(
                f"    [!!!] Image {i} ({images[i].filename}) is CORRUPTED!",
                "red",
                attrs=["bold"],
            )
            cprint(f"          Pillow failed to open it: {e}", "red")
            is_consistent = False

    if not is_consistent:
        cprint(
            "[*] [CONCLUSION] The batch of images is NOT consistent or is corrupted. DALI is likely to fail.",
            "red",
        )
    else:
        cprint(
            "[*] [CONCLUSION] Images appear consistent. If DALI still fails, the issue is more subtle.",
            "green",
        )


def print_action_debug(actions, color="yellow"):
    """
    专门用于打印 Action 的调试函数。
    不管输入是 Tensor 还是 Numpy，不管 1D 还是 2D，都能打得漂漂亮亮。
    """
    # 数据清洗：Tensor -> Numpy (CPU)
    if isinstance(actions, torch.Tensor):
        _act = actions.detach().cpu().numpy()
    else:
        _act = np.array(actions)

    # 维度统一：强制转为 2D (N, D)
    _act = np.atleast_2d(_act)

    # 格式化字符串构造
    content = ",\n".join(
        [f"  [{', '.join([f'{v:.3f}' for v in row])}]" for row in _act]
    )

    cprint(f"[*] Get action: [\n{content}\n]", color)


@app.post("/predict")
async def predict(
    app_handle: Request,
    qpos: str = Form(...),
    images: List[UploadFile] = File(...),
):
    try:
        app = app_handle.app
        policy = app.state.policy
        temporal = app.state.temporal

        image_bytes_list = [await file.read() for file in images]
        obs = adjust_dict(app, qpos, image_bytes_list)

        if obs is None:
            return JSONResponse(
                content={"success": False, "error": "Wrong number of pictures!"},
                status_code=400,
            )

        agent = app.state.sac_agent
        obs_dict = oepnpi_obs2dict(obs)
        noise = np.asarray(agent.sample_actions(obs_dict))
        noise = np.pad(noise, [(0, 0)] * (noise.ndim - 1) + [(0, max(0, 32 - noise.shape[-1]))], mode="constant")
        noise = np.repeat(noise[:, np.newaxis, :], 50, axis=1)  # Pi0 action_horizon=50：同一噪声复制 50 份
        raw_actions = policy.infer(obs=obs, noise=noise)
        raw_actions_dp = policy.infer(obs=obs)
        analysis_actions(raw_actions_dp["actions"], raw_actions["actions"])

        actions = temporal.update(raw_actions["actions"])

        print_action_debug(actions)

        return JSONResponse(content={"success": True, "actions": actions.tolist()})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.get("/health")  # 测试耗时 4-5ms
async def health_check():
    return JSONResponse(content={"status": "ok"})


@app.post("/cost_time")
async def data_cost_check(
    app_handle: Request,
    qpos: str = Form(...),
    images: List[UploadFile] = File(...),
):
    return JSONResponse(content={"status": "ok"})


@app.post("/clear_cache")
async def clear_cache(request: Request):
    try:
        policy = app.state.policy
        opt: EvaluateConfig = app.state.opt
        temporal = app.state.temporal

        task_description = None
        try:
            data = await request.json()
            task_description = data.get("task_description")
        except Exception as json_error:
            cprint(
                f"No JSON payload detected (or parsing failed): {json_error}. This is likely normal.",
                "magenta",
            )

        if task_description:
            cprint(f"New task received, resetting: {task_description}", "cyan")
            new_opt = dataclasses.replace(opt, task_description=task_description)
            app.state.opt = new_opt
        else:
            cprint("No new tasks received, only cache cleared.", "cyan")
        temporal.reset()

        return JSONResponse(content={"status": "ok"})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


def load_model(app: FastAPI, opt: EvaluateConfig):
    policy = _policy_config.create_trained_policy(
        opt.config,
        opt.ckpt_path,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
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
        default_prompt=opt.task_description,
    )
    cprint(f"Load VLM model from {opt.ckpt_path}", "magenta")

    app.state.policy = policy
    app.state.temporal = TemporalEnsembler(opt)
    app.state.sac_agent = init_agent()

    # total_params = sum(param.size for param in policy._model.llm.params.values())
    # cprint(
    #     f"[*] Total Trainable Params (action decoder): {total_params / 1e9:.2f}B ({total_params:,} params)",
    #     "magenta",
    # )

    cprint("Model loaded successfully", "green")


def model_test(app: FastAPI, opt: EvaluateConfig):
    if opt.load_model is True:
        load_model(app, opt)
        policy = app.state.policy
        temporal = app.state.temporal
    else:
        # 不加载模型，使用初始化的模型（带默认参数）
        policy = _policy_config.create_trained_policy(
            opt.config, opt.defalut_ckpt_path, default_prompt=opt.task_description
        )
        temporal = TemporalEnsembler(opt)

    obs = {}
    for cam_name in opt.camera_names:
        image_tensor = torch.zeros((3, 480, 640), dtype=torch.float32)
        obs[f"observation.images.{cam_name}"] = image_tensor

    action_dim = 14
    obs["observation.state"] = torch.zeros((action_dim,), dtype=torch.float32)
    obs["prompt"] = opt.task_description

    from tqdm import tqdm

    steps = 100
    for _ in tqdm(range(steps), desc="Model warm-up"):
        raw_action = policy.infer(obs=obs);
        _ = temporal.update(raw_action["actions"])
        time.sleep(0.05)

    cost_time = 0
    for _ in tqdm(range(steps), desc="Inference speed test"):
        start_time = time.perf_counter()
        raw_action = policy.infer(obs=obs)
        action = temporal.update(raw_action["actions"])
        cost_time += time.perf_counter() - start_time
        time.sleep(0.05)
    cprint(
        f"One step cost time {cost_time * 1000 / steps}ms...",
        "blue",
        "on_light_green",
    )
    """action 内容：
    {
        "actions": <shape: (50, 14), dtype: float64>
        "policy_timing":
        {
            Key: infer_ms, value: 9.833616204559803
        }
    }
    """

    values = [f"{v:.2f}" for v in (action if action.ndim == 1 else action[-1])]
    cprint("Get action: [" + ", ".join(values) + "]", "cyan", "on_light_yellow")


def main() -> None:
    # 浮点数只打印三位
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    np.set_printoptions(linewidth=200)

    parser = ArgumentParser()
    parser.add_class_arguments(EvaluateConfig, as_group=False)  # 直接注册为顶级参数
    args = parser.parse_args()
    opt = EvaluateConfig(**vars(args))

    app.state.opt = opt
    app.state.policy = None
    app.state.sac_agent = None

    if opt.model_test:
        model_test(app, opt)
    else:
        load_model(app, opt)

        # 在启动时创建全局的、配置为推理模式的DALI流水线
        batch_size = len(opt.camera_names)
        device_id = 1
        cprint(
            f"✅ Building DALI pipeline for batch_size={batch_size}, device_id={device_id}",
            "green",
        )
        dali_pipeline = ImageProcessingPipeline(
            batch_size=batch_size, num_threads=4, device_id=device_id
        )
        dali_pipeline.build()
        app.state.dali_pipeline = dali_pipeline

        cprint("🚀 Server starting...", "cyan")
        uvicorn.run(app, host="0.0.0.0", port=opt.port, reload=False)


if __name__ == "__main__":
    main()

'''
conda activate dsrl_openpi
cd /root/storage/CODE/lx/dsrl_pi0_lx/openpi
export CUDA_VISIBLE_DEVICES=1,2
python ./scripts/remote.py \
    --defalut_ckpt_path /root/storage/CODE/lx/dsrl_pi0_lx/openpi/openpi-assets/openpi-assets/checkpoints/pi0_base \
    --ckpt_path /root/storage/CODE/lx/dsrl_pi0_lx/openpi/checkpoints/pi0_airbot_local/lx_experiment/99999 \
    --task_description "use the left arm to pick up the yellow cup, then use the right arm to pour the coke" \
    --config pi0_airbot_local \
    --asset_id 0407 \
    --temporal_size 30 \
    --port 6161

cd /root/storage/CODE/lx/dsrl_pi0_lx/openpi
uv run python ./scripts/remote.py --help
'''

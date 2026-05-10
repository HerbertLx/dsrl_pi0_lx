#! /usr/bin/env python
"""
离线 RL 训练脚本：使用 LeRobot 数据集训练 DSRL (pi0 + SAC)

基于 train_sim.py 改造，核心变化：
1. 不需要仿真环境，直接从 LeRobot 数据集加载轨迹
2. 观测格式适配 4 相机数据（cam_high, cam_low, cam_left_wrist, cam_right_wrist）
3. 动作 14 维填充到 32 维（DSRL 的噪声动作空间）
4. 纯离线训练循环（无在线采样）

数据格式：
  obs = {
      "images": {
          "cam_high": (480, 640, 3),
          "cam_low": (480, 640, 3),
          "cam_left_wrist": (480, 640, 3),
          "cam_right_wrist": (480, 640, 3),
      },
      "state": (14,),  # 关节位置
  }
  action = (14,)  # 关节动作

DSRL 内部格式：
  obs_dict = {
      "pixels": (1, resize_image, resize_image, 12, 1),  # 4图拼接
      "state": (14, 1),
  }
  action = (1, 32)  # 14维action + 18维填充0
"""

import os
import sys
import json

# XLA 优化
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib
import jax
import numpy as np
import PIL
from tqdm import tqdm

import gymnasium as gym
from gym.spaces import Dict, Box

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name

import tempfile
from functools import partial
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi_client import image_tools

# JAX 编译缓存
home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

# 数据集根目录
DATASET_ROOT = "/root/storage/CODE/txy/dsrl_pi0_lx/openpi/dataset/0316_pouring_water_easy"


def shard_batch(batch, sharding):
    """将 batch 切分到多个设备"""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """
    虚拟环境，用于定义观测和动作空间。
    支持 4 相机图像拼接 + 14 维状态。
    """
    def __init__(self, variant):
        self.variant = variant
        # 4 个相机，每个 3 通道，拼接后 12 通道
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        
        if variant.add_states:
            state_dim = 14  # 双臂关节位置
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
            
        self.observation_space = Dict(obs_dict)
        # action: 14 维真实动作 + 18 维填充 0 = 32 维
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)


# ============================================================
# 数据加载：从 LeRobot 数据集读取
# ============================================================

def load_lerobot_dataset():
    """
    加载 LeRobot 数据集的元信息。
    
    Returns:
        episodes: list, 每条轨迹的摘要信息
        info: dict, 数据集整体信息
    """
    with open(os.path.join(DATASET_ROOT, "meta", "info.json"), "r") as f:
        info = json.load(f)
    with open(os.path.join(DATASET_ROOT, "meta", "episodes.jsonl"), "r") as f:
        episodes = [json.loads(line) for line in f]
    return episodes, info


def _get_parquet_path(episode_num: int) -> str:
    return os.path.join(DATASET_ROOT, "data", "chunk-000", f"episode_{episode_num:06d}.parquet")


def _get_video_path(episode_num: int, camera: str) -> str:
    return os.path.join(DATASET_ROOT, "videos", "chunk-000", 
                        f"observation.images.{camera}",
                        f"episode_{episode_num:06d}.mp4")


def _load_parquet(episode_num: int):
    import pyarrow.parquet as pq
    path = _get_parquet_path(episode_num)
    return pq.ParquetFile(path).read().to_pandas()


def _extract_frame_from_video(episode_num: int, step_num: int, camera: str) -> np.ndarray:
    """从视频文件中提取指定帧，返回 RGB ndarray"""
    import av
    video_path = _get_video_path(episode_num, camera)
    container = av.open(video_path)
    for i, frame in enumerate(container.decode(video=0)):
        if i == step_num:
            img = frame.to_ndarray(format="rgb24")
            container.close()
            return img
    container.close()
    raise RuntimeError(f"Frame {step_num} not found in {video_path}")


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize 图像到指定尺寸"""
    return np.array(PIL.Image.fromarray(img).resize((size, size)))


def _process_images(images_dict: dict, resize_image: int) -> np.ndarray:
    """
    处理 4 相机图像，resize 并拼接。
    
    Args:
        images_dict: {"cam_high": img, "cam_low": img, ...}
        resize_image: 目标尺寸
    
    Returns:
        img_all: ndarray, shape=(resize_image, resize_image, 12, 1)
                 4 张图沿 channel 拼接，每张 3 通道
    """
    # 按固定顺序处理相机
    cam_order = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    imgs = []
    for cam in cam_order:
        img = images_dict[cam]
        img = _resize_image(img, resize_image)
        imgs.append(img)
    # 沿 channel 拼接: (H, W, 3*4) = (H, W, 12)
    img_all = np.concatenate(imgs, axis=2)
    # 添加最后的 1 维: (H, W, 12, 1)
    img_all = img_all[..., np.newaxis]
    return img_all


def _pad_action(action_14: np.ndarray) -> np.ndarray:
    """
    将 14 维 action 填充到 32 维。
    
    Args:
        action_14: ndarray, shape=(14,) or (T, 14)
    
    Returns:
        action_32: ndarray, shape=(1, 32) or (T, 1, 32)
    """
    action_14 = np.atleast_1d(action_14)
    if action_14.ndim == 1:
        # single action: (14,) -> (1, 32)
        padded = np.zeros(32, dtype=np.float32)
        padded[:14] = action_14
        return padded[np.newaxis, :]  # (1, 32)
    elif action_14.ndim == 2:
        # batch: (T, 14) -> (T, 1, 32)
        T = action_14.shape[0]
        padded = np.zeros((T, 1, 32), dtype=np.float32)
        padded[:, 0, :14] = action_14
        return padded
    else:
        raise ValueError(f"Unexpected action shape: {action_14.shape}")


def _make_obs_dict(images_dict: dict, state: np.ndarray, resize_image: int, add_states: bool) -> dict:
    """
    构造 DSRL 格式的观测字典。
    
    Args:
        images_dict: 4 相机图像字典
        state: ndarray, shape=(14,)
        resize_image: 图像 resize 尺寸
        add_states: 是否添加 state
    
    Returns:
        obs_dict: {"pixels": ..., "state": ...}
    """
    pixels = _process_images(images_dict, resize_image)  # (H, W, 12, 1)
    # 添加 batch 维: (1, H, W, 12, 1)
    pixels = pixels[np.newaxis, ...]
    
    obs_dict = {"pixels": pixels}
    if add_states:
        # state: (14,) -> (14, 1) -> (1, 14, 1)
        state = state[..., np.newaxis]  # (14, 1)
        state = state[np.newaxis, ...]   # (1, 14, 1)
        obs_dict["state"] = state
    return obs_dict


def _make_pi0_input(images_dict: dict, state: np.ndarray) -> dict:
    """
    构造 pi0 策略的输入格式。
    参考 pi0_airbot_local 配置的 repack_transforms。
    
    Args:
        images_dict: 4 相机图像字典
        state: ndarray, shape=(14,)
    
    Returns:
        pi0_input: {"state": ..., "images": {"cam_high": CHW, ...}}
    """
    pi0_images = {}
    for cam_name, img in images_dict.items():
        # resize to 224x224 for pi0
        img_224 = image_tools.resize_with_pad(img, 224, 224)
        img_224 = image_tools.convert_to_uint8(img_224)
        # HWC -> CHW
        pi0_images[cam_name] = np.transpose(img_224, (2, 0, 1))
    
    return {
        "state": state.astype(np.float32),
        "images": pi0_images,
    }


def _load_video_frames(episode_num: int, camera: str) -> np.ndarray:
    """一次性加载整个视频的所有帧，返回 (T, H, W, 3) ndarray"""
    import av
    video_path = _get_video_path(episode_num, camera)
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames)  # (T, H, W, 3)


def load_traj_to_buffer(episode_num: int, variant, replay_buffer, agent_dp=None):
    """
    从 LeRobot 数据集加载单条轨迹到 ReplayBuffer。
    
    Args:
        episode_num: 轨迹编号
        variant: 配置对象
        replay_buffer: ReplayBuffer 实例
        agent_dp: pi0 策略（用于生成 action chunk，可选）
    
    Returns:
        traj_info: dict, 轨迹信息
    """
    df = _load_parquet(episode_num)
    num_frames = len(df)
    
    # 一次性加载所有视频的帧（比逐帧提取快 100 倍）
    videos = {
        "cam_head": _load_video_frames(episode_num, "cam_head"),
        "cam_low": _load_video_frames(episode_num, "cam_low"),
        "cam_left": _load_video_frames(episode_num, "cam_left"),
        "cam_right": _load_video_frames(episode_num, "cam_right"),
    }
    
    # 预计算所有 resize 后的图像（避免每帧重复 resize）
    cam_order = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    cam_src = ["cam_head", "cam_low", "cam_left", "cam_right"]
    
    # 批量 resize: (T, H, W, 3) -> (T, resize, resize, 3)
    resized = {}
    for src_cam in cam_src:
        imgs = videos[src_cam]  # (T, 480, 640, 3)
        T = imgs.shape[0]
        rs = []
        for t in range(T):
            rs.append(_resize_image(imgs[t], variant.resize_image))
        resized[src_cam] = np.stack(rs)  # (T, resize, resize, 3)
    
    # 批量拼接 4 相机图像: (T, resize, resize, 12)
    imgs_concat = np.concatenate([resized[src] for src in cam_src], axis=3)  # (T, H, W, 12)
    imgs_concat = imgs_concat[..., np.newaxis]  # (T, H, W, 12, 1)
    
    # 添加 batch 维: (T, 1, H, W, 12, 1)
    pixels_all = imgs_concat[:, np.newaxis, ...]
    
    # 状态: (T, 14)
    states_all = np.stack(df["observation.state"].values).astype(np.float32)
    
    # action: (T, 14) -> (T, 1, 32)
    actions_14 = np.stack(df["action"].values).astype(np.float32)
    actions_32 = _pad_action(actions_14)
    
    # rewards 和 masks
    rewards = -np.ones(num_frames, dtype=np.float32)
    masks = np.ones(num_frames, dtype=np.float32)
    
    # 插入到 replay buffer
    query_freq = variant.query_freq
    for t in range(num_frames - 1):
        obs = {"pixels": pixels_all[t]}
        next_obs = {"pixels": pixels_all[t + 1]}
        
        if variant.add_states:
            obs["state"] = states_all[t][..., np.newaxis][np.newaxis, ...]  # (1, 14, 1)
            next_obs["state"] = states_all[t + 1][..., np.newaxis][np.newaxis, ...]
        
        # remove batch dimension for buffer insertion
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions_32[t],
            next_actions=actions_32[t + 1],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** query_freq
        )
        replay_buffer.insert(insert_dict)
    
    replay_buffer.increment_traj_counter()
    
    return {
        'episode_num': episode_num,
        'num_frames': num_frames,
        'actions_shape': actions_32.shape,
    }


# ============================================================
# 离线训练循环
# ============================================================

def offline_training_loop(variant, agent, replay_buffer, wandb_logger, shard_fn=None):
    """
    纯离线训练循环：从 replay buffer 中采样 batch 进行梯度更新。
    
    Args:
        variant: 配置对象
        agent: PixelSACLearner
        replay_buffer: 已加载离线数据的 ReplayBuffer
        wandb_logger: W&B 日志器
        shard_fn: batch 分片函数
    """
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        for i in range(variant.max_steps):
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)
            
            pbar.update()
            
            if i % variant.log_interval == 0:
                update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                for k, v in update_info.items():
                    if v.ndim == 0:
                        wandb_logger.log({f'training/{k}': v}, step=i)
                    elif v.ndim <= 2:
                        wandb_logger.log_histogram(f'training/{k}', v, i)
                
                wandb_logger.log({
                    'replay_buffer_size': len(replay_buffer),
                }, i)
            
            if i % variant.eval_interval == 0:
                wandb_logger.log({'num_offline_samples': len(replay_buffer)}, step=i)
                if hasattr(agent, 'perform_eval'):
                    agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, None)
            
            if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


# ============================================================
# 主入口
# ============================================================

def main(variant):
    """离线训练主入口"""
    
    # 1. 硬件配置
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)
    
    tf.config.set_visible_devices([], "GPU")
    
    # 2. 训练参数
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
    
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]
    
    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
    
    outputdir = os.path.abspath(os.path.join(os.environ['EXP'], expname))
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    # 3. 加载数据集信息
    episodes, info = load_lerobot_dataset()
    print(f"Dataset: {info['total_episodes']} episodes, {info['total_frames']} frames")
    variant.task_description = episodes[0]['tasks'][0] if episodes else "offline task"
    variant.env_max_reward = 0  # 离线训练没有明确的成功信号
    
    # 4. 初始化 WandB
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name
    )
    
    # 5. 模型初始化准备
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    
    # 6. 加载 pi0 策略
    config = openpi_config.get_config("pi0_airbot_local")
    checkpoint_dir = "/root/storage/CODE/txy/dsrl_pi0_lx/openpi/checkpoints/pi0_airbot_local/lx_experiment/70000"
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
    
    # 7. 创建 ReplayBuffer 并加载离线数据
    buffer_size = info['total_frames'] + 1000  # 留一些余量
    replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, buffer_size)
    replay_buffer.seed(variant.seed)
    
    print("Loading offline data into replay buffer...")
    for ep in tqdm(episodes):
        ep_num = ep['episode_index']
        traj_info = load_traj_to_buffer(ep_num, variant, replay_buffer, agent_dp)
    
    print(f"Loaded {len(replay_buffer)} transitions from {info['total_episodes']} episodes")
    
    # 8. 执行离线训练
    offline_training_loop(
        variant,
        agent,
        replay_buffer,
        wandb_logger,
        shard_fn=shard_fn,
    )

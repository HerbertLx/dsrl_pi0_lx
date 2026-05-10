#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) # 32 is the noise action space of pi 0


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
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
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        task_id = 57
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    

    if variant.env == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed); breakpoint()
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)








# ========================================================================================================================================================================







'''
#! /usr/bin/env python
import os

# ==========================================
# 环境变量与底层优化配置
# ==========================================
# 告诉 XLA (JAX 的编译器) 使用 Triton GEMM 优化矩阵乘法。
# 根据 HuggingFace 团队的测试，这能在部分 GPU 上提升约 30% 的训练速度 (steps/sec)。
# 参考: https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

# 初始化 JAX 编译缓存，避免每次启动脚本时重复编译相同计算图，大幅缩短启动时间
home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

# ==========================================
# 辅助函数定义
# ==========================================

def _get_libero_env(task, resolution, seed):
    """
    初始化并返回 LIBERO 仿真环境及其任务描述。

    LIBERO 是一个用于机器人操作的基准测试套件。该函数负责加载特定任务的 
    BDDL（行为领域描述语言）文件，并配置离屏渲染环境。

    Args:
        task (libero.libero.benchmark.Task): LIBERO 任务对象，包含任务属性。
        resolution (int): 渲染相机的分辨率（长和宽一致）。
        seed (int): 随机种子，用于保证环境初始化的可复现性。

    Returns:
        tuple: 
            - env (OffScreenRenderEnv): 实例化后的 LIBERO 仿真环境。
            - task_description (str): 任务的自然语言描述（例如："pick up the apple"）。
    """
    task_description = task.language
    # 获取定义环境物理逻辑的 BDDL 文件路径
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    
    # 配置环境参数（分辨率设定）
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    
    # 重要提示：即使使用固定的初始状态，seed 也会影响场景中物体的生成位置
    env.seed(seed)  
    return env, task_description

def shard_batch(batch, sharding):
    """
    将数据批次 (Batch) 沿着第一个维度（Batch 维度）切片并分配到不同的设备 (GPU/TPU) 上。

    这通常用于 JAX 的数据并行训练 (Data Parallelism)，确保每个 GPU 处理 batch 的一部分。

    Args:
        batch (pytree): 一个由数组组成的 Pytree（通常包含 obs, action, reward 等）。
        sharding (jax.sharding.Sharding): JAX 的分片规则对象，形状为 (num_devices,)。

    Returns:
        pytree: 分片并放置在对应物理设备上的数据 pytree。
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):
    """
    虚拟环境包装器，用于提取和统一定义 Observation (观察) 和 Action (动作) 空间。

    由于构建真实环境（如 Libero 离屏渲染）通常很慢且占用资源，使用 DummyEnv 
    可以快速获取空间的 Shape (形状) 和 Dtype (数据类型)。这些信息对于后续
    初始化神经网络参数 (JAX 需要先跑一次 dummy forward pass) 和 回放缓冲区 (Replay Buffer) 至关重要。
    """
    def __init__(self, variant):
        """
        Args:
            variant (Config): 包含配置参数的对象（如环境类型、图像大小等）。
        """
        self.variant = variant
        # 定义图像输入形状：(长, 宽, 3通道 * 相机数量, 1)
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        
        obs_dict = {}
        # 统一视觉观测空间（像素）
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        
        # 统一低维本体感觉状态空间 (Proprioceptive State)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8   # Libero 通常是 8 维状态 (例如关节角度 + 夹爪状态)
            elif variant.env == 'aloha_cube':
                state_dim = 14  # ALOHA 双臂通常是 14 维状态 (7维/臂)
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
            
        self.observation_space = Dict(obs_dict)
        # 预设动作空间，维度为 32。注释说明这匹配了 pi0 模型的噪声动作空间 (noise action space)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) 


# ==========================================
# 主训练入口
# ==========================================

def main(variant):
    """
    主训练脚本逻辑。负责初始化计算资源、日志系统、仿真环境、算法模型，并拉起训练循环。

    Args:
        variant (Config): 包含所有超参数和运行配置的对象。
    """
    # 1. 硬件与并行配置
    devices = jax.local_devices()
    num_devices = len(devices)
    # 确保批量大小可以被设备数量整除，以防止负载不均导致报错
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    
    # 创建位置分片规则（将 batch 维度均匀分配给所有计算设备）
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # 【关键内存优化】禁止 TensorFlow 使用 GPU。
    # 因为底层数据加载可能用了 tf.data，防止 TF 和 JAX 抢占 GPU 显存导致 OOM (Out Of Memory)
    tf.config.set_visible_devices([], "GPU")
    
    # 2. 训练超参数与日志路径配置
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        # 如果启用了余弦退火学习率，将衰减步数与总训练步数对齐
        kwargs['decay_steps'] = variant.max_steps
        
    # 如果未指定实验前缀，随机生成一个唯一的 5 位 UUID 作为前缀
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    # 构建完整的实验名称
    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    # 拼接并创建输出目录，用于保存检查点 (Checkpoints) 和本地日志
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    # 3. 仿真环境初始化
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]() # 使用 libero_90 任务集
        task_id = 57 # 选定具体的任务 ID
        task = task_suite.get_task(task_id)
        # 初始化训练和评估环境
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
        
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        # 动态注册 ALOHA 魔方转移任务
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env) # 评估环境使用独立副本
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        
    # 4. 初始化 WandB (Weights & Biases) 实验追踪
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    # 5. 模型初始化准备 (通过 DummyEnv 采样获取张量形状)
    dummy_env = DummyEnv(variant)
    # 增加 batch_dim 用于 JAX 网络的初始化 forward pass
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    
    # 6. 加载预训练的 pi0 大脑/策略模型
    if variant.env == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
        
    # 加载专家/基座策略模型 (pi0)
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)
    
    # 实例化 SAC (Soft Actor-Critic) 学习器，它将负责主要的 RL 训练迭代
    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # 7. 初始化经验回放缓冲区 (Replay Buffer)
    # 大小取决于 max_steps 和 multi_grad_step 的比例
    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    
    # 8. 启动主训练循环
    # 交替进行环境交互收集轨迹 (trajectory) 和网络参数更新
    trajwise_alternating_training_loop(
        variant, 
        agent, 
        env, 
        eval_env, 
        online_replay_buffer, 
        replay_buffer, 
        wandb_logger, 
        shard_fn=shard_fn, 
        agent_dp=agent_dp
    )
'''
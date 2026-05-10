#! /usr/bin/env python  # 使用当前环境的 Python 解释器执行脚本
import os  # 读取环境变量与拼接路径

# ==========================================
# 环境变量与底层优化配置
# ==========================================
# 告诉 XLA (JAX 的编译器) 使用 Triton GEMM 优化矩阵乘法。
# 根据 HuggingFace 团队的测试，这能在部分 GPU 上提升约 30% 的训练速度 (steps/sec)。
# 参考: https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')  # 读取已有 XLA 配置
xla_flags += ' --xla_gpu_triton_gemm_any=True'  # 打开 Triton GEMM 优化
os.environ['XLA_FLAGS'] = xla_flags  # 写回环境变量给 JAX 使用

import pathlib, copy  # 文件路径工具与深拷贝工具

import jax  # JAX 计算与并行库
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner  # SAC 学习器
from jaxrl2.utils.general_utils import add_batch_dim  # 为样本补 batch 维度
import numpy as np  # 数值计算库

import gymnasium as gym  # 通用 Gym 环境接口
import gym_aloha  # ALOHA 仿真环境注册
from gym.spaces import Dict, Box  # 观测与动作空间定义

from libero.libero import benchmark  # LIBERO 任务集合
from libero.libero import get_libero_path  # LIBERO 数据路径
from libero.libero.envs import OffScreenRenderEnv  # LIBERO 离屏渲染环境

from jaxrl2.data import ReplayBuffer  # 回放缓冲区
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name  # W&B 日志工具
import tempfile  # 临时目录工具
from functools import partial  # 固定部分参数的函数包装
from examples.train_utils_sim import trajwise_alternating_training_loop, read_collect_training_loop   # 仿真训练循环
import tensorflow as tf  # 用于数据管线，但这里禁用 GPU
from jax.experimental.compilation_cache import compilation_cache  # JAX 编译缓存

from openpi.training import config as openpi_config  # pi0 配置加载
from openpi.policies import policy_config  # pi0 策略构造
from openpi.shared import download  # 远程权重下载

# 初始化 JAX 编译缓存，避免每次启动脚本时重复编译相同计算图，大幅缩短启动时间
home_dir = os.environ['HOME']  # 用户主目录
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))  # 初始化 JAX 编译缓存

# ==========================================
# 辅助函数定义
# ==========================================

def _get_libero_env(task, resolution, seed):  # 构造 LIBERO 仿真环境
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
    task_description = task.language  # 任务自然语言描述
    # 获取定义环境物理逻辑的 BDDL 文件路径
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file  # BDDL 文件路径
    
    # 配置环境参数（分辨率设定）
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}  # 环境参数
    env = OffScreenRenderEnv(**env_args)  # 创建离屏渲染环境
    
    # 重要提示：即使使用固定的初始状态，seed 也会影响场景中物体的生成位置
    env.seed(seed)  # 固定随机种子保证可复现
    return env, task_description  # 返回环境和任务描述

def shard_batch(batch, sharding):  # 将 batch 切分到多个设备
    """
    将数据批次 (Batch) 沿着第一个维度（Batch 维度）切片并分配到不同的设备 (GPU/TPU) 上。

    这通常用于 JAX 的数据并行训练 (Data Parallelism)，确保每个 GPU 处理 batch 的一部分。

    Args:
        batch (pytree): 一个由数组组成的 Pytree（通常包含 obs, action, reward 等）。
        sharding (jax.sharding.Sharding): JAX 的分片规则对象，形状为 (num_devices,)。

    Returns:
        pytree: 分片并放置在对应物理设备上的数据 pytree。
    """
    return jax.tree_util.tree_map(  # 逐字段搬运到对应设备
        lambda x: jax.device_put(  # 把数组放到分片后的设备上
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))  # 只切 batch 维
        ),
        batch,  # 输入 pytree
    )


class DummyEnv(gym.ObservationWrapper):  # 仅用于构造空间形状的虚拟环境
    """
    虚拟环境包装器，用于提取和统一定义 Observation (观察) 和 Action (动作) 空间。

    由于构建真实环境（如 Libero 离屏渲染）通常很慢且占用资源，使用 DummyEnv 
    可以快速获取空间的 Shape (形状) 和 Dtype (数据类型)。这些信息对于后续
    初始化神经网络参数 (JAX 需要先跑一次 dummy forward pass) 和 回放缓冲区 (Replay Buffer) 至关重要。
    """
    def __init__(self, variant):  # 根据配置定义观测和动作空间
        """
        Args:
            variant (Config): 包含配置参数的对象（如环境类型、图像大小等）。
        """
        self.variant = variant  # 保存运行配置
        # 定义图像输入形状：(长, 宽, 3通道 * 相机数量, 1)
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)  # 图像张量形状
        
        obs_dict = {}  # 观测空间字典
        # 统一视觉观测空间（像素）
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)  # 像素观测
        
        # 统一低维本体感觉状态空间 (Proprioceptive State)
        if variant.add_states:  # 需要低维状态时才加入
            if variant.env == 'libero':
                state_dim = 8   # Libero 通常是 8 维状态
            elif variant.env == 'aloha_cube':
                state_dim = 14  # ALOHA 双臂通常是 14 维状态
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)  # 状态空间
            
        self.observation_space = Dict(obs_dict)  # 组装观测空间
        # 预设动作空间，维度为 32。注释说明这匹配了 pi0 模型的噪声动作空间 (noise action space)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)  # pi0 噪声动作空间

class SmokeEnv(gym.Env):
    """
    烟雾测试环境 (Smoke Test Environment)。
    不执行任何真实物理仿真，step 时直接返回随机观测。
    用于快速验证训练管线、网络前向传播和日志记录是否正常。
    """
    def __init__(self, variant):  # 根据配置定义观测和动作空间
        self.variant = variant  # 保存运行配置
        # 定义图像输入形状：(长, 宽, 3通道 * 相机数量, 1)
        variant.num_cameras = 3
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)  # 图像张量形状
        
        obs_dict = {}  # 观测空间字典
        # 统一视觉观测空间（像素）
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)  # 像素观测
        
        # 统一低维本体感觉状态空间 (Proprioceptive State)
        if variant.add_states:  # 需要低维状态时才加入
            state_dim = 14  # ALOHA 双臂通常是 14 维状态
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)  # 状态空间
            
        self.observation_space = Dict(obs_dict)  # 组装观测空间
        # 预设动作空间，维度为 32。注释说明这匹配了 pi0 模型的噪声动作空间 (noise action space)
        self.action_space = Box(low=-1, high=1, shape=(1, 14,), dtype=np.float32)  # pi0 噪声动作空间

    def reset(self, seed=None, options=None):
        self._step_count = 0
        obs = self.observation_space.sample()
        # 确保图像是有效的 uint8，避免全黑或异常值
        if 'pixels' in obs:
            obs['pixels'] = np.random.randint(0, 256, size=obs['pixels'].shape, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        if 'pixels' in obs:
            obs['pixels'] = np.random.randint(0, 256, size=obs['pixels'].shape, dtype=np.uint8)
        # 固定返回 -1 奖励，永不成功，也永不提前终止
        reward = -1.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}
# ==========================================
# 主训练入口
# ==========================================

def main(variant):  # 主训练入口
    """
    主训练脚本逻辑。负责初始化计算资源、日志系统、仿真环境、算法模型，并拉起训练循环。

    Args:
        variant (Config): 包含所有超参数和运行配置的对象。
    """
    # 1. 硬件与并行配置
    devices = jax.local_devices()  # 当前可见设备列表
    num_devices = len(devices)  # 设备数量
    # 确保批量大小可以被设备数量整除，以防止负载不均导致报错
    assert variant.batch_size % num_devices == 0  # 保证 batch 可均分到所有设备
    print('num devices', num_devices)  # 打印设备数量
    print('batch size', variant.batch_size)  # 打印 batch 大小
    
    # 创建位置分片规则（将 batch 维度均匀分配给所有计算设备）
    sharding = jax.sharding.PositionalSharding(devices)  # 按设备顺序创建分片规则
    shard_fn = partial(shard_batch, sharding=sharding)  # 固定分片规则的 batch 分片函数

    # 【关键内存优化】禁止 TensorFlow 使用 GPU。
    # 因为底层数据加载可能用了 tf.data，防止 TF 和 JAX 抢占 GPU 显存导致 OOM (Out Of Memory)
    tf.config.set_visible_devices([], "GPU")  # 禁止 TensorFlow 占用 GPU
    
    # 2. 训练超参数与日志路径配置
    kwargs = variant['train_kwargs']  # 读取训练超参数
    if kwargs.pop('cosine_decay', False):  # 如果启用了余弦退火
        kwargs['decay_steps'] = variant.max_steps  # 将衰减步数设为总步数
        
    # 如果未指定实验前缀，随机生成一个唯一的 5 位 UUID 作为前缀
    if not variant.prefix:  # 没有前缀就自动生成
        import uuid  # 生成随机实验前缀
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]  # 截取前 5 位

    # 构建完整的实验名称
    if variant.suffix:  # 如果有后缀就拼接上去
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"  # 完整实验名
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)  # 仅使用前缀和种子
   
    # 拼接并创建输出目录，用于保存检查点 (Checkpoints) 和本地日志
    # Orbax 要求存盘路径为绝对路径；EXP 常为 ./logs/... 相对路径
    outputdir = os.path.abspath(os.path.join(os.environ['EXP'], expname))
    variant.outputdir = outputdir  # 记录到配置中
    if not os.path.exists(outputdir):  # 目录不存在就创建
        os.makedirs(outputdir)  # 创建实验输出目录
    print('writing to output dir ', outputdir)  # 打印输出目录
    
    # 3. 仿真环境初始化
    if variant.env == 'libero':  # LIBERO 仿真分支
        benchmark_dict = benchmark.get_benchmark_dict()  # 获取 benchmark 字典
        task_suite = benchmark_dict["libero_90"]() # 使用 libero_90 任务集
        task_id = 57 # 选定具体的任务 ID
        task = task_suite.get_task(task_id)  # 读取任务对象
        env, task_description = _get_libero_env(task, 256, variant.seed)  # 构造训练环境
        eval_env = env  # 评估环境复用同一环境
        variant.task_description = task_description  # 保存任务文本描述
        variant.env_max_reward = 1  # LIBERO 奖励上限
        variant.max_timesteps = 400  # 每回合最大步数
        
    elif variant.env == 'aloha_cube':  # ALOHA 仿真分支
        from gymnasium.envs.registration import register  # 动态注册任务
        register(  # 注册 ALOHA transfer cube 环境
            id="gym_aloha/AlohaTransferCube-v0",  # 环境 ID
            entry_point="gym_aloha.env:AlohaEnv",  # 环境入口
            max_episode_steps=400,  # 最大步数
            nondeterministic=True,  # 非确定性任务
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},  # 任务参数
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")  # 创建环境
        eval_env = copy.deepcopy(env) # 评估环境使用独立副本
        variant.env_max_reward = 4  # ALOHA 奖励上限
        variant.max_timesteps = 400  # 每回合最大步数
        
    elif variant.env == 'smoke':  # 烟雾测试分支
        env = SmokeEnv(variant)
        eval_env = SmokeEnv(variant)
        variant.env_max_reward = 0  # 永不成功
        variant.max_timesteps = 400
        variant.task_description = "smoke test"
        
    # 4. 初始化 WandB (Weights & Biases) 实验追踪
    group_name = variant.prefix + '_' + variant.launch_group_id  # 生成分组名
    wandb_output_dir = tempfile.mkdtemp()  # 创建临时日志目录
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)  # 初始化日志器

    # 5. 模型初始化准备 (通过 DummyEnv 采样获取张量形状)
    # dummy_env = DummyEnv(variant)  # 构造虚拟环境以推断形状
    dummy_env = SmokeEnv(variant)  # 构造虚拟环境以推断形状
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())  # 为观测加 batch 维
    sample_action = add_batch_dim(dummy_env.action_space.sample())  # 为动作加 batch 维
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])  # 打印观测形状
    print('sample action shape', sample_action.shape)  # 打印动作形状
    
    # 6. 加载预训练的 pi0 大脑/策略模型
    config = openpi_config.get_config("pi0_airbot_local")  # 加载 pi0_libero 配置
    checkpoint_dir = "/root/storage/CODE/txy/dsrl_pi0_lx/openpi/checkpoints/pi0_airbot_local/lx_experiment/70000"  # 从配置中获取 checkpoint_dir
    '''
    if variant.env == 'libero':  # LIBERO 对应的 pi0 配置
        config = openpi_config.get_config("pi0_libero")  # 加载 pi0_libero 配置
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")  # 下载权重
    elif variant.env == 'aloha_cube':  # ALOHA 对应的 pi0 配置
        config = openpi_config.get_config("pi0_aloha_sim")  # 加载 pi0_aloha_sim 配置
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")  # 下载权重
    else:
        raise NotImplementedError()  # 不支持的环境类型
    '''
        
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)  # 构造预训练 pi0 策略
    print("Loaded pi0 policy from %s", checkpoint_dir)  # 打印权重路径

    # Pixel SAC 的数据增强按「相机」切片，每相机 3 个 RGB 通道。train_kwargs 里默认 num_cameras=1，
    # 但 SmokeEnv / DummyEnv 会把 variant.num_cameras 设为真实相机数；若不同步，color_transform
    # 会把多相机拼成的 9 通道当成单图处理，输出 (H,W,3,3)，Encoder reshape 后只剩 3 通道，
    # 与 init 时的 9 通道卷积核冲突（ScopeParamShapeError: kernel 期望 in_features=9 实为 3）。
    kwargs['num_cameras'] = variant.num_cameras

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)  # 初始化 SAC 学习器

    online_buffer_size = variant.max_steps  // variant.multi_grad_step  # 在线 buffer 容量
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))  # 创建回放缓冲区
    replay_buffer = online_replay_buffer  # 训练直接复用在线 buffer
    replay_buffer.seed(variant.seed)  # 设置随机种子
    
    if variant.env == 'smoke':
        read_collect_training_loop(
            variant,  # 配置对象
            agent,  # RL 学习器
            env,  # 训练环境
            eval_env,  # 评估环境
            online_replay_buffer,  # 在线回放缓冲区
            replay_buffer,  # 采样用回放缓冲区
            wandb_logger,  # 日志器
            shard_fn=shard_fn,  # batch 分片函数
            agent_dp=agent_dp,  # 预训练 pi0 策略
            perform_control_evals=False,  # 是否进行控制评估
        )  # 执行交替采样与更新
    else:
        trajwise_alternating_training_loop(
            variant,  # 配置对象
            agent,  # RL 学习器
            env,  # 训练环境
            eval_env,  # 评估环境
            online_replay_buffer,  # 在线回放缓冲区
            replay_buffer,  # 采样用回放缓冲区
            wandb_logger,  # 日志器
            shard_fn=shard_fn,  # batch 分片函数
            agent_dp=agent_dp,  # 预训练 pi0 策略
            perform_control_evals=False,  # 是否进行控制评估
        )  # 执行交替采样与更新
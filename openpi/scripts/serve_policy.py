import dataclasses
import enum
import logging
import socket

import tyro  # 用于将 Python 结构体自动转换为命令行界面的工具

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """支持的机器人运行环境枚举类。"""

    ALOHA = "aloha"        # 实体 ALOHA 机器人环境
    ALOHA_SIM = "aloha_sim" # ALOHA 仿真环境
    DROID = "droid"        # DROID 机器人数据集/环境
    LIBERO = "libero"      # LIBERO 机器人基准测试环境


@dataclasses.dataclass
class Checkpoint:
    """
    配置类：用于从指定的本地或云端路径加载模型检查点。
    
    Attributes:
        config: 训练配置的名称，用于确定模型架构（如 "pi0_aloha_sim"）。
        dir: 检查点文件所在的目录路径。
    """

    # 训练配置名称 (例如: "pi0_aloha_sim")
    config: str 
    # 检查点目录路径 (例如: "checkpoints/pi0_aloha_sim/exp/10000")
    dir: str 


@dataclasses.dataclass
class Default:
    """配置类：标记使用对应环境的官方默认策略。"""


@dataclasses.dataclass
class Args:
    """
    serve_policy 脚本的命令行参数定义类。
    使用 tyro 解析，可以通过命令行 --env 或 --port 等进行修改。
    """

    # 运行策略的目标环境，默认为 ALOHA 仿真环境
    env: EnvMode = EnvMode.ALOHA_SIM

    # 默认提示词：如果输入数据中没有 "prompt" 键，则使用此默认文本
    default_prompt: str | None = None

    # WebSocket 服务器监听的端口号，默认 8000
    port: int = 8123
    
    # 是否记录策略的行为数据（用于后续调试分析）
    record: bool = False

    # 指定如何加载策略：可以是具体的 Checkpoint 路径，也可以是 Default（默认）
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 定义各个环境对应的默认检查点（通常存储在 Google Cloud Storage 上）
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """
    创建指定环境的默认训练策略。
    
    输入:
        env: 环境模式枚举
        default_prompt: 可选的默认文本提示
    输出:
        实例化后的 Policy 对象
    逻辑: 从预定义的映射表中查找路径并调用框架接口加载模型。
    """
    # 尝试从预定义的字典中获取对应环境的检查点信息
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        # 调用 openpi 接口，根据配置名和目录创建已训练好的策略实例
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    # 如果环境不在支持列表中，抛出错误
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """
    根据命令行参数决定创建哪种策略实例。
    
    输入:
        args: 解析后的命令行参数对象
    输出:
        Policy 策略实例
    逻辑: 使用模式匹配判断用户提供的是具体路径还是要求使用默认配置。
    """
    # 匹配 args.policy 的类型
    match args.policy:
        # 如果用户通过命令行提供了具体的 checkpoint 路径和配置
        case Checkpoint():
            # 使用用户指定的路径和配置加载模型
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        # 如果用户选择了 Default（默认行为）
        case Default():
            # 调用函数创建该环境对应的官方默认策略
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    """
    脚本主入口：加载模型并启动 WebSocket 服务器。
    
    输入:
        args: 包含所有运行配置的 Args 对象
    逻辑: 1. 创建策略 2. (可选)封装记录器 3. 获取网络信息 4. 启动持久化服务器
    """
    # 根据参数创建核心策略对象（模型）
    policy = create_policy(args)
    # 提取策略的元数据（如动作空间定义、输入规范等）
    policy_metadata = policy.metadata

    # 如果开启了记录模式
    if args.record:
        # 使用 PolicyRecorder 包装原始策略，将推理过程保存到 "policy_records" 文件夹
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 获取当前服务器的主机名
    hostname = socket.gethostname()
    # 根据主机名获取本地 IP 地址，方便用户知道连接到哪里
    local_ip = socket.gethostbyname(hostname)
    # 打印服务器创建的日志信息
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 实例化 WebSocket 服务器
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,              # 处理请求的策略模型
        host="0.0.0.0",             # 监听所有网卡接口
        port=args.port,             # 服务器端口
        metadata=policy_metadata,    # 传递给客户端的模型元数据
    )
    # 启动服务器并进入无限循环，等待客户端连接
    server.serve_forever()


if __name__ == "__main__":
    # 配置日志系统，设置级别为 INFO，force=True 确保配置覆盖之前的设置
    logging.basicConfig(level=logging.INFO, force=True)
    # 使用 tyro.cli 解析命令行输入并直接运行 main 函数
    main(tyro.cli(Args))
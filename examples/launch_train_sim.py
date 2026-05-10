import argparse  # 命令行参数解析
import sys  # 进程退出
from examples.train_sim import main  # 仿真训练主入口
from jaxrl2.utils.launch_util import parse_training_args  # 将默认训练参数与 CLI 合并


if __name__ == '__main__':  # 仅在脚本直接执行时运行
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)  # 全局随机种子
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')  # W&B 分组 ID
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)  # 每次评估回合数
    parser.add_argument('--env', default='libero', help='name of environment')  # 环境类型: libero / aloha_cube
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)  # 训练日志间隔
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)  # 评估间隔
    parser.add_argument('--checkpoint_interval', default=50000, help='checkpoint interval.', type=int)  # 存 ckpt 间隔, -1 表示关闭
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)  # 训练 batch size
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)  # 最大梯度步数
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)  # 是否拼接低维状态
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')  # W&B 项目名
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)  # 开始更新前先收集的步数
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')  # 算法标识
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')  # 实验名前缀
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')  # 实验名后缀
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)  # UTD 比率
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)  # 图像缩放尺寸
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)  # 每隔多少步向策略查询动作块
    
    train_args_dict = dict(  # 这些键会被 parse_training_args 自动注册成 --<key> CLI 参数
        actor_lr=1e-4,  # actor 学习率
        critic_lr= 3e-4,  # critic 学习率
        temp_lr=3e-4,  # 熵温度学习率
        hidden_dims= (128, 128, 128),  # MLP 隐层宽度
        cnn_features= (32, 32, 32, 32),  # CNN 每层通道数
        cnn_strides= (2, 1, 1, 1),  # CNN 每层步幅
        cnn_padding= 'VALID',  # CNN padding 策略
        latent_dim= 50,  # bottleneck 潜变量维度
        discount= 0.999,  # 折扣因子
        tau= 0.005,  # 目标网络软更新系数
        critic_reduction = 'mean',  # 多 Q 聚合方式
        dropout_rate=0.0,  # dropout 比例
        aug_next=1,  # 是否对 next_obs 做增强
        use_bottleneck=True,  # 是否启用 bottleneck
        encoder_type='small',  # 图像编码器类型
        encoder_norm='group',  # 编码器归一化类型
        use_spatial_softmax=True,  # 是否启用 spatial softmax
        softmax_temperature=-1,  # spatial softmax 温度
        target_entropy='auto',  # 目标熵
        num_qs=10,  # Q 网络数量
        action_magnitude=1.0,  # 动作幅值缩放
        num_cameras=1,  # 相机数量
        )

    variant, args = parse_training_args(train_args_dict, parser)  # 解析 CLI 并合并到配置对象
    print(variant)  # 打印最终配置，便于复现实验
    main(variant)  # 进入仿真训练主流程
    sys.exit()  # 正常退出
    
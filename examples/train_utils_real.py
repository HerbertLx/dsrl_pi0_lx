import os
import time
from tqdm import tqdm
import time
import numpy as np
import jax
import sys
import select
import tty
import termios
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, agent_dp=None, robot_config=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
        
    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
   
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp, wandb_logger, total_num_traj, robot_config)
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_num_traj)
            print('total env steps:', total_env_steps)
            
            if i == 0:
                num_gradsteps = 5000
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
            print(f'num_gradsteps: {num_gradsteps}')
            if total_num_traj >= variant.num_initial_traj_collect:
                for _ in range(num_gradsteps):

                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                    
                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    # 定义函数：将在线收集的一条轨迹数据拆分为单步转移，插入到在线回放缓冲区中
    
    discount_horizon = variant.query_freq  # 获取查询频率，用于计算折扣因子
    actions = np.array(traj['actions']) # (T, chunk_size, 14)  # 将轨迹中的动作序列转换为numpy数组，形状为(时间步数, 块大小, 14)
    episode_len = len(actions)  # 计算轨迹长度（时间步数）
    rewards = np.array(traj['rewards'])  # 将轨迹中的奖励序列转换为numpy数组
    masks = np.array(traj['masks'])  # 将轨迹中的掩码序列转换为numpy数组（用于标记有效/无效数据）

    for t in range(episode_len):  # 遍历轨迹中的每一个时间步
        obs = traj['observations'][t]  # 获取当前时间步的观测数据
        next_obs = traj['observations'][t + 1]  # 获取下一个时间步的观测数据
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}  # 移除观测数据中的batch维度（取第一个样本）
        next_obs = {k: v[0] for k, v in next_obs.items()}  # 对下一时刻观测同样移除batch维度
        if not variant.add_states:  # 如果配置中不需要添加状态信息
            obs.pop('state', None)  # 从当前观测中移除'state'键（如果不存在则返回None）
            next_obs.pop('state', None)  # 从下一时刻观测中同样移除'state'键
        
        insert_dict = dict(  # 构建单步转移字典，用于插入回放缓冲区
            observations=obs,  # 当前时刻观测
            next_observations=next_obs,  # 下一时刻观测
            actions=actions[t],  # 当前时刻执行的动作
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],  # 下一时刻动作；若是最后一步则复用当前动作
            rewards=rewards[t],  # 当前时刻获得的奖励
            masks=masks[t],  # 当前时刻的掩码
            discount=variant.discount ** discount_horizon  # 计算折扣因子：折扣率的discount_horizon次幂
        )
        online_replay_buffer.insert(insert_dict)  # 将单步转移数据插入在线回放缓冲区
    online_replay_buffer.increment_traj_counter()  # 轨迹处理完成后，增加轨迹计数器

def collect_traj(variant, agent, env, i, agent_dp=None, wandb_logger=None, traj_id=None, robot_config=None):
    query_frequency = variant.query_freq  # 查询频率，每隔多少步生成一次动作块
    instruction = variant.instruction  # 当前任务的自然语言指令
    max_timesteps = robot_config['max_timesteps']  # 该 trial 最大步数
    agent._rng, rng = jax.random.split(agent._rng)  # 拆分 JAX 随机数生成器
    try:
        env.reset()  # 重置环境
    except Exception as e:
        print(f"Environment reset failed")  # 打印环境重置失败
        import traceback
        traceback.print_exc()  # 打印详细异常信息
        import pdb; pdb.set_trace()  # 进入调试
    step_time = 1 / 15 # 15 Hz  # 控制机械臂执行频率为 15Hz
    last_step_time = time.time()  # 记录上一次步进的时间戳
    old_settings = termios.tcgetattr(sys.stdin)  # 保存终端原始设置
    
    rewards = []  # 存储每步奖励
    action_list = []  # 存储每次 query 时的动作噪声
    obs_list = []  # 存储每次 query 时的观测
    image_list = []  # 存储每步的相机图像

    old_settings = termios.tcgetattr(sys.stdin)  # 再次保存终端设置（防止后续修改）
    try:
        tty.setcbreak(sys.stdin.fileno())  # 设置终端为 cbreak 模式，便于实时读取键盘输入
        for t in tqdm(range(max_timesteps)):    # 主循环，遍历每个时间步
            # 检查是否有键盘输入（如 q 退出）
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)  # 读取一个字符
                if char_input.lower() == 'q':  # 按 q 退出
                    print("'q' pressed, stopping loop.")
                    break
            
            try:
                _env_obs = env.get_observation()  # 获取环境观测
            except Exception as e:
                print(f"Environment get obs failed")  # 获取观测失败
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()
            curr_obs = _extract_observation(
                    robot_config,  # 机器人配置
                    _env_obs,      # 原始观测
            )  # 解析出结构化观测
            image_list.append(curr_obs[robot_config['camera_to_use'] + "_image"])  # 存储当前相机图像

            request_data = get_pi0_input(curr_obs, robot_config, instruction)  # 构造 pi0 输入
        
            if t % query_frequency == 0:  # 每隔 query_frequency 步生成一次动作块

                rng, key = jax.random.split(rng)  # 再次拆分随机数

                img_all = process_images(variant, curr_obs)  # 拼接所有相机图像
                
                # 提取 pi0 VLM backbone 特征，并与关节状态拼接
                img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)  # 提取视觉特征
                img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048) 取最后一帧特征
                qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])  # 拼接关节+夹爪+视觉特征

                obs_dict = {
                    'pixels': img_all,  # 图像观测
                    'state': qpos[np.newaxis, ..., np.newaxis],  # 状态观测，扩展 batch 维和通道维
                }
                if i == 0:
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))  # 第一次采样用高斯噪声
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 10 - noise.shape[1], axis=1)  # 补齐长度
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)  # 拼成完整动作块
                    actions_noise = noise[0, :agent.action_chunk_shape[0], :]  # 取前 action_chunk_shape[0] 个
                else:
                    # sac agent 预测扩散模型噪声
                    actions_noise = agent.sample_actions(obs_dict)  # SAC 输出噪声
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)  # 调整形状
                    noise = np.repeat(actions_noise[-1:, :], 10 - actions_noise.shape[0], axis=0)  # 补齐长度
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]  # 拼成完整动作块
                action_list.append(actions_noise)  # 存储噪声
                obs_list.append(obs_dict)  # 存储观测
                action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]  # pi0 推理出动作块

            action_t = action[t % query_frequency]  # 取当前步对应的动作
            
            # 二值化夹爪动作（大于0.5为开，否则为关）
            if action_t[-1].item() > 0.5:
                action_t = np.concatenate([action_t[:-1], np.ones((1,))])  # 夹爪开
            else:
                action_t = np.concatenate([action_t[:-1], np.zeros((1,))])  # 夹爪关
            action_t = np.clip(action_t, -1, 1)  # 限幅到 [-1, 1]
            
            try:
                env.step(action_t)  # 执行动作
            except Exception as e:
                print(f"Environment step failed")  # 执行动作失败
                import traceback
                traceback.print_exc()  # 打印详细异常
                import pdb; pdb.set_trace()
        
            now = time.time()  # 当前时间
            dt = now - last_step_time  # 距离上次步进的时间
            if dt < step_time:
                time.sleep(step_time - dt)  # 控制频率
                last_step_time = time.time()  # 更新时间戳
            else:
                last_step_time = now  # 更新时间戳
            
        print("Trial finished. Mark as (1) Success or (0) Failure:")  # 试验结束，人工标注成功/失败
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)  # 读取输入
                if char_input == '1':
                    print("Trial marked as SUCCESS.")
                    is_success = True  # 标记成功
                    break
                elif char_input == '0':
                    print("Trial marked as FAILURE.")                    
                    is_success = False  # 标记失败
                    break
                else:
                    print("Invalid input. Please enter '1' for Success or '0' for Failure:")  # 输入无效
            time.sleep(0.01) # 防止忙等

        try:
            _env_obs = env.get_observation()  # 获取最后观测
        except Exception as e:
            print(f"Environment get obs failed")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
        
        # 添加最后一步观测
        curr_obs = _extract_observation(
                    robot_config,
                    _env_obs,
            )
        image_list.append(curr_obs[robot_config['camera_to_use'] + "_image"])  # 存储最后图像
        request_data = get_pi0_input(curr_obs, robot_config, instruction)  # 构造 pi0 输入
        img_all = process_images(variant, curr_obs)  # 拼接所有相机图像
        img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)  # 提取视觉特征
        img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048)
        qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])  # 拼接关节+夹爪+视觉特征
        obs_dict = {
            'pixels': img_all,
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
        obs_list.append(obs_dict)  # 存储观测
        print(f'Rollout Done')  # 打印结束
        
    finally:
        if is_success:
            query_steps = len(action_list)  # 查询步数
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])  # 成功奖励
            masks = np.concatenate([np.ones(query_steps - 1), [0]])  # 成功 mask
        else:
            query_steps = len(action_list)
            rewards = -np.ones(query_steps)  # 失败奖励
            masks = np.ones(query_steps)  # 失败 mask
            
        if wandb_logger is not None:
            wandb_logger.log({f'is_success': int(is_success)}, step=i)  # 日志记录成功
            wandb_logger.log({f'total_num_traj': traj_id}, step=i)  # 日志记录轨迹数

        video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}.mp4')  # 保存视频路径
        video = np.stack(image_list)  # 拼接视频帧
        ImageSequenceClip(list(video), fps=15).write_videofile(video_path, codec="libx264")  # 写视频
       
        print("Episide Done! Press c after resetting the environment")  # 提示重置
        try:
            env.reset()  # 重置环境
        except Exception as e:
            print(f"Environment reset failed")
            import traceback
            traceback.print_exc()  # 打印异常
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()  # 进入调试
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)  # 恢复终端设置
    
    traj = {
        'observations': obs_list,  # 观测序列
        'actions': action_list,    # 动作噪声序列
        'rewards': rewards,        # 奖励序列
        'masks': masks,            # mask 序列
        'is_success': is_success,  # 是否成功
        'env_steps': t + 1,       # 总步数
    }
    
    return traj  # 返回轨迹


def _extract_observation(robot_config, obs_dict):
    '''
    from https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py
    '''
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations.keys():
        if robot_config['left_camera_id'] in key and "left" in key:
            left_image = image_observations[key]
        elif robot_config['right_camera_id'] in key and "left" in key:
            right_image = image_observations[key]
        elif robot_config['wrist_camera_id'] in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }
    
def get_pi0_input(obs, robot_config, instruction):
    external_image = obs[robot_config['camera_to_use'] + "_image"]
    request_data = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            external_image, 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_image"], 224, 224),
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": instruction,
    }
    return request_data
    

def process_images(variant, obs):
    # 定义函数：处理并拼接来自多个摄像头的图像数据
    '''
    concat the images from all cameras
    '''
    im1 = image_tools.resize_with_pad(obs["left_image"], variant.resize_image, variant.resize_image)  # 对左侧摄像头图像进行等比例缩放并填充到指定尺寸
    im2 = image_tools.resize_with_pad(obs["right_image"], variant.resize_image, variant.resize_image)  # 对右侧摄像头图像进行等比例缩放并填充到指定尺寸
    im3 = image_tools.resize_with_pad(obs["wrist_image"], variant.resize_image, variant.resize_image)  # 对手腕摄像头图像进行等比例缩放并填充到指定尺寸
    img_all = np.concatenate([im1, im2, im3], axis=2)[np.newaxis, ..., np.newaxis]  # 沿通道维度(axis=2)拼接三张图像，并添加batch维度和额外的空维度
    return img_all  # 返回处理后的拼接图像
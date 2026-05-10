import json
import os

from tqdm import tqdm
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL

import cv2
import pyarrow.parquet as pq

from examples.train_utils_offline import generate_traj, generate_trajs, read_local_trajs, eval_agent


def _scalar_training_kv(update_info, keys):
    """从一步 ``agent.update`` 的 info 里抽出若干标量，格式化为 ``k=v`` 片段列表。"""
    ui = {k: jax.device_get(v) for k, v in update_info.items()}
    out = []
    for key in keys:
        if key not in ui:
            continue
        a = np.asarray(ui[key]).squeeze()
        if a.shape != () and a.size != 1:
            continue
        out.append(f'{key}={float(a):.6g}')
    return out


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    elif variant.env == 'smoke':
        # smoke env 的 pixels 已经是 (H, W, C) 格式
        curr_image = obs["pixels"][..., 0] if obs["pixels"].ndim == 4 else obs["pixels"]
        if curr_image.ndim == 3 and curr_image.shape[-1] == 1:
            curr_image = np.repeat(curr_image, 3, axis=-1)
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    elif variant.env == 'smoke':
        # smoke env: 直接取 pixels，resize 到 224x224
        img = obs["pixels"]
        if img.ndim == 4:
            img = img[..., 0]  # 去掉最后的 1
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": np.zeros(14, dtype=np.float32),
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    elif variant.env == 'smoke':
        # smoke env: 从 state 中提取或生成假 qpos
        if "state" in obs:
            qpos = obs["state"][..., 0] if obs["state"].ndim >= 2 else obs["state"]
        else:
            qpos = np.zeros(14, dtype=np.float32)
    else:
        raise NotImplementedError()
    return qpos

# 定义轨迹交替训练循环（在线采样与离线/在线混合更新交替进行）
def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=False, shard_fn=None, agent_dp=None):
    # 从回放池（包含离线数据和新采集的在线数据）中获取训练数据的迭代器
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    
    # 如果定义了数据分片函数（多设备训练用），则对迭代器进行封装映射
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0  # 初始化总环境交互步数计数器
    i = 0                # 初始化训练梯度步数（Grad steps）计数器
    
    # 在 WandB 日志中记录初始状态：在线样本数、轨迹数和环境步数均为 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    # 使用 tqdm 创建进度条，最大进度为预设的总梯度更新步数
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        # 当训练步数未达到最大设定值时持续循环
        while i <= variant.max_steps:
            # 【关键步骤】调用当前 Actor 策略在环境中采集一条完整的轨迹
            if variant.env == 'smoke':
                traj = generate_traj()
            else:
                traj = collect_traj(variant, agent, env, i, agent_dp)
            # breakpoint()
            # 获取在线回放池当前的轨迹总数
            traj_id = online_replay_buffer._traj_counter
            # 将采集到的这条轨迹数据（包含 s, a, r, s'）存入在线回放池
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            # 累加总的环境交互步数
            total_env_steps += traj['env_steps']
            
            # 打印当前训练进度信息：池中数据量、轨迹总数及总步数
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            # 确定本次采集后要进行的梯度更新步数
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                # 若配置中指定了固定步数，则使用该值
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                # 否则根据本条轨迹的长度乘以一个放大系数（multi_grad_step）来决定更新次数
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            # 只有当在线回放池中的样本量超过“预热”阈值时，才开始进行网络更新
            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # 在进行第一次更新前，先对初始模型做一次性能评估（可视化）
                    # if i == 0:
                    if i == -50:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            # 执行控制评估（如在评估环境中跑一遍看成功率）
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            # 调用 agent 自带的评估函数（可能包含 Q 值检查等）
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # 【核心更新】从回放池迭代器中获取下一批次（Batch）训练数据
                    batch = next(replay_buffer_iterator)
                    # 执行 agent 的更新逻辑（内部包含算法 1 中的 Q^A, Q^w, π^w 的梯度更新）
                    update_info = agent.update(batch)

                    pbar.update() # 更新进度条
                    i += 1        # 训练步数计数器加 1
                        
                    # 达到日志记录间隔时，将训练信息上传到 WandB
                    if i % variant.log_interval == 0:
                        # 将 JAX 格式的显存数据转换到 CPU 内存上
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0: # 记录标量数据（如 Loss）
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2: # 记录直方图数据（如权重分布）
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        
                        # 记录当前池子大小、本次探索轨迹的回报以及是否成功
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    # 达到评估间隔时，在 eval_env 中进行正式的策略评估
                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            pass
                            # agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # 达到保存间隔时，将模型权重保存为 Checkpoint 文件
                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env or variant.env == 'smoke':
        obs, _ = env.reset()
    breakpoint()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        
        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if i == 0:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
            
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env or variant.env == 'smoke':
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1 
    }

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env or variant.env == 'smoke':
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                rng, key = jax.random.split(rng)
                assert agent_dp is not None
                
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                
                
                if i == 0:
                    # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                    noise = jax.random.normal(rng, (1, 50, 32))
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                    
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env or variant.env == 'smoke':
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)

# 定义纯训练循环（不包含在线采样）
def training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, perform_control_evals=False, shard_fn=None, agent_dp=None):
    # 从回放池（包含离线数据和新采集的在线数据）中获取训练数据的迭代器
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    
    # 如果定义了数据分片函数（多设备训练用），则对迭代器进行封装映射
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0  # 初始化总环境交互步数计数器
    i = 0                # 初始化训练梯度步数（Grad steps）计数器
    
    # 在 WandB 日志中记录初始状态：在线样本数、轨迹数和环境步数均为 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    # 使用 tqdm 创建进度条，最大进度为预设的总梯度更新步数
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        # 当训练步数未达到最大设定值时持续循环
        while i <= variant.max_steps:
            # 【关键步骤】调用当前 Actor 策略在环境中采集一条完整的轨迹
            if variant.env == 'smoke':
                traj = generate_traj()
            else:
                traj = collect_traj(variant, agent, env, i, agent_dp)
            # 获取在线回放池当前的轨迹总数
            traj_id = online_replay_buffer._traj_counter
            # 将采集到的这条轨迹数据（包含 s, a, r, s'）存入在线回放池
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            # 累加总的环境交互步数
            total_env_steps += traj['env_steps']
            
            # 打印当前训练进度信息：池中数据量、轨迹总数及总步数
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            # 确定本次采集后要进行的梯度更新步数
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                # 若配置中指定了固定步数，则使用该值
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                # 否则根据本条轨迹的长度乘以一个放大系数（multi_grad_step）来决定更新次数
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            # 只有当在线回放池中的样本量超过“预热”阈值时，才开始进行网络更新
            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # 在进行第一次更新前，先对初始模型做一次性能评估（可视化）
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            # 执行控制评估（如在评估环境中跑一遍看成功率）
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            # 调用 agent 自带的评估函数（可能包含 Q 值检查等）
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # 【核心更新】从回放池迭代器中获取下一批次（Batch）训练数据
                    batch = next(replay_buffer_iterator)
                    # 执行 agent 的更新逻辑（内部包含算法 1 中的 Q^A, Q^w, π^w 的梯度更新）
                    update_info = agent.update(batch)

                    pbar.update() # 更新进度条
                    i += 1        # 训练步数计数器加 1
                        
                    # 达到日志记录间隔时，将训练信息上传到 WandB
                    if i % variant.log_interval == 0:
                        # 将 JAX 格式的显存数据转换到 CPU 内存上
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0: # 记录标量数据（如 Loss）
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2: # 记录直方图数据（如权重分布）
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        
                        # 记录当前池子大小、本次探索轨迹的回报以及是否成功
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    # 达到评估间隔时，在 eval_env 中进行正式的策略评估
                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # 达到保存间隔时，将模型权重保存为 Checkpoint 文件
                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

# 直接读取本地数据的训练逻辑
def read_collect_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                               perform_control_evals=False, shard_fn=None, agent_dp=None):
    """
    先用 ``generate_trajs`` 批量生成合成轨迹并逐条写入 buffer，再仅从 replay buffer 采样做离线式更新（不再 ``collect_traj``）。
    """
    num_prefill = int(variant.get('offline_prefill_trajs', 100))  # 预填充轨迹条数，可用 variant.offline_prefill_trajs 覆盖，默认 100

    total_env_steps = 0  # 累计「环境交互步数」统计（各条 traj 的 env_steps 之和，仅用于日志）
    # prefill_trajs = generate_trajs(num_trajs=num_prefill)
    data_path = os.environ.get(
        "LEROBOT_DATA_ROOT",
        "/root/storage/CODE/txy/dsrl_pi0_lx/openpi/dataset/0316_pouring_water_easy",
    )
    prefill_trajs = read_local_trajs(data_path, max_episodes=num_prefill, pixel_h=64, pixel_w=64)
    for traj in tqdm(prefill_trajs, desc='prefill replay buffer'):
        add_online_data_to_buffer(variant, traj, online_replay_buffer)
        total_env_steps += traj['env_steps']

    print(  # 预填充结束打印摘要，便于确认容量是否足够 start_online_updates 等阈值
        f'Prefill done: {num_prefill} trajs, buffer size={len(online_replay_buffer)}, '
        f'total_env_steps={total_env_steps}'
    )

    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)  # 无限迭代器：均匀采样 batch_size 条 transition
    if shard_fn is not None:  # 多设备时沿 batch 维切分到各 JAX device
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    traj_id = online_replay_buffer._traj_counter - 1  # 最后一条已写入轨迹的 0-based 索引（用于日志中的 traj 计数）

    wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=0)  # 记录初始 buffer 中样本条数
    wandb_logger.log({'num_online_trajs': traj_id + 1}, step=0)  # 记录已写入轨迹段数（traj_id+1 = 轨迹总数）
    wandb_logger.log({'env_steps': total_env_steps}, step=0)  # 记录预填充累计环境步数

    with tqdm(total=variant.max_steps, initial=0) as pbar:  # 进度条总长 = 规划梯度更新总步数
        for i in range(variant.max_steps):  # 纯离线训练：共执行 max_steps 次梯度更新
            if i == 0:  # 仅在第一步做一次「初始 checkpoint」评估（与原先 alternating 循环语义对齐）
                print('performing evaluation for initial checkpoint')
                if perform_control_evals:  # 若开启：在真实/仿真环境里跑 pi0+SAC 控制评估
                    perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                if hasattr(agent, 'perform_eval'):  # 若 learner 实现了可选评估（如 Q 诊断）
                    pass
                    # agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

            batch = next(replay_buffer_iterator)  # 从 buffer 取下一个 mini-batch
            update_info = agent.update(batch)  # SAC 一步更新（actor/critic/temperature），无 collect_traj

            pbar.update(1)  # 进度条前进一格

            if i % variant.log_interval == 0:  # 按间隔写训练标量/直方图到 W&B
                update_info = {k: jax.device_get(v) for k, v in update_info.items()}  # JAX 数组拉回 CPU 便于记录
                for k, v in update_info.items():
                    if v.ndim == 0:  # 标量 loss 等
                        wandb_logger.log({f'training/{k}': v}, step=i)
                    elif v.ndim <= 2:  # 可展开为直方图的向量
                        wandb_logger.log_histogram(f'training/{k}', v, i)

                wandb_logger.log({
                    'replay_buffer_size': len(online_replay_buffer),  # 当前 buffer 占用长度（本循环中不变）
                }, i)

            if i % variant.eval_interval == 0 and i > 0:  # 跳过 i==0（已在上方做过初始 eval），周期性评估
                wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)  # 同步样本数指标
                wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)  # 轨迹段数（离线阶段为常数）
                wandb_logger.log({'env_steps': total_env_steps}, step=i)  # 环境步累计（预填充常数）
                if perform_control_evals:
                    perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                if hasattr(agent, 'perform_eval'):
                    pass
                    # agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

            if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0 and i > 0:  # i>0 避免与「仅初始化权重」重复存盘
                agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)  # 按间隔保存 SAC checkpoint
                loss_bits = _scalar_training_kv(
                    update_info,
                    (
                        'critic_loss',
                        'actor_loss',
                        'temperature_loss',
                        'entropy',
                        'temperature',
                        'q_pi_in_actor',
                        'target_q',
                    ),
                )
                tail = ' | '.join(loss_bits)
                print(  # 存盘时单行摘要，便于在真机/日志里快速扫训练健康度
                    '[checkpoint] '
                    f'step={i} | out={variant.outputdir} | '
                    f'buf={len(online_replay_buffer)} | bs={variant.batch_size} | '
                    f'gamma={getattr(variant, "discount", "n/a")}'
                    + (' | ' + tail if tail else '')
                )

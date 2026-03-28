# RobotEnv 接口规范（面向 real 训练链路）

本文档定义了当前项目在 real 训练流程中对 `RobotEnv` 的最小接口要求。
目标是让你在接入非 DROID / 非 Franka 机械臂时，有一份可直接对照实现的契约。

适用代码路径：
- `examples/train_real.py`
- `examples/train_utils_real.py`


## 1. 最小必需接口

你的环境类至少需要实现以下方法：

1. `reset(...)`
2. `step(action)`
3. `get_observation()`

当前训练代码不会调用 Gym 的 `observation_space` / `action_space`，
因此只要上述 3 个方法语义正确即可启动训练。


## 2. 方法契约

### 2.1 `reset(...)`

推荐签名：

```python
def reset(self, randomize: bool = False):
    ...
```

要求：
- 应让机械臂和夹爪回到可执行状态（安全姿态/起始姿态）。
- 允许被频繁调用：
  - 每条轨迹开始时会调用一次。
  - 每条轨迹结束后还会调用一次。
- 如果失败，应抛出异常（不要静默失败）。


### 2.2 `step(action)`

推荐签名：

```python
def step(self, action: np.ndarray):
    ...
```

要求：
- 输入 `action` 在当前流程中是 1D 向量，默认长度为 8。
- 语义默认是：
  - 前 7 维：机械臂关节控制量（通常 joint velocity 或等价映射）
  - 第 8 维：夹爪控制（已在上游被二值化为 0/1）
- 上游已做 `[-1, 1]` 裁剪，但你仍应在底层做安全检查。
- 可返回任意对象（当前训练循环不使用返回值），但建议返回执行状态字典。


### 2.3 `get_observation()`

推荐签名：

```python
def get_observation(self) -> dict:
    ...
```

这是最关键接口。当前 `examples/train_utils_real.py` 会按 DROID 风格解析。

必须返回形如：

```python
{
    "image": {
        # key 里需要包含相机 ID 字符串，且外部相机/腕部相机 key 中需要包含 "left"
        "<left_camera_id>_left":  np.ndarray(...),
        "<right_camera_id>_left": np.ndarray(...),
        "<wrist_camera_id>_left": np.ndarray(...),
        ...
    },
    "robot_state": {
        "cartesian_position": list/ndarray,   # 6 维（位置+姿态）
        "joint_positions":   list/ndarray,    # 7 维
        "gripper_position":  float,           # 标量
    }
}
```

要求细节：
- 图像可为 `HWC`，支持 3 通道或 4 通道（上游会截去 alpha）。
- 当前上游会把图像从 BGR 转 RGB；如果你本来就是 RGB，不要重复转换。
- `joint_positions` 与 `gripper_position` 的尺度应与控制定义一致，且在轨迹内连续。


## 3. 与当前训练代码的耦合点

### 3.1 相机键名规则

当前 `_extract_observation(...)` 使用以下逻辑找图像：
- 遍历 `obs_dict["image"]` 的键
- 键中包含 `left_camera_id/right_camera_id/wrist_camera_id`
- 同时键中包含字符串 `"left"`

如果你的相机系统键名不同，建议在 `RobotEnv.get_observation()` 内先做一次映射，
不要把差异泄漏到训练逻辑层。


### 3.2 机器人状态字段

当前代码显式读取：
- `obs_dict["robot_state"]["joint_positions"]`
- `obs_dict["robot_state"]["gripper_position"]`
- `obs_dict["robot_state"]["cartesian_position"]`

缺任一字段都会导致训练阶段报错。


### 3.3 控制频率

当前 real 训练循环按 15 Hz 节拍执行 `step`。
若你的底层驱动频率不同（例如 100 Hz），建议在 `RobotEnv.step()` 内做插值或速度限幅，
保证上层接口仍是 15 Hz 的离散控制语义。


## 4. 推荐的增强接口（非必须）

建议额外实现以下方法，便于运维与故障恢复：

1. `close()`：释放相机、机器人连接、线程资源。
2. `health_check()`：返回硬件在线状态、相机状态、急停状态。
3. `emergency_stop()`：紧急停机。
4. `set_control_mode(mode)`：在 position/velocity 等模式间切换。


## 5. 最小自测脚本（接入新机械臂前必跑）

```python
import numpy as np

env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

print("[检查] reset")
env.reset()

print("[检查] get_observation")
obs = env.get_observation()
print("顶层键:", obs.keys())
print("image键数量:", len(obs["image"]))
print("robot_state键:", obs["robot_state"].keys())

print("[检查] step")
for i in range(3):
    action = np.zeros(8, dtype=np.float32)
    action[-1] = 1.0 if i % 2 == 0 else 0.0
    env.step(action)
    obs2 = env.get_observation()
    print(f"step {i}: joint[0]={obs2['robot_state']['joint_positions'][0]:.4f}, gripper={obs2['robot_state']['gripper_position']}")

print("[完成] RobotEnv 最小接口可用")
```


## 6. 新品牌机械臂接入建议

为了尽量少改训练代码，推荐采用“适配器”模式：

1. 新建一个类（例如 `YourBrandRobotEnv`）直接实现本文档接口。
2. 在该类内部对接厂商 SDK（读状态、读相机、发动作）。
3. 在 `get_observation()` 中统一输出 DROID 风格字段。
4. 仅在 `examples/train_real.py` 中切换 `env` 实例化，不改训练主循环。

这样能把硬件差异限制在一个文件内，后续维护成本最低。
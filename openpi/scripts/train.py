import dataclasses  # 导入 dataclasses，用于不可变状态对象的复制与配置序列化。
import functools  # 导入 functools，用于偏函数封装训练步骤函数。
import logging  # 导入 logging，用于输出结构化训练日志。
import platform  # 导入 platform，用于记录当前运行主机信息。
from typing import Any  # 导入 Any，用于宽泛类型标注（如策略元信息）。

import etils.epath as epath  # 导入 epath，提供统一路径操作接口（本地/云端兼容）。
import flax.nnx as nnx  # 导入 flax.nnx，作为模型状态与梯度计算框架。
from flax.training import common_utils  # 导入 common_utils，用于跨设备聚合指标。
import flax.traverse_util as traverse_util  # 导入 traverse_util，用于树结构扁平化/反扁平化。
import jax  # 导入 jax，用于随机数、JIT、分布式与数组计算。
import jax.experimental  # 导入 jax.experimental，保留实验性命名空间依赖。
import jax.numpy as jnp  # 导入 jax.numpy，以 NumPy 风格进行张量运算。
import numpy as np  # 导入 numpy，用于日志图片拼接时转普通数组。
import optax  # 导入 optax，用于优化器更新与范数统计。
import tqdm_loggable.auto as tqdm  # 导入 tqdm_loggable，输出可记录的进度条。
import wandb  # 导入 wandb，用于实验追踪、指标与可视化记录。

import openpi.models.model as _model  # 导入模型接口定义（观测、动作、损失计算等）。
import openpi.shared.array_typing as at  # 导入数组类型检查工具与装饰器。
import openpi.shared.nnx_utils as nnx_utils  # 导入 NNX 辅助工具（路径过滤、状态映射等）。
import openpi.training.checkpoints as _checkpoints  # 导入 checkpoint 管理逻辑（保存/恢复）。
import openpi.training.config as _config  # 导入训练配置入口（CLI 解析与预设配置）。
import openpi.training.data_loader as _data_loader  # 导入数据加载器构建逻辑。
import openpi.training.optimizer as _optimizer  # 导入优化器与学习率策略构建逻辑。
import openpi.training.sharding as sharding  # 导入并行分片与 mesh 管理工具。
import openpi.training.utils as training_utils  # 导入训练通用工具（状态结构与可视化信息）。
import openpi.training.weight_loaders as _weight_loaders  # 导入权重加载器接口（基座模型初始化）。


def init_logging():  # 定义日志初始化函数，统一日志等级和输出格式。
    """Custom logging format for better readability."""  # 说明：自定义日志格式以提升可读性。
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}  # 将长等级名映射为单字符标签。

    class CustomFormatter(logging.Formatter):  # 定义自定义格式化器，重写等级显示逻辑。
        def format(self, record):  # 重写 format，在最终格式化前修改 record 内容。
            record.levelname = level_mapping.get(record.levelname, record.levelname)  # 将标准等级名替换为缩写。
            return super().format(record)  # 调用父类逻辑，按模板生成最终日志字符串。

    formatter = CustomFormatter(  # 创建格式化器实例并指定时间、等级、文件与行号模板。
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",  # 设置日志主模板，便于定位问题来源。
        datefmt="%H:%M:%S",  # 设置时间格式为时:分:秒。
    )  # 完成格式化器构建。

    logger = logging.getLogger()  # 获取根 logger，确保全局日志行为一致。
    logger.setLevel(logging.INFO)  # 将根 logger 等级设为 INFO。
    logger.handlers[0].setFormatter(formatter)  # 为默认处理器绑定自定义格式。


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):  # 定义 wandb 初始化函数，支持新建和续训两种模式。
    if not enabled:  # 如果用户关闭 wandb，则进入禁用模式。
        wandb.init(mode="disabled")  # 初始化为 disabled，避免后续 log 报错。
        return  # 提前返回，不再执行在线追踪初始化。

    ckpt_dir = config.checkpoint_dir  # 读取本次实验对应的 checkpoint 目录。
    if not ckpt_dir.exists():  # 若目录不存在，说明环境或参数配置异常。
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")  # 抛出异常，阻止无效训练继续。
    if resuming:  # 若为断点续训，则复用之前的 wandb run。
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()  # 从 checkpoint 目录读取历史 run id。
        wandb.init(id=run_id, resume="must", project=config.project_name)  # 强制以该 run id 恢复记录。
    else:  # 否则为新实验，创建新的 wandb run。
        wandb.init(  # 初始化新 run，并上传核心配置。
            name=config.exp_name,  # 设置实验展示名。
            config=dataclasses.asdict(config),  # 将 dataclass 配置转字典后写入 wandb 配置页。
            project=config.project_name,  # 指定 wandb project 名称。
        )  # 完成 run 初始化。
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)  # 将新 run 的 id 写入磁盘，便于后续 resume。

    if log_code:  # 若启用代码快照上传，则记录源码版本。
        wandb.run.log_code(epath.Path(__file__).parent.parent)  # 上传脚本上级目录代码到 wandb。


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:  # 定义权重加载与校验函数，保证结构一致后返回可用子集。
    """Loads and validates the weights. Returns a loaded subset of the weights."""  # 说明：加载权重并验证，返回实际加载到的参数子树。
    loaded_params = loader.load(params_shape)  # 按目标参数结构请求外部权重加载器返回参数。
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)  # 校验加载结果与目标结构在键、形状、dtype 上一致。

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.  # 说明：剔除纯形状占位，确保返回值仅包含真实已加载张量。
    return traverse_util.unflatten_dict(  # 将过滤后的扁平字典还原为嵌套参数树。
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}  # 仅保留非 ShapeDtypeStruct 的叶子项。
    )  # 返回过滤后的已加载参数。


@at.typecheck  # 启用运行时类型检查，尽早暴露参数类型不匹配问题。
def init_train_state(  # 定义训练状态初始化函数。
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool  # 输入训练配置、初始化随机种子、设备 mesh 与是否续训标志。
) -> tuple[training_utils.TrainState, Any]:  # 返回训练状态（或其 shape）及其分片描述。
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)  # 根据配置构建优化器变换链。

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:  # 定义内部 init，用于 eval_shape 与 jit 真正初始化共用。
        rng, model_rng = jax.random.split(rng)  # 拆分随机数，分离后续流程与模型初始化随机性。
        # initialize the model (and its parameters).  # 说明：创建模型对象及其初始参数。
        model = config.model.create(model_rng)  # 使用模型配置创建模型实例。

        # Merge the partial params into the model.  # 说明：若提供外部权重子集，则将其合并进模型状态。
        if partial_params is not None:  # 若存在预加载参数，则执行参数覆盖。
            graphdef, state = nnx.split(model)  # 拆分模型为图定义与可变状态。
            # This will produce an error if the partial params are not a subset of the state.  # 说明：若键不匹配将抛错，防止静默错误覆盖。
            state.replace_by_pure_dict(partial_params)  # 将纯字典形式参数写回状态对象。
            model = nnx.merge(graphdef, state)  # 重新合并得到更新后的模型。

        params = nnx.state(model)  # 提取模型参数状态树。
        # Convert frozen params to bfloat16.  # 说明：冻结参数转为 bfloat16 以降低显存占用。
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))  # 仅对冻结参数应用 dtype 转换。

        return training_utils.TrainState(  # 构造并返回训练状态对象。
            step=0,  # 初始训练步设为 0。
            params=params,  # 写入模型参数。
            model_def=nnx.graphdef(model),  # 保存模型图定义，后续可与参数合并复原模型。
            tx=tx,  # 保存优化器变换对象。
            opt_state=tx.init(params.filter(config.trainable_filter)),  # 仅基于可训练参数初始化优化器状态。
            ema_decay=config.ema_decay,  # 记录 EMA 衰减系数。
            ema_params=None if config.ema_decay is None else params,  # 若启用 EMA，则初始 EMA 参数等于当前参数。
        )  # 返回完整 TrainState。

    train_state_shape = jax.eval_shape(init, init_rng)  # 仅推断初始化输出结构，不实际分配完整参数张量。
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)  # 按 FSDP 规则为训练状态生成分片策略。

    if resume:  # 若是恢复训练，仅返回结构与分片，参数由 checkpoint 恢复。
        return train_state_shape, state_sharding  # 直接返回 shape 状态与 sharding 描述。

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())  # 加载并校验预训练权重子集。
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())  # 构建全复制分片规则（用于输入参数）。

    # Initialize the train state and mix in the partial params.  # 说明：通过 jit 初始化并注入预加载参数。
    train_state = jax.jit(  # 对 init 编译，确保初始化与分片在设备端高效执行。
        init,  # 目标初始化函数。
        donate_argnums=(1,),  # donate 第二个参数（partial_params）以减少内存拷贝。
        in_shardings=replicated_sharding,  # 指定输入分片规则为全复制。
        out_shardings=state_sharding,  # 指定输出训练状态的目标分片布局。
    )(init_rng, partial_params)  # 传入随机种子与预加载参数执行初始化。

    return train_state, state_sharding  # 返回初始化完成的训练状态与其分片描述。

def peek_input(observation, action):
    '''
    把observation和action转化成numpy格式
    然后把observation的图像存储到/root/storage/CODE/txy/dsrl_pi0_lx/test/peek_input文件夹中
    把action的文本存储到/root/storage/CODE/txy/dsrl_pi0_lx/test/peek_input/action.txt文件中
    输入 (Input):
        - observation: 包含图像和状态的 JAX 对象。
            - observation.images: 字典，键为视角名称（如 'base_0_rgb'），值为 (B, H, W, C) 张量。
            - observation.state: 机器人当前的关节状态或端到端状态张量。
        - action: 模型预测或 Ground Truth 的动作序列张量，通常形状为 (B, Horizon, Action_Dim)。

    输出 (Output):
        在路径 `/root/storage/CODE/txy/dsrl_pi0_lx/test/peek_input/` 下生成以下文件：
        - action.txt: 易读的文本格式动作数组。
        - action.npy & state.npy: 原始数值的二进制文件，方便后续用脚本加载分析。
        - *.png: 自动反归一化后的观察图像，用于确认模型“看”到的图像是否正常。
        - normalization_report.json: 包含所有输入项的 min/max/mean/std 及推断值域的详细统计报告。
        - .peek_done: 标记文件，存在时将跳过后续保存动作。
    '''
    import json
    from PIL import Image # 导入图像处理库，用于保存图片

    # 定义输出目录路径（使用 epath 这种跨平台路径库）
    output_dir = epath.Path("/root/storage/CODE/txy/dsrl_pi0_lx/test/peek_input")
    output_dir.mkdir(parents=True, exist_ok=True) # 递归创建目录，如果已存在则忽略

    done_flag = output_dir / ".peek_done"

    def _stats(x):
        arr = np.asarray(x)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def _infer_range(min_v, max_v):
        if min_v >= -1.0 and max_v <= 1.0:
            return "[-1, 1] (normalized)"
        if min_v >= 0.0 and max_v <= 1.0:
            return "[0, 1]"
        if min_v >= 0.0 and max_v <= 255.0:
            return "[0, 255]"
        return "out-of-common-range"

    def _save_input(obs_host, action_host):
        try:
            # 仅保存一次，避免每个 step 都写盘造成训练变慢。
            if done_flag.exists():
                return

            # 将 action 写入文本文件，便于检查形状和值域。
            action_array = np.asarray(action_host)
            print("[peek_input] 开始保存调试输入...")
            print(f"[peek_input] action shape: {action_array.shape}")
            print(f"[peek_input] action min/max/mean/std: {action_array.min():.6f}/{action_array.max():.6f}/{action_array.mean():.6f}/{action_array.std():.6f}")
            action_path = output_dir / "action.txt" # 拼接 action.txt 的完整存储路径
            with action_path.open("w", encoding="utf-8") as f: # 以写入模式打开文件
                f.write(f"action shape: {action_array.shape}\n") # 先记录 action 的维度信息（比如 [8, 50, 32]）
                f.write(np.array2string(action_array, separator=", ", threshold=10_000))
            print(f"[peek_input] 已写入: {action_path}")
            np.save(output_dir / "action.npy", action_array)

            # observation.to_dict() 在 callback 中是 dict；兼容直接传 Observation 的情况。
            if isinstance(obs_host, dict):
                images_dict = obs_host.get("image", {})
                state_value = obs_host.get("state")
            else:
                images_dict = obs_host.images
                state_value = obs_host.state

            report = {
                "action": _stats(action_array),
                "state": None,
                "images": {},
            }

            if state_value is not None:
                state_array = np.asarray(state_value)
                report["state"] = _stats(state_array)
                print(
                    "[peek_input] state min/max/mean/std: "
                    f"{state_array.min():.6f}/{state_array.max():.6f}/{state_array.mean():.6f}/{state_array.std():.6f}"
                )
                np.save(output_dir / "state.npy", state_array)

            # 遍历 observation 中的所有图像数据
            for image_name, image_value in images_dict.items():
                print(f"[peek_input] 处理图像视角: {image_name}")
                image_array = np.asarray(image_value) # 确保单个图像值是 numpy 格式
                if image_array.ndim == 4: # 如果是 batch 数据 (B, H, W, C)，默认取第一张图
                    image_array = image_array[0]

                orig_min = float(np.min(image_array))
                orig_max = float(np.max(image_array))
                orig_mean = float(np.mean(image_array))
                orig_std = float(np.std(image_array))
                inferred_range = _infer_range(orig_min, orig_max)
                print(
                    "[peek_input] image stats "
                    f"min/max/mean/std={orig_min:.6f}/{orig_max:.6f}/{orig_mean:.6f}/{orig_std:.6f}, "
                    f"range={inferred_range}"
                )

                report["images"][image_name] = {
                    "shape": list(image_array.shape),
                    "dtype": str(image_array.dtype),
                    "min": orig_min,
                    "max": orig_max,
                    "mean": orig_mean,
                    "std": orig_std,
                    "inferred_range": inferred_range,
                }

                # 处理非 uint8 格式的图像（通常是 float32 型的模型输入）
                if image_array.dtype != np.uint8:
                    image_min = float(np.min(image_array)) # 计算当前图像像素最小值
                    image_max = float(np.max(image_array)) # 计算当前图像像素最大值

                    # 情况1：如果是 [-1, 1] 范围的归一化数据（Pi0 常用）
                    if image_min >= -1.0 and image_max <= 1.0:
                        # 映射公式：(x + 1) * 0.5 * 255
                        image_array = ((np.clip(image_array, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)
                    # 情况2：如果是 [0, 1] 范围的数据
                    elif image_min >= 0.0 and image_max <= 1.0:
                        image_array = (np.clip(image_array, 0.0, 1.0) * 255.0).astype(np.uint8)
                    # 情况3：超出上述范围但又是浮点数，直接截断到 [0, 255]
                    else:
                        image_array = np.clip(image_array, 0.0, 255.0).astype(np.uint8)

                # 替换路径中的斜杠防止文件名错误，并保存为 png
                image_path = output_dir / f"{image_name.replace('/', '_')}.png"
                Image.fromarray(image_array).save(image_path) # 将数组转回 PIL Image 对象并写入磁盘
                print(f"[peek_input] 已保存图像: {image_path}")

            print("[peek_input] 保存完成。")
        except Exception as e:
            # 不让 callback 抛异常中断训练流程。
            print(f"[peek_input] 保存失败: {e}")
            return

        report_path = output_dir / "normalization_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[peek_input] 已写入归一化报告: {report_path}")

        done_flag.write_text("done", encoding="utf-8")
        print("[peek_input] 首次保存完成，后续 step 将跳过写盘。")

    jax.debug.callback(_save_input, observation.to_dict(), action)
    print("Callback has been staged. Files will appear once the device executes this op.")

@at.typecheck  # 启用类型检查，约束 train_step 输入输出结构。
def train_step(  # 定义单步训练逻辑（前向、反向、优化、EMA、指标统计）。
    config: _config.TrainConfig,  # 训练配置对象。
    rng: at.KeyArrayLike,  # 当前随机数种子。
    state: training_utils.TrainState,  # 当前训练状态（参数、优化器状态、步数等）。
    batch: tuple[_model.Observation, _model.Actions],  # 一个 batch 的观测与动作监督信号。
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:  # 返回更新后的状态和日志指标。
    model = nnx.merge(state.model_def, state.params)  # 将模型定义与参数合并成可执行模型。
    model.train()  # 切换到训练模式（启用训练期行为）。

    @at.typecheck  # 对损失函数同样启用类型检查。
    def loss_fn(  # 定义用于自动求导的标量损失函数。
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions  # 明确损失函数输入类型。
    ):  # 结束函数签名。
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)  # 计算每个样本或时间片段的损失。
        return jnp.mean(chunked_loss)  # 取均值得到标量损失，便于反向传播。

    train_rng = jax.random.fold_in(rng, state.step)  # 将当前 step 混入随机数，保证每步随机性稳定可复现。
    observation, actions = batch  # 解包 batch 为观测与动作标签。
    # peek_input(observation, actions)  

    # Filter out frozen params.  # 说明：只对可训练参数求导。
    diff_state = nnx.DiffState(0, config.trainable_filter)  # 指定 arg0(model) 中可参与求导的参数过滤器。
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)  # 执行前向并计算损失与梯度。

    params = state.params.filter(config.trainable_filter)  # 提取可训练参数子树。
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)  # 用优化器根据梯度生成参数更新量与新优化器状态。
    new_params = optax.apply_updates(params, updates)  # 将更新量应用到可训练参数。

    # Update the model in place and return the new full state.  # 说明：将可训练参数写回完整模型，生成新全量参数。
    nnx.update(model, new_params)  # 原地更新模型中的可训练参数节点。
    new_params = nnx.state(model)  # 提取更新后的全量参数状态。

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)  # 生成基础新状态（步数+1、参数和优化器状态更新）。
    if state.ema_decay is not None:  # 若启用 EMA，则同步更新 EMA 参数。
        new_state = dataclasses.replace(  # 复制状态并写入新的 ema_params。
            new_state,  # 基于当前新状态继续更新。
            ema_params=jax.tree.map(  # 对参数树逐叶执行 EMA 公式。
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params  # EMA: ema = decay*old + (1-decay)*new。
            ),  # 完成树映射。
        )  # 返回带 EMA 的状态。

    # Filter out params that aren't kernels.  # 说明：仅统计 kernel 范数，排除 bias/scale/embedding 等参数。
    kernel_params = nnx.state(  # 根据过滤条件从模型中提取用于统计的参数子集。
        model,  # 目标模型对象。
        nnx.All(  # 组合多个过滤条件。
            nnx.Param,  # 只保留参数节点。
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),  # 按路径排除偏置、缩放和嵌入项。
            lambda _, x: x.value.ndim > 1,  # 再过滤掉一维参数，只保留矩阵及以上维度。
        ),  # 结束过滤器组合。
    )  # 得到 kernel 参数树。
    info = {  # 组织训练日志指标字典。
        "loss": loss,  # 当前 batch 的平均损失。
        "grad_norm": optax.global_norm(grads),  # 梯度全局范数。
        "param_norm": optax.global_norm(kernel_params),  # kernel 参数全局范数。
    }  # 完成指标字典构建。
    return new_state, info  # 返回更新后的状态与日志信息。


def main(config: _config.TrainConfig):  # 定义训练主函数，串联初始化、训练循环、保存和收尾。
    init_logging()  # 初始化日志系统。
    logging.info(f"Running on: {platform.node()}")  # 打印主机名，便于区分多机训练日志。

    if config.batch_size % jax.device_count() != 0:  # 检查全局 batch 是否可被设备数整除。
        raise ValueError(  # 若不整除则抛错，避免数据并行切分异常。
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."  # 给出明确错误信息。
        )  # 结束异常构造。

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))  # 设置 JAX 编译缓存目录，减少重复编译开销。

    rng = jax.random.key(config.seed)  # 根据配置 seed 创建根随机种子。
    train_rng, init_rng = jax.random.split(rng)  # 拆分出训练与初始化两路随机数。

    mesh = sharding.make_mesh(config.fsdp_devices)  # 根据配置创建设备 mesh。
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))  # 定义数据按 DATA_AXIS 分片策略。
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())  # 定义全复制分片策略。

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(  # 初始化 checkpoint 管理器并判断是否处于恢复模式。
        config.checkpoint_dir,  # 传入当前实验的 checkpoint 路径。
        keep_period=config.keep_period,  # 指定保留周期策略。
        overwrite=config.overwrite,  # 指定是否覆盖已有实验目录。
        resume=config.resume,  # 指定是否尝试从历史 checkpoint 恢复。
    )  # 完成 checkpoint 初始化。
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)  # 初始化 wandb，模式由是否恢复决定。

    data_loader = _data_loader.create_data_loader(  # 按配置构建数据加载器。
        config,  # 传入训练配置。
        sharding=data_sharding,  # 指定 batch 张量的分片布局。
        shuffle=True,  # 训练阶段开启随机打乱。
    )  # 返回可迭代数据加载器。
    data_iter = iter(data_loader)  # 创建数据迭代器。
    batch = next(data_iter)  # 预取第一个 batch，用于日志与 shape 触发。
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")  # 打印首个 batch 的结构信息。

    # Log images from first batch to sanity check.  # 说明：记录首批图像，快速检查相机输入是否正确。
    images_to_log = [  # 构建要上传到 wandb 的可视化图像列表。
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))  # 将同一时刻多视角图像横向拼接后封装为 wandb.Image。
        for i in range(min(5, len(next(iter(batch[0].images.values())))))  # 最多记录 5 条样本，避免日志过大。
    ]  # 完成图像列表构建。
    wandb.log({"camera_views": images_to_log}, step=0)  # 在 step=0 上传首批可视化图像。

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)  # 初始化训练状态（或恢复模式下的 shape 状态）。
    jax.block_until_ready(train_state)  # 同步等待初始化完成，避免异步延迟影响后续计时。
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")  # 打印参数树结构信息。

    if resuming:  # 若为恢复训练，则从 checkpoint 读取完整状态。
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)  # 恢复参数、优化器状态及数据读取进度。

    ptrain_step = jax.jit(  # 编译并行训练步函数。
        functools.partial(train_step, config),  # 固定 config，只留下 rng/state/batch 作为运行时输入。
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),  # 指定输入分片：rng 复制、状态按 FSDP、数据按数据轴。
        out_shardings=(train_state_sharding, replicated_sharding),  # 指定输出分片：新状态按 FSDP、日志指标全复制。
        donate_argnums=(1,),  # donate state 参数，降低显存峰值与拷贝开销。
    )  # 得到编译后的训练步函数。

    start_step = int(train_state.step)  # 记录起始步（恢复训练时可能非 0）。
    pbar = tqdm.tqdm(  # 创建训练进度条。
        range(start_step, config.num_train_steps),  # 设置迭代区间。
        initial=start_step,  # 设置进度条初始位置。
        total=config.num_train_steps,  # 设置总步数。
        dynamic_ncols=True,  # 启用终端宽度自适应。
    )  # 完成进度条初始化。

    infos = []  # 用于缓存多个 step 的日志指标，按间隔统一求均值。
    for step in pbar:  # 遍历每一个训练步。
        with sharding.set_mesh(mesh):  # 进入 mesh 上下文，确保分片规则生效。
            train_state, info = ptrain_step(train_rng, train_state, batch)  # 执行一次并行训练步，得到新状态与本步指标。
        infos.append(info)  # 缓存本步指标。
        if step % config.log_interval == 0:  # 到达日志间隔时进行聚合与上报。
            stacked_infos = common_utils.stack_forest(infos)  # 将指标列表堆叠为树结构数组。
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))  # 对每个指标取均值并从设备取回主机。
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())  # 格式化为可读字符串。
            pbar.write(f"Step {step}: {info_str}")  # 在进度条上方打印当前汇总指标。
            wandb.log(reduced_info, step=step)  # 将汇总指标写入 wandb。
            infos = []  # 清空缓存，开始下一个统计窗口。
        batch = next(data_iter)  # 读取下一个 batch 供下一步训练。

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:  # 到达保存间隔或最后一步时触发保存。
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)  # 保存训练状态与数据迭代器状态。

    logging.info("Waiting for checkpoint manager to finish")  # 提示等待异步 checkpoint 写盘结束。
    checkpoint_manager.wait_until_finished()  # 阻塞直到所有保存任务完成。


if __name__ == "__main__":  # Python 脚本入口判断，仅直接运行时执行主流程。
    main(_config.cli())  # 从 CLI 解析配置后启动训练主函数。

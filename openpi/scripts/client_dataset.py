import argparse  # 解析命令行参数。
import json  # 读写 JSON 文本。
import random  # 随机抽样 episode。
from datetime import datetime  # 生成带时间戳的输出目录名。
from pathlib import Path  # 跨平台路径处理。

import av  # 读取 mp4 视频帧。
import matplotlib.pyplot as plt  # 绘制动作对比图。
import numpy as np  # 数值计算与数组处理。
import pyarrow.parquet as pq  # 读取 parquet 数据。
from openpi_client import websocket_client_policy  # WebSocket 推理客户端。


DATASET_PATH = Path("/root/storage/CODE/txy/dsrl_pi0_lx/openpi/dataset/0316_pouring_water_easy")  # 默认数据集路径。
OUTPUT_DIR = Path("./result/eval")  # 默认输出根目录（相对当前工作目录）。


def _episode_path(dataset_root: Path, episode_index: int) -> Path:
    """构造单条 episode 的 parquet 文件路径。

    输入:
    - dataset_root: 数据集根目录。
    - episode_index: episode 序号（整数）。

    输出:
    - Path: 对应的 parquet 文件路径。

    逻辑:
    - 将 episode 索引补零为 6 位并拼接到固定目录结构中。
    """
    return dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"  # 返回该 episode 的 parquet 路径。


def _list_episode_indices(dataset_root: Path) -> list[int]:
    """列出数据集中可用的 episode 索引。

    输入:
    - dataset_root: 数据集根目录。

    输出:
    - list[int]: 所有 episode 索引（升序）。

    逻辑:
    - 扫描 data/chunk-000 下的 episode_*.parquet，解析文件名中的编号。
    """
    data_dir = dataset_root / "data" / "chunk-000"  # 定位 parquet 存储目录。
    indices = []  # 初始化 episode 索引容器。
    for p in sorted(data_dir.glob("episode_*.parquet")):  # 按文件名排序遍历所有 episode parquet。
        indices.append(int(p.stem.split("_")[-1]))  # 从文件名提取末尾数字并转为 int。
    if not indices:  # 如果没有找到任何 episode 文件。
        raise FileNotFoundError(f"No episode parquet files found in {data_dir}")  # 抛出错误提示目录为空。
    return indices  # 返回解析得到的索引列表。


def _load_task_prompt(dataset_root: Path, task_index: int) -> str:
    """根据 task_index 从 tasks.jsonl 中查找文本提示词。

    输入:
    - dataset_root: 数据集根目录。
    - task_index: 任务编号。

    输出:
    - str: 对应任务文本；若未找到则返回空字符串。

    逻辑:
    - 顺序读取 meta/tasks.jsonl，匹配 task_index 后返回 task 字段。
    """
    tasks_file = dataset_root / "meta" / "tasks.jsonl"  # 构造任务映射文件路径。
    with tasks_file.open("r", encoding="utf-8") as f:  # 以 UTF-8 打开 tasks 文件。
        for line in f:  # 逐行读取 JSONL。
            item = json.loads(line)  # 解析每行 JSON。
            if item.get("task_index") == task_index:  # 判断是否匹配目标 task_index。
                return item.get("task", "")  # 返回任务文本，缺失时返回空串。
    return ""  # 未匹配到任务时返回空串。


def _read_video_frame_hwc_uint8(video_path: Path, target_frame_idx: int) -> np.ndarray:
    """读取视频中指定帧，返回 HWC uint8 RGB 图像。

    输入:
    - video_path: 视频文件路径。
    - target_frame_idx: 目标帧序号（从 0 开始）。

    输出:
    - np.ndarray: 形状为 (H, W, C) 的 uint8 RGB 图像。

    逻辑:
    - 使用 PyAV 逐帧解码，命中目标帧后转为 RGB numpy 数组返回。
    """
    with av.open(str(video_path)) as container:  # 打开视频容器。
        stream = container.streams.video[0]  # 获取第一个视频流。
        for i, frame in enumerate(container.decode(stream)):  # 顺序解码每一帧。
            if i == target_frame_idx:  # 命中目标帧。
                return frame.to_ndarray(format="rgb24")  # 转成 HWC RGB uint8 返回。
    raise IndexError(f"frame_idx={target_frame_idx} out of range for {video_path}")  # 目标帧越界时报错。


def _to_chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    """将 HWC 图像转换为 CHW uint8。

    输入:
    - image_hwc: HWC 格式图像数组。

    输出:
    - np.ndarray: CHW 格式 uint8 图像。

    逻辑:
    - 先保证 dtype 为 uint8，再把维度顺序从 (H, W, C) 改为 (C, H, W)。
    """
    if image_hwc.dtype != np.uint8:  # 若输入不是 uint8。
        image_hwc = image_hwc.astype(np.uint8)  # 转换到 uint8。
    return np.transpose(image_hwc, (2, 0, 1))  # 返回 CHW 排布。


def _build_observation(dataset_root: Path, ep_name: str, frame_idx: int, state: np.ndarray, prompt: str) -> dict:
    """构造一次推理请求的 observation 字典。

    输入:
    - dataset_root: 数据集根目录。
    - ep_name: episode 文件名主干（例如 episode_000017）。
    - frame_idx: 当前要取的帧序号。
    - state: 当前帧状态向量（14 维 float32）。
    - prompt: 当前任务文本。

    输出:
    - dict: 发送给 policy server 的 observation。

    逻辑:
    - 读取四路相机对应帧，转换为 CHW uint8，并按 AlohaInputs 期望键名封装。
    """
    videos_root = dataset_root / "videos" / "chunk-000"  # 视频根目录。
    cam_high = _to_chw_uint8(  # 读取并转换头部相机图像。
        _read_video_frame_hwc_uint8(videos_root / "observation.images.cam_head" / f"{ep_name}.mp4", frame_idx)  # 指定帧。
    )  # 完成转换。
    cam_low = _to_chw_uint8(  # 读取并转换低位相机图像。
        _read_video_frame_hwc_uint8(videos_root / "observation.images.cam_low" / f"{ep_name}.mp4", frame_idx)  # 指定帧。
    )  # 完成转换。
    cam_left = _to_chw_uint8(  # 读取并转换左腕相机图像。
        _read_video_frame_hwc_uint8(videos_root / "observation.images.cam_left" / f"{ep_name}.mp4", frame_idx)  # 指定帧。
    )  # 完成转换。
    cam_right = _to_chw_uint8(  # 读取并转换右腕相机图像。
        _read_video_frame_hwc_uint8(videos_root / "observation.images.cam_right" / f"{ep_name}.mp4", frame_idx)  # 指定帧。
    )  # 完成转换。

    return {  # 返回推理输入字典。
        "images": {  # 图像字段。
            "cam_high": cam_high,  # 头部视角。
            "cam_low": cam_low,  # 低位视角。
            "cam_left_wrist": cam_left,  # 左腕视角（服务端键名）。
            "cam_right_wrist": cam_right,  # 右腕视角（服务端键名）。
        },  # 图像字典结束。
        "state": state.astype(np.float32),  # 状态向量，确保为 float32。
        "prompt": prompt,  # 任务提示词。
    }  # 返回 observation。


def _merge_chunks(total_frames: int, chunk_records: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """把重叠 action chunk 对齐并融合成完整动作序列。

    输入:
    - total_frames: 本 episode 总帧数。
    - chunk_records: 预测块列表，每项为 (start_frame, action_chunk[chunk_len, 14])。

    输出:
    - merged: 形状 (total_frames, 14) 的融合动作（重叠区域按平均融合）。
    - mask: 形状 (total_frames,) 的布尔掩码，True 表示该帧有预测覆盖。

    逻辑:
    - 对每个 chunk 按起始帧写入到全局时间线上，并记录覆盖次数，最后做逐帧平均。
    """
    merged = np.zeros((total_frames, 14), dtype=np.float32)  # 初始化融合结果数组。
    counts = np.zeros((total_frames, 1), dtype=np.float32)  # 初始化每帧覆盖计数。
    for start, chunk in chunk_records:  # 遍历所有预测块。
        end = min(total_frames, start + chunk.shape[0])  # 计算该块在全局时间线的结束位置。
        valid = end - start  # 计算有效长度（防越界）。
        if valid <= 0:  # 若该块在边界外无有效数据。
            continue  # 跳过该块。
        merged[start:end] += chunk[:valid]  # 累加该块有效区域。
        counts[start:end] += 1.0  # 对应帧覆盖计数加一。
    mask = counts[:, 0] > 0  # 标记哪些帧至少被一个 chunk 覆盖。
    merged[mask] = merged[mask] / counts[mask]  # 对重叠区域按覆盖次数求平均。
    return merged, mask  # 返回融合动作和有效掩码。


def _plot_compare(
    action_real: np.ndarray,
    chunk_records: list[tuple[int, np.ndarray]],
    merged_action_vla: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    out_file: Path,
) -> dict:
    """绘制 14 维动作对比图（14 行 1 列）并返回每维误差指标。

    输入:
    - action_real: 真实动作，形状 (T, 14)。
    - chunk_records: 原始预测块列表，用于展示每个 chunk 的覆盖段。
    - merged_action_vla: 融合后的预测动作，形状 (T, 14)。
    - valid_mask: 有效预测掩码，形状 (T,)。
    - title: 图标题。
    - out_file: 图像保存路径。

    输出:
    - dict: 每个维度的 MAE/RMSE/L2 指标。

    逻辑:
    - 每一行对应一个动作维度，同时绘制真实曲线、所有 chunk 曲线和融合曲线。
    - 在每个子图右侧用科学计数法标注误差，保留小数点后 3 位。
    """
    fig, axes = plt.subplots(14, 1, figsize=(16, 42), sharex=True)  # 创建 14 行 1 列子图。
    metrics = {}  # 存放每个维度的误差指标。

    t = np.arange(action_real.shape[0])  # 全局时间轴。
    for d in range(14):  # 遍历 14 个动作维度。
        ax = axes[d] if isinstance(axes, np.ndarray) else axes  # 兼容 axes 类型。
        real_d = action_real[:, d]  # 取第 d 维真实动作。
        vla_d = merged_action_vla[:, d]  # 取第 d 维融合预测动作。
        valid_real = real_d[valid_mask]  # 取有预测覆盖区间的真实值。
        valid_vla = vla_d[valid_mask]  # 取有预测覆盖区间的预测值。
        mae = float(np.mean(np.abs(valid_vla - valid_real)))  # 计算 MAE。
        rmse = float(np.sqrt(np.mean((valid_vla - valid_real) ** 2)))  # 计算 RMSE。
        l2 = float(np.linalg.norm(valid_vla - valid_real))  # 计算 L2 范数。
        metrics[d] = {"mae": mae, "rmse": rmse, "l2": l2}  # 记录该维指标。

        ax.plot(t, real_d, label="action_real", linewidth=1.5, color="black")  # 画真实动作曲线。
        for i, (start, chunk) in enumerate(chunk_records):  # 遍历每个 chunk 并画在其时间窗口。
            end = min(action_real.shape[0], start + chunk.shape[0])  # 该 chunk 在全局时间线的结束点。
            x = np.arange(start, end)  # 当前 chunk 的横坐标范围。
            y = chunk[: end - start, d]  # 当前 chunk 第 d 维有效纵坐标。
            ax.plot(  # 绘制 chunk 曲线。
                x,  # 横坐标。
                y,  # 纵坐标。
                linewidth=0.9,  # 线宽较细。
                alpha=0.45,  # 半透明方便看重叠。
                color="tab:blue",  # chunk 统一蓝色。
                label="action_chunk" if i == 0 else None,  # 只给第一条 chunk 加图例标签。
            )  # 完成该 chunk 绘图。

        ax.plot(  # 绘制融合后的预测曲线。
            t[valid_mask],  # 仅在有效覆盖区间绘制。
            vla_d[valid_mask],  # 对应融合预测值。
            label="action_vla_merged",  # 图例标签。
            linewidth=1.2,  # 线宽。
            color="tab:red",  # 融合预测用红色。
        )  # 完成融合曲线绘制。
        ax.set_title(f"dim {d}")  # 设置子图标题。
        ax.grid(alpha=0.25)  # 开启浅色网格便于读数。
        ax.text(  # 在子图右侧标注误差。
            1.01,  # x 轴归一化坐标（右侧外一点）。
            0.5,  # y 轴归一化坐标（垂直居中）。
            f"MAE={mae:.3e}\\nRMSE={rmse:.3e}\\nL2={l2:.3e}",  # 科学计数法且保留 3 位小数。
            transform=ax.transAxes,  # 使用轴坐标系。
            va="center",  # 垂直居中对齐。
            fontsize=9,  # 字号。
        )  # 完成文本标注。

    first_ax = axes[0] if isinstance(axes, np.ndarray) else axes  # 获取首个子图用于提取图例。
    handles, labels = first_ax.get_legend_handles_labels()  # 提取图例句柄与标签。
    fig.legend(handles, labels, loc="upper center", ncol=3)  # 在整图上方放置统一图例。
    fig.suptitle(title)  # 设置整图标题。
    fig.tight_layout(rect=[0, 0, 1, 0.985])  # 自动布局并给总标题留边距。
    out_file.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    fig.savefig(out_file, dpi=150)  # 保存图片。
    plt.close(fig)  # 关闭图形释放内存。
    return metrics  # 返回每维误差指标。


def run_episode(
    client: websocket_client_policy.WebsocketClientPolicy,
    dataset_root: Path,
    episode_index: int,
    take_chunk: int,
    out_dir: Path,
) -> None:
    """评估单个 episode，并生成 14 维动作对比图与指标文件。

    输入:
    - client: 已连接的 WebSocket 推理客户端。
    - dataset_root: 数据集根目录。
    - episode_index: 当前评估 episode 索引。
    - take_chunk: 每次从 action_chunk 中取用的步数（例如 30）。
    - out_dir: 当前运行的输出目录。

    输出:
    - 无（副作用: 保存 png 图和 json 指标，并打印摘要日志）。

    逻辑:
    - 读取真实动作与状态，按步长 take_chunk 采样关键帧进行推理，得到多个 action_chunk。
    - 将所有 chunk 对齐到全局时间轴融合，最后与真实动作做对比并可视化。
    """
    ep_name = f"episode_{episode_index:06d}"  # 格式化 episode 名称。
    table = pq.read_table(_episode_path(dataset_root, episode_index))  # 读取该 episode parquet。
    total_frames = table.num_rows  # 获取该 episode 总帧数。

    action_real = np.asarray(table["action"].to_pylist(), dtype=np.float32)  # 读取完整真实动作序列。
    task_index = int(table["task_index"][0].as_py())  # 读取该 episode 的 task_index。
    prompt = _load_task_prompt(dataset_root, task_index)  # 读取对应 prompt 文本。

    chunk_records: list[tuple[int, np.ndarray]] = []  # 存放 (起始帧, 预测动作块)。
    for frame_idx in range(0, total_frames, take_chunk):  # 以 take_chunk 为步长取关键帧。
        state = np.asarray(table["observation.state"][frame_idx].as_py(), dtype=np.float32)  # 读取该帧状态。
        observation = _build_observation(dataset_root, ep_name, frame_idx, state, prompt)  # 构造推理输入。
        breakpoint() # observation['states'].shape=(14,)
        noise = np.random.uniform(-1.0, 1.0, size=(50, 14)).astype(np.float32)
        action_chunk = np.asarray(client.infer(observation, noise=noise)["actions"], dtype=np.float32)  # 调服务端推理得到 chunk。
        chunk_records.append((frame_idx, action_chunk[:, :14]))  # 仅保留前 14 维并记录起始帧。

    if not chunk_records:  # 理论上空 episode 才会出现该情况。
        print(f"[{ep_name}] empty chunk_records, skip.")  # 打印跳过信息。
        return  # 直接返回。

    action_real = action_real[:, :14]  # 真实动作也只取前 14 维对齐比较。
    merged_vla, valid_mask = _merge_chunks(total_frames, chunk_records)  # 融合多个重叠预测块。

    fig_path = out_dir / f"{ep_name}_compare.png"  # 生成该 episode 的图像保存路径。
    metrics = _plot_compare(  # 绘制并返回每维指标。
        action_real,  # 真实动作。
        chunk_records,  # 全部原始 chunk 记录。
        merged_vla,  # 融合后的预测动作。
        valid_mask,  # 有效覆盖掩码。
        title=(  # 整图标题。
            f"{ep_name} | frames={total_frames} | "  # 显示 episode 和总帧数。
            f"action_chunk={chunk_records[0][1].shape[0]} | "  # 显示模型每次输出 chunk 长度。
            f"take_chunk={take_chunk}"  # 显示采样步长。
        ),  # 标题结束。
        out_file=fig_path,  # 图片输出路径。
    )  # 完成绘图。

    summary = {  # 组织摘要指标。
        "episode_index": episode_index,  # 当前 episode 索引。
        "total_frames": int(total_frames),  # 总帧数。
        "evaluated_frames": int(valid_mask.sum()),  # 被预测覆盖的帧数。
        "num_chunks": int(len(chunk_records)),  # 预测块数量。
        "action_chunk": int(chunk_records[0][1].shape[0]),  # 单块动作长度（通常 50）。
        "take_chunk": int(take_chunk),  # 实际 take_chunk 参数。
        "metric_mean_mae": float(np.mean([m["mae"] for m in metrics.values()])),  # 14 维 MAE 平均值。
        "metric_mean_rmse": float(np.mean([m["rmse"] for m in metrics.values()])),  # 14 维 RMSE 平均值。
        "figure": str(fig_path),  # 对应图片路径。
    }  # 摘要结束。
    metrics_path = out_dir / f"{ep_name}_metrics.json"  # 指标 JSON 保存路径。
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")  # 写入指标文件。

    print(  # 打印评估摘要。
        f"[{ep_name}] total_frames={total_frames}, evaluated_frames={int(valid_mask.sum())}, "  # 打印帧统计。
        f"mean_mae={summary['metric_mean_mae']:.3e}, mean_rmse={summary['metric_mean_rmse']:.3e}"  # 科学计数法输出指标。
    )  # 结束摘要打印。
    print(f"[{ep_name}] figure saved: {fig_path}")  # 打印图片路径。
    print(f"[{ep_name}] metrics saved: {metrics_path}")  # 打印指标文件路径。


def main() -> None:
    """主流程入口。

    输入:
    - 来自命令行参数（dataset 路径、抽样数量、take_chunk、host/port、seed、输出目录）。

    输出:
    - 无（副作用: 在 ./result/eval/时间戳 目录写入多张对比图和指标 json）。

    逻辑:
    - 解析参数 -> 固定随机种子 -> 连接服务端 -> 随机抽 episode -> 逐条评估并保存结果。
    """
    parser = argparse.ArgumentParser(  # 创建参数解析器。
        description="Compare VLA-predicted actions vs dataset actions on random episodes."  # 参数说明文本。
    )  # 解析器创建完成。
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_PATH))  # 数据集路径参数。
    parser.add_argument("--data-review-number", type=int, default=10)  # 评估 episode 数量参数。
    parser.add_argument("--take-chunk", type=int, default=30)  # 每次取用 chunk 的步长参数。
    parser.add_argument("--host", type=str, default="localhost")  # 推理服务 host 参数。
    parser.add_argument("--port", type=int, default=8123)  # 推理服务端口参数。
    parser.add_argument("--seed", type=int, default=42)  # 随机种子参数。
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))  # 输出根目录参数。
    args = parser.parse_args()  # 解析命令行参数。

    random.seed(args.seed)  # 固定 Python 随机数种子。
    np.random.seed(args.seed)  # 固定 numpy 随机数种子。

    dataset_root = Path(args.dataset_path)  # 转换数据集路径对象。
    out_dir = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")  # 生成本次运行输出目录。
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录。

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)  # 初始化推理客户端。

    episode_indices = _list_episode_indices(dataset_root)  # 获取全部可用 episode 索引。
    sample_n = min(args.data_review_number, len(episode_indices))  # 防止抽样数量超过可用数量。
    picked = random.sample(episode_indices, sample_n)  # 随机无放回抽样 episode。

    print(f"dataset: {dataset_root}")  # 打印数据集路径。
    print(f"picked episodes ({sample_n}): {picked}")  # 打印抽样结果。
    print("action_chunk: 50 (from policy output)")  # 打印模型输出 chunk 长度说明。
    print(f"take_chunk: {args.take_chunk}")  # 打印当前 take_chunk。
    print(f"output_dir: {out_dir}")  # 打印输出目录。

    for ep_idx in picked:  # 逐条评估抽样到的 episode。
        run_episode(client, dataset_root, ep_idx, args.take_chunk, out_dir)  # 执行单条评估。


if __name__ == "__main__":  # 当脚本作为主程序运行时。
    main()  # 进入主流程。

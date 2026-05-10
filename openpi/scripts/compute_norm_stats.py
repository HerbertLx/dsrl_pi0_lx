"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)


"""计算并保存训练配置对应的数据归一化统计量。

用途：
- 根据给定的训练配置读取数据集。
- 统计 state 和 actions 的归一化参数。
- 将统计结果写入配置对应的 assets 目录，供训练时 Normalize 使用。

输入：
- config_name: str
  配置名。会传给 openpi.training.config.get_config(config_name)，用于解析
  数据源、模型维度、batch_size、num_workers、assets_dirs 等信息。
- max_frames: int | None
  可选。最多使用多少帧样本来估计统计量。
  - 为 None: 使用整套数据（按脚本中计算出的批次数）。
  - 为正整数: 仅使用前 max_frames 对应的数据量，常用于快速估计或调试。

输出：
- 统计对象 norm_stats，包含以下键：
  - "state"
  - "actions"
- 每个键下的统计字段来自 RunningStats.get_statistics()：
  - mean: 每个维度的均值
  - std: 每个维度的标准差
  - q01: 每个维度 1% 分位数
  - q99: 每个维度 99% 分位数

输出文件位置：
- 目录：config.assets_dirs / data_config.repo_id
- 文件名：norm_stats.json
- 最终路径形式：
  <assets_dirs>/<repo_id>/norm_stats.json

主流程概览：
1. 读取配置并构建 DataConfig（包含 repack_transforms、data_transforms 等）。
2. 根据数据类型选择数据加载器：
   - RLDS 数据：create_rlds_dataloader
   - LeRobot/Torch 数据：create_torch_dataloader
3. 在统计前应用与训练一致的“前置数据变换”：
   - repack_transforms.inputs
   - data_transforms.inputs
   - RemoveStrings（去掉字符串字段，避免数值统计报错）
4. 遍历批数据，分别对 state/actions 执行 RunningStats.update。
5. 汇总得到 norm_stats 并写入 norm_stats.json。

命令行参数（通过 tyro 自动生成）：
- 位置参数：
  - config_name
- 可选参数：
  - --max-frames <int>

示例：
cd /root/storage/CODE/txy/dsrl_pi0_lx/openpi
export HF_LEROBOT_HOME=/root/storage/CODE/txy/dsrl_pi0_lx/openpi/dataset
export CUDA_VISIBLE_DEVICES=2
python ./scripts/compute_norm_stats.py --config-name pi0_airbot_local

python ./scripts/compute_norm_stats.py pi0_airbot_local --max-frames 100

"""
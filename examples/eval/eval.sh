#!/usr/bin/env bash
# 从仓库根目录启动 eval，保证 ``from examples....`` 等包导入可用。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# 项目根优先加入 Python 搜索路径（修复直接 ``python /path/to/eval.py`` 时的 ModuleNotFoundError）
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# 与训练脚本一致的可选路径（eval 扩展 pi0 / 日志时可沿用）
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${REPO_ROOT}/openpi}"
export EXP="${EXP:-${REPO_ROOT}/logs}"

# 若本机装有 LIBERO 且需要从该路径导入，可取消注释或在外部 export：
# export PYTHONPATH="${PYTHONPATH}:/path/to/LIBERO"

# SAC checkpoint（可选覆盖）
# export SAC_CHECKPOINT_DIR="/path/to/checkpoint170000"

export PYTHONPATH=$PYTHONPATH:/root/storage/CODE/txy/dsrl_pi0_lx/LIBERO
export WANDB_API_KEY=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

export CUDA_VISIBLE_DEVICES=1,2

exec python -m examples.eval.eval "$@"

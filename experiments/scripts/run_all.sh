#!/bin/bash
# ============================================================
# 一键运行全部实验 (B2 + B3 + B4)
#
# 用法:
#   NPROC=1 bash run_all.sh          # 单卡
#   NPROC=4 bash run_all.sh          # 4卡
#   NPROC=1 bash run_all.sh --skip-data  # 跳过数据准备
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export NPROC=${NPROC:-1}
export MODEL_PATH="/root/cqx-code-generation/experiments/Qwen3-0.6B"

SKIP_DATA=false
for arg in "$@"; do
    if [ "$arg" = "--skip-data" ]; then
        SKIP_DATA=true
    fi
done

echo "============================================================"
echo "  CAFCL 全部实验"
echo "  模型: ${MODEL_PATH}"
echo "  GPU数: ${NPROC}"
echo "============================================================"

# ===== Step 0: 数据准备 =====
# if [ "$SKIP_DATA" = false ]; then
#     echo ""
#     echo "[Step 0] 数据准备..."
#     python "${PROJECT_DIR}/prepare_data.py" \
#         --save_dir "${PROJECT_DIR}/data" \
#         --seed 42 \
#         --threshold 0.45
#     echo ""
# fi

# 检查数据是否存在
for f in evol_66k_all.parquet evol_filtered.parquet curriculum_stage1.parquet curriculum_stage2.parquet curriculum_stage3.parquet; do
    if [ ! -f "${PROJECT_DIR}/data/${f}" ]; then
        echo "ERROR: 数据文件不存在: ${PROJECT_DIR}/data/${f}"
        echo "请先运行: python prepare_data.py --save_dir ./data"
        exit 1
    fi
done

# ===== Step 1: B2 =====
echo ""
echo "[Step 1/3] B2: 全量 66k + 随机顺序"
echo "============================================================"
bash "${SCRIPT_DIR}/run_b2_random.sh"

# ===== Step 2: B3 =====
echo ""
echo "[Step 2/3] B3: 筛选后 + 随机顺序"
echo "============================================================"
bash "${SCRIPT_DIR}/run_b3_filtered.sh"

# ===== Step 3: B4 (CAFCL) =====
echo ""
echo "[Step 3/3] B4: CAFCL (筛选 + 课程学习)"
echo "============================================================"
bash "${SCRIPT_DIR}/run_b4_cafcl.sh"

echo ""
echo "============================================================"
echo "  全部实验完成！"
echo ""
echo "  Checkpoints:"
echo "    B2: ${PROJECT_DIR}/checkpoints/b2_random_66k/"
echo "    B3: ${PROJECT_DIR}/checkpoints/b3_filtered/"
echo "    B4: ${PROJECT_DIR}/checkpoints/b4_cafcl/stage3/"
echo "============================================================"

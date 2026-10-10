#!/bin/bash
# ============================================================
# 一键环境搭建：克隆外部仓库 + 安装 Python 依赖
#
# 用法:
#   bash setup.sh
#
# 国内网络:
#   HF_ENDPOINT=https://hf-mirror.com bash setup.sh
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "============================================================"

# ============================================================
# 1. 克隆外部仓库（.gitignore 已排除，不会入库）
# ============================================================
echo ""
echo "[1/4] 克隆外部仓库..."

# verl - SFT 训练框架
if [ -d "${PROJECT_ROOT}/verl" ]; then
    echo "  verl/ 已存在，跳过克隆"
else
    echo "  克隆 verl..."
    git clone https://github.com/volcengine/verl.git "${PROJECT_ROOT}/verl"
fi

# 安装 verl（editable 模式，不重复装已有依赖）
echo "  安装 verl..."
pip install -e "${PROJECT_ROOT}/verl"

# ============================================================
# 2. 数据准备依赖
# ============================================================
echo ""
echo "[2/4] 数据准备依赖..."
pip install datasets pandas pyarrow

# ============================================================
# 3. 训练 & 模型依赖
# ============================================================
echo ""
echo "[3/4] 训练 & 模型依赖..."
pip install transformers peft accelerate

# ============================================================
# 4. HumanEval 评测
# ============================================================
echo ""
echo "[4/4] HumanEval 评测依赖..."
pip install human-eval

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "  环境搭建完成！"
echo ""
echo "  接下来:"
echo "    cd experiments"
echo ""
echo "    # 1. 准备数据 (国内加 HF_ENDPOINT=https://hf-mirror.com)"
echo "    python prepare_data.py --save_dir ./data --sample_size 10000"
echo ""
echo "    # 2. 跑训练"
echo "    NPROC=1 bash scripts/run_all.sh --skip-data"
echo ""
echo "    # 3. 跑评测"
echo "    python eval_humaneval.py --model Qwen/Qwen3-0.6B"
echo "    python eval_humaneval.py --model Qwen/Qwen3-0.6B --lora_path ./checkpoints/b4_cafcl/stage3"
echo "============================================================"

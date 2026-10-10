#!/bin/bash
# ============================================================
# B2: Evol-66k 全量 + 随机顺序训练 (标准 SFT 基线)
# ============================================================
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERL_DIR="$(dirname "$PROJECT_DIR")/verl"
DATA_DIR="${PROJECT_DIR}/data"

# 默认参数（可通过环境变量覆盖）
NPROC=${NPROC:-1}
MODEL_PATH="/root/cqx-code-generation/experiments/Qwen3-0.6B"
SAVE_DIR=${SAVE_DIR:-"${PROJECT_DIR}/checkpoints/b2_random_66k"}

echo "============================================================"
echo "实验 B2: 全量 66k + 随机顺序"
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${DATA_DIR}/evol_66k_all.parquet"
echo "  输出: ${SAVE_DIR}"
echo "  GPU数: ${NPROC}"
echo "============================================================"

cd "${VERL_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/evol_66k_all.parquet" \
    data.val_files=null \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=4 \
    data.enable_thinking_default=False \
    data.ignore_input_ids_mismatch=True \
    data.train_batch_size=64 \
    data.max_length=2048 \
    data.truncation=left \
    data.pad_mode=no_padding \
    model.path="${MODEL_PATH}" \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.target_modules="[q_proj,v_proj]" \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.lr_scheduler_type=cosine \
    engine=fsdp \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name=cafcl-code-gen \
    trainer.experiment_name=b2-random-66k \
    trainer.total_epochs=3 \
    trainer.logger=console \
    trainer.save_freq=-1 \
    trainer.seed=42 \
    "$@"

echo "B2 训练完成: ${SAVE_DIR}"

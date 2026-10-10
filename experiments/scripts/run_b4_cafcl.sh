#!/bin/bash
# ============================================================
# B4 (CAFCL): 筛选后数据 + 课程学习 (3 阶段渐进式训练)
#
# 课程学习策略:
#   Stage 1: Easy 子集，训练 3 epochs (从 base 模型开始)        → ~81 steps
#   Stage 2: Easy + Medium，训练 3 epochs (LoRA adapter 从 Stage1) → ~165 steps
#   Stage 3: Easy + Medium + Hard，训练 3 epochs (LoRA adapter 从 Stage2) → ~243 steps
#   总步数 ~489，对齐 B2 (random 66k, 468 steps) 做公平对比
#
# 注意: 不用 resume_from_path，因为那会加载 stage1 的 optimizer/LR 状态，
#       而 stage1 的 cosine LR 结束时已降到 0，resume 会导致 stage2/3 不学习。
#       改为每个 stage 结束后用 model_merger 导出 HF LoRA adapter，
#       下一 stage 通过 model.lora_adapter_path 加载权重、optimizer/LR 全新开始。
# ============================================================
set -xe

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERL_DIR="$(dirname "$PROJECT_DIR")/verl"
DATA_DIR="${PROJECT_DIR}/data"

NPROC=${NPROC:-1}
MODEL_PATH="/root/cqx-code-generation/experiments/Qwen3-0.6B"
SAVE_DIR=${SAVE_DIR:-"${PROJECT_DIR}/checkpoints/b4_cafcl"}

# 公共训练参数
COMMON_ARGS=(
    data.val_files=null
    data.messages_key=messages
    data.micro_batch_size_per_gpu=4
    data.train_batch_size=64
    data.enable_thinking_default=False
    data.ignore_input_ids_mismatch=True
    data.max_length=2048
    data.truncation=left
    data.pad_mode=no_padding
    model.lora_rank=16
    model.lora_alpha=32
    "model.target_modules=[q_proj,v_proj]"
    optim.lr=2e-5
    optim.lr_warmup_steps_ratio=0.1
    optim.lr_scheduler_type=cosine
    engine=fsdp
    trainer.project_name=cafcl-code-gen
    trainer.logger=wandb
    trainer.seed=42
)

cd "${VERL_DIR}"

# 工具函数: 读取某个 stage 目录下最新的 global_step_N 路径
latest_ckpt() {
    local stage_dir="$1"
    local tracker="${stage_dir}/latest_checkpointed_iteration.txt"
    if [[ ! -f "${tracker}" ]]; then
        echo "ERROR: checkpoint tracker not found: ${tracker}" >&2
        exit 1
    fi
    local step
    step=$(cat "${tracker}")
    echo "${stage_dir}/global_step_${step}"
}

# 工具函数: 把 stage 的 FSDP LoRA shard 导出成 HF adapter
# model_merger 会在 target_dir 下同时产出合并后的全模型(直接放在 target_dir)
# 和单独的 LoRA adapter(放在 target_dir/lora_adapter/)。
# 我们要的是 LoRA adapter 子目录用于 model.lora_adapter_path 加载。
export_lora() {
    local stage_dir="$1"
    local ckpt_dir
    ckpt_dir="$(latest_ckpt "${stage_dir}")"
    local merge_dir="${stage_dir}/hf_merged"
    local adapter_dir="${merge_dir}/lora_adapter"

    if [[ -f "${adapter_dir}/adapter_model.safetensors" ]]; then
        echo "HF adapter already exists: ${adapter_dir}, skip export" >&2
    else
        echo "Exporting HF LoRA adapter: ${ckpt_dir} -> ${merge_dir}" >&2
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${ckpt_dir}" \
            --target_dir "${merge_dir}" >&2
        # merger 会在 merge_dir 下同时写入合并后的全量模型 (~1.5GB) 和 tokenizer，
        # 我们只要 lora_adapter 子目录，删其余回血。
        find "${merge_dir}" -maxdepth 1 -type f -delete
        # 原始 FSDP shard (model_*.pt ~2.9GB, optim_*.pt) 已经通过 adapter 持久化，
        # 后续 stage 不需要，立刻删掉释放磁盘（盘只有 30GB，堆 3 stage 必然爆）。
        find "${ckpt_dir}" -maxdepth 1 -type f \( -name "model_world_size_*.pt" -o -name "optim_world_size_*.pt" \) -delete
        echo "Cleaned merger blobs in ${merge_dir} and FSDP shards in ${ckpt_dir}" >&2
    fi
    echo "${adapter_dir}"
}

# ===== Stage 1: Easy =====
echo "============================================================"
echo "CAFCL Stage 1/3: Easy 子集"
echo "============================================================"

STAGE1_DIR="${SAVE_DIR}/stage1"

if [[ -f "${STAGE1_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Stage 1 已存在 checkpoint, 跳过训练: ${STAGE1_DIR}"
else
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        -m verl.trainer.sft_trainer \
        data.train_files="${DATA_DIR}/curriculum_stage1.parquet" \
        trainer.default_local_dir="${STAGE1_DIR}" \
        trainer.experiment_name=b4-cafcl-stage1 \
        trainer.total_epochs=3 \
        trainer.save_freq=-1 \
        trainer.resume_mode=disable \
        model.path="${MODEL_PATH}" \
        "${COMMON_ARGS[@]}" \
        "$@"
    echo "Stage 1 完成: ${STAGE1_DIR}"
fi

# 导出 Stage 1 的 HF LoRA adapter，供 Stage 2 加载
STAGE1_ADAPTER="$(export_lora "${STAGE1_DIR}")"

# ===== Stage 2: Easy + Medium =====
echo "============================================================"
echo "CAFCL Stage 2/3: Easy + Medium"
echo "============================================================"

STAGE2_DIR="${SAVE_DIR}/stage2"

if [[ -f "${STAGE2_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Stage 2 已存在 checkpoint, 跳过训练: ${STAGE2_DIR}"
else
    # 用 Stage 1 的 LoRA adapter 初始化权重，optimizer/LR 全新开始
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        -m verl.trainer.sft_trainer \
        data.train_files="${DATA_DIR}/curriculum_stage2.parquet" \
        trainer.default_local_dir="${STAGE2_DIR}" \
        trainer.experiment_name=b4-cafcl-stage2 \
        trainer.total_epochs=3 \
        trainer.save_freq=-1 \
        trainer.resume_mode=disable \
        model.path="${MODEL_PATH}" \
        model.lora_adapter_path="${STAGE1_ADAPTER}" \
        "${COMMON_ARGS[@]}" \
        "$@"
    echo "Stage 2 完成: ${STAGE2_DIR}"
fi

# 导出 Stage 2 的 HF LoRA adapter，供 Stage 3 加载
STAGE2_ADAPTER="$(export_lora "${STAGE2_DIR}")"

# ===== Stage 3: All (Easy + Medium + Hard) =====
echo "============================================================"
echo "CAFCL Stage 3/3: 全部筛选后数据"
echo "============================================================"

STAGE3_DIR="${SAVE_DIR}/stage3"

if [[ -f "${STAGE3_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Stage 3 已存在 checkpoint, 跳过训练: ${STAGE3_DIR}"
else
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        -m verl.trainer.sft_trainer \
        data.train_files="${DATA_DIR}/curriculum_stage3.parquet" \
        trainer.default_local_dir="${STAGE3_DIR}" \
        trainer.experiment_name=b4-cafcl-stage3 \
        trainer.total_epochs=3 \
        trainer.save_freq=-1 \
        trainer.resume_mode=disable \
        model.path="${MODEL_PATH}" \
        model.lora_adapter_path="${STAGE2_ADAPTER}" \
        "${COMMON_ARGS[@]}" \
        "$@"
fi

# 最后导出 Stage 3 的 adapter，方便后续评测
STAGE3_ADAPTER="$(export_lora "${STAGE3_DIR}")"

echo "============================================================"
echo "CAFCL 全部 3 个阶段训练完成！"
echo "  Stage3 checkpoint : ${STAGE3_DIR}"
echo "  Stage3 HF adapter : ${STAGE3_ADAPTER}"
echo "============================================================"

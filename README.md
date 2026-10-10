# CAFCL: Complexity-Aware Curriculum Learning for Code Generation

EI 会议论文配套代码，从原始 `cqx-code-generation` 工程整理。

论文全文：[CAFCL 论文 PDF](paper/CAFCL.pdf)。

## 文件结构

- `experiments/prepare_data.py`：数据筛选、复杂度评分和三阶段课程数据生成。
- `experiments/scripts/`：B2 随机训练、B3 筛选训练、B4 CAFCL 课程训练。
- `experiments/eval_humaneval.py`：代码生成、后处理与 HumanEval 评测。
- `experiments/setup.sh`：克隆 verl 并安装依赖。
- `figure/`：训练日志 CSV、已有曲线图和绘图脚本。
- `paper_en.md` 和中文 Markdown：原始论文与说明材料。

## 使用方法

训练脚本面向 Linux/Bash 和 CUDA GPU 环境。先创建独立 Python 环境并安装与 CUDA 匹配的 PyTorch，然后执行：

```bash
bash experiments/setup.sh
cd experiments
python prepare_data.py --save_dir ./data --seed 42
NPROC=1 bash scripts/run_all.sh --skip-data
python eval_humaneval.py --model Qwen/Qwen3-0.6B --exp_name base
python eval_humaneval.py --model Qwen/Qwen3-0.6B --lora_path ./checkpoints/b4_cafcl/stage3/hf_merged/lora_adapter --exp_name b4
```

## 复现实验前需要配置

1. 原始训练脚本将 `MODEL_PATH` 固定为 `/root/cqx-code-generation/experiments/Qwen3-0.6B`。请在 `run_all.sh`、`run_b2_random.sh`、`run_b3_filtered.sh`、`run_b4_cafcl.sh` 中调整为实际模型目录或 Hugging Face 模型名称。
2. `run_all.sh` 的自动数据准备部分已被原作者注释，应先手动运行 `prepare_data.py`。
3. 训练使用 Weights & Biases 日志；运行前配置自己的账号。绘图脚本中的 `ENTITY_PROJECT` 也需修改，并安装 `matplotlib numpy wandb`。
4. `setup.sh` 会克隆当前 verl 版本；原始工程未记录依赖版本锁定信息，运行时需确认接口兼容性。
5. HumanEval 会执行生成的代码，请在隔离环境中评测。

本仓库保留原始算法及训练脚本。模型权重、训练数据、checkpoint 和临时文件不纳入版本管理。导入时仅检查脚本语法，未重新执行 GPU 训练或验证论文结果。
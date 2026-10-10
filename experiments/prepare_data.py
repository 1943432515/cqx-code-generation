"""
数据预处理脚本：下载 Evol-instruction-66k 并生成论文实验所需的全部 parquet 数据集。

生成的数据集:
  data/evol_66k_all.parquet          -- B2: 全部 66k，随机顺序
  data/evol_filtered.parquet         -- B3/B4: 筛选后 ~55k（去除不可执行 + 低质量样本）
  data/curriculum_stage1.parquet     -- B4 Stage1: Easy 子集
  data/curriculum_stage2.parquet     -- B4 Stage2: Easy + Medium
  data/curriculum_stage3.parquet     -- B4 Stage3: Easy + Medium + Hard (= filtered 全集)

用法:
  python prepare_data.py [--save_dir ./data] [--seed 42]
"""

import argparse
import ast
import os
import re
import sys
import textwrap

import datasets
import numpy as np
import pandas as pd


# ============================================================
# 1. 指令复杂度评分 (ICS)
# ============================================================

# 约束关键词（中英文）
CONSTRAINT_KEYWORDS = [
    "must", "should", "ensure", "require", "constraint", "mandatory",
    "not allowed", "forbidden", "at least", "at most", "exactly",
    "simultaneously", "both", "either", "neither",
    "必须", "不得", "同时满足", "确保", "约束", "至少", "至多",
]

# 逻辑嵌套关键词
DEPTH_KEYWORDS = [
    "if", "else", "elif", "nested", "within", "inside",
    "for each", "while", "recursive", "loop",
]

# 多步骤关键词
STEP_KEYWORDS = [
    "first", "then", "next", "after", "finally", "step",
    "followed by", "subsequently", "lastly", "第一步", "然后", "最后",
    "1.", "2.", "3.", "4.", "5.",
]


def compute_ics(instruction: str) -> dict:
    """计算指令复杂度评分的三个子维度。"""
    text = instruction.lower()

    # f_constraint: 约束数量
    f_constraint = sum(1 for kw in CONSTRAINT_KEYWORDS if kw.lower() in text)

    # f_depth: 逻辑嵌套深度
    f_depth = sum(1 for kw in DEPTH_KEYWORDS if kw.lower() in text)

    # f_step: 独立子任务数量
    f_step = sum(1 for kw in STEP_KEYWORDS if kw.lower() in text)

    # 加权得分 (w1=0.4, w2=0.35, w3=0.25)
    raw_score = 0.4 * f_constraint + 0.35 * f_depth + 0.25 * f_step

    return {
        "f_constraint": f_constraint,
        "f_depth": f_depth,
        "f_step": f_step,
        "ics_raw": raw_score,
    }


# ============================================================
# 2. 代码可执行性验证
# ============================================================

def check_executability(code: str, timeout: int = 5) -> bool:
    """
    检查代码是否可以通过语法解析（ast.parse）。
    不做实际执行（避免安全风险和耗时），但能过滤 SyntaxError。
    """
    # 尝试从 output 中提取代码块
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
    if code_blocks:
        code_to_check = "\n".join(code_blocks)
    else:
        # 如果没有代码块标记，尝试直接解析整个 output
        code_to_check = code

    try:
        ast.parse(code_to_check)
        return True
    except SyntaxError:
        return False
    except Exception:
        # 其他解析异常也视为可疑
        return False


# ============================================================
# 3. 指令-代码一致性评分 (简化版 ICAS)
# ============================================================

def compute_icas_simple(instruction: str, output: str) -> float:
    """
    简化版一致性评分。
    通过启发式方法评估 instruction 和 output 的匹配程度：
    - 指令中的关键实体（函数名、类名等）是否出现在 output 中
    - output 长度是否合理（太短可能是敷衍回答）
    - output 是否包含代码（代码生成任务应有代码）
    """
    score = 0.0

    # (a) 指令中的关键词是否在 output 中出现
    # 提取指令中可能的函数名/变量名
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', instruction)
    # 过滤掉常见停用词
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "can", "shall",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "that", "this", "it", "and", "or", "not", "no", "but", "if",
                 "then", "else", "when", "while", "as", "so", "than", "write",
                 "create", "implement", "function", "class", "method", "code",
                 "program", "python", "return", "input", "output", "given",
                 "using", "use", "make", "take", "get", "set", "def", "your"}
    key_terms = [w for w in identifiers if len(w) > 2 and w.lower() not in stopwords]
    if key_terms:
        matched = sum(1 for t in key_terms if t.lower() in output.lower())
        score += 0.4 * (matched / len(key_terms))
    else:
        score += 0.2  # 无明确关键词时给基线分

    # (b) output 是否包含代码
    has_code = bool(re.search(r'```|def |class |import |print\(|return ', output))
    score += 0.3 if has_code else 0.0

    # (c) output 长度合理性 (太短 < 50 字符可能是垃圾)
    if len(output) >= 200:
        score += 0.3
    elif len(output) >= 50:
        score += 0.15
    else:
        score += 0.0

    return min(score, 1.0)


# ============================================================
# 4. 综合筛选
# ============================================================

def compute_total_score(ics_norm: float, exec_ok: bool, icas: float,
                        lam1=0.3, lam2=0.3, lam3=0.4) -> float:
    """综合质量分数。"""
    return lam1 * ics_norm + lam2 * float(exec_ok) + lam3 * icas


# ============================================================
# 5. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="准备 CAFCL 实验数据")
    parser.add_argument("--save_dir", default="./data",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="筛选阈值 tau")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="从原始数据中随机抽样的数量（如 10000），不设则用全部")
    parser.add_argument("--dataset_path", default=None,
                        help="本地数据集路径（如已下载），否则从 HuggingFace 下载")
    args = parser.parse_args()

    np.random.seed(args.seed)
    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # -- 加载数据 --
    print("=" * 60)
    print("Step 1: 加载 Evol-instruction-66k 数据集...")
    if args.dataset_path:
        ds = datasets.load_dataset(args.dataset_path)
    else:
        ds = datasets.load_dataset("codefuse-ai/Evol-instruction-66k")

    # 数据集可能只有 train split
    if "train" in ds:
        df = ds["train"].to_pandas()
    else:
        df = pd.DataFrame(ds)

    print(f"  原始样本数: {len(df)}")
    print(f"  列名: {list(df.columns)}")

    # 可选：随机抽样
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
        print(f"  抽样后样本数: {len(df)}")

    # 自动检测列名 (可能是 instruction/output 或 Instruction/Output)
    inst_col = None
    out_col = None
    for c in df.columns:
        if c.lower() == "instruction":
            inst_col = c
        if c.lower() == "output":
            out_col = c
    if inst_col is None or out_col is None:
        print(f"ERROR: 无法检测到 instruction/output 列。当前列: {list(df.columns)}")
        sys.exit(1)
    print(f"  指令列: {inst_col}, 输出列: {out_col}")

    # -- Step 2: 计算质量评分 --
    print("=" * 60)
    print("Step 2: 计算三维质量评分...")

    ics_scores = []
    exec_flags = []
    icas_scores = []

    for i, row in df.iterrows():
        inst = str(row[inst_col])
        out = str(row[out_col])

        ics = compute_ics(inst)
        ics_scores.append(ics["ics_raw"])

        ex = check_executability(out)
        exec_flags.append(ex)

        ica = compute_icas_simple(inst, out)
        icas_scores.append(ica)

        if (i + 1) % 10000 == 0:
            print(f"  已处理 {i + 1}/{len(df)} ...")

    df["ics_raw"] = ics_scores
    df["exec_ok"] = exec_flags
    df["icas"] = icas_scores

    # 归一化 ICS 到 [0, 1]
    ics_min, ics_max = df["ics_raw"].min(), df["ics_raw"].max()
    if ics_max > ics_min:
        df["ics_norm"] = (df["ics_raw"] - ics_min) / (ics_max - ics_min)
    else:
        df["ics_norm"] = 0.5

    # 综合分数
    df["total_score"] = df.apply(
        lambda r: compute_total_score(r["ics_norm"], r["exec_ok"], r["icas"]),
        axis=1
    )

    print(f"  ICS 原始分布: min={df['ics_raw'].min():.2f}, "
          f"median={df['ics_raw'].median():.2f}, max={df['ics_raw'].max():.2f}")
    print(f"  可执行率: {df['exec_ok'].mean():.1%}")
    print(f"  ICAS 均值: {df['icas'].mean():.3f}")
    print(f"  综合分数分布: min={df['total_score'].min():.3f}, "
          f"median={df['total_score'].median():.3f}, max={df['total_score'].max():.3f}")

    # -- Step 3: 转为 messages 格式并保存全量数据 (B2) --
    print("=" * 60)
    print("Step 3: 生成 B2 全量数据集 (66k, 随机顺序)...")

    def to_messages(row):
        return [
            {"role": "user", "content": str(row[inst_col])},
            {"role": "assistant", "content": str(row[out_col])},
        ]

    df["messages"] = df.apply(to_messages, axis=1)

    # 随机打乱
    df_all = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_all[["messages"]].to_parquet(os.path.join(save_dir, "evol_66k_all.parquet"))
    print(f"  保存: {save_dir}/evol_66k_all.parquet ({len(df_all)} 条)")

    # -- Step 4: 筛选 (B3) --
    print("=" * 60)
    print(f"Step 4: 质量筛选 (阈值 tau={args.threshold})...")

    df_filtered = df[df["total_score"] > args.threshold].copy()
    df_filtered = df_filtered.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    df_filtered[["messages"]].to_parquet(os.path.join(save_dir, "evol_filtered.parquet"))
    print(f"  筛选后样本数: {len(df_filtered)} (过滤率: {1 - len(df_filtered)/len(df):.1%})")
    print(f"  保存: {save_dir}/evol_filtered.parquet")

    # -- Step 5: 课程学习分桶 (B4) --
    print("=" * 60)
    print("Step 5: 课程学习难度分桶...")

    q33 = df_filtered["ics_norm"].quantile(0.33)
    q66 = df_filtered["ics_norm"].quantile(0.66)
    print(f"  三分位点: q33={q33:.3f}, q66={q66:.3f}")

    easy = df_filtered[df_filtered["ics_norm"] <= q33]
    medium = df_filtered[(df_filtered["ics_norm"] > q33) & (df_filtered["ics_norm"] <= q66)]
    hard = df_filtered[df_filtered["ics_norm"] > q66]

    print(f"  Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}")

    # Stage 1: Easy only
    stage1 = easy.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    stage1[["messages"]].to_parquet(os.path.join(save_dir, "curriculum_stage1.parquet"))

    # Stage 2: Easy + Medium
    stage2 = pd.concat([easy, medium]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    stage2[["messages"]].to_parquet(os.path.join(save_dir, "curriculum_stage2.parquet"))

    # Stage 3: All filtered data
    stage3 = df_filtered.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    stage3[["messages"]].to_parquet(os.path.join(save_dir, "curriculum_stage3.parquet"))

    print(f"  保存: curriculum_stage1.parquet ({len(stage1)} 条)")
    print(f"  保存: curriculum_stage2.parquet ({len(stage2)} 条)")
    print(f"  保存: curriculum_stage3.parquet ({len(stage3)} 条)")

    # -- 统计摘要 --
    print("=" * 60)
    print("数据准备完成！摘要:")
    print(f"  B2 全量:      {save_dir}/evol_66k_all.parquet     ({len(df_all)} 条)")
    print(f"  B3 筛选后:    {save_dir}/evol_filtered.parquet     ({len(df_filtered)} 条)")
    print(f"  B4 Stage1:    {save_dir}/curriculum_stage1.parquet ({len(stage1)} 条)")
    print(f"  B4 Stage2:    {save_dir}/curriculum_stage2.parquet ({len(stage2)} 条)")
    print(f"  B4 Stage3:    {save_dir}/curriculum_stage3.parquet ({len(stage3)} 条)")

    # 保存筛选统计信息（供论文写作参考）
    stats = {
        "total_samples": len(df),
        "filtered_samples": len(df_filtered),
        "filter_rate": f"{1 - len(df_filtered)/len(df):.1%}",
        "exec_pass_rate": f"{df['exec_ok'].mean():.1%}",
        "icas_mean": f"{df['icas'].mean():.3f}",
        "threshold": args.threshold,
        "q33": f"{q33:.3f}",
        "q66": f"{q66:.3f}",
        "easy_count": len(easy),
        "medium_count": len(medium),
        "hard_count": len(hard),
    }
    stats_path = os.path.join(save_dir, "data_stats.txt")
    with open(stats_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  统计信息已保存: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

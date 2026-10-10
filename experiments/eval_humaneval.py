"""
HumanEval 评测脚本（裸版，不依赖 evalplus）

两阶段:
  Phase A (generation): 加载模型 → 生成 raw_samples_{name}.jsonl
  Phase B (post + judge): 读 raw → _extract_code → samples_{name}.jsonl → pass@k

默认 (`--stage all`) 跑完整流程。`--stage rejudge` 只重跑 Phase B（秒级），
这样改后处理逻辑不需要重新跑 GPU。

依赖:
  pip install human-eval transformers torch peft

用法:
  # 完整评测 base
  python eval_humaneval.py --model Qwen/Qwen3-0.6B --exp_name base

  # 评测 LoRA 微调后的模型
  python eval_humaneval.py --model Qwen/Qwen3-0.6B \
      --lora_path ./checkpoints/b2_random_66k/hf_merged/lora_adapter --exp_name b2

  # 只重跑后处理 + 评测（已有 raw_samples_<exp>.jsonl）
  python eval_humaneval.py --stage rejudge --exp_name b2
"""

import argparse
import json
import os
import re
import sys
import torch

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


def build_prompt(prompt: str) -> str:
    """把 HumanEval 的 function signature + docstring 包装成 chat 格式。"""
    return (
        f"Complete the following Python function. "
        f"Only output the function body, no explanation.\n\n{prompt}"
    )


def _extract_code(raw: str) -> str:
    """Extract the Python code segment from a chat-style model completion.

    Qwen-SFT'd models typically answer HumanEval prompts by (a) wrapping code in
    a ```python ... ``` fence and often (b) restating the full `def foo(...):`
    signature instead of only the body. Concatenating such output directly after
    HumanEval's prompt (which ends inside a function definition after the
    docstring) would be invalid Python. But if we extract just the fenced block,
    the redefined `def foo` becomes a second top-level definition that shadows
    the first — Python happily executes it and unit tests see the new body.
    Strategy: grab the fenced block if present, otherwise fall back to the first
    code-like line.
    """
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        is_code_keyword = stripped.startswith(("def ", "class ", "import ", "from ", "@"))
        is_indented_code = line[:1] in (" ", "\t") and bool(stripped)
        if is_code_keyword or is_indented_code:
            out = []
            for ln in lines[i:]:
                if ln.strip().startswith("```"):
                    break
                out.append(ln)
            return "\n".join(out).rstrip()
    return raw.rstrip()


def generate_one_raw(model, tokenizer, prompt: str, args) -> str:
    """生成单个原始 completion（未经后处理）。"""
    messages = [{"role": "user", "content": build_prompt(prompt)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def resolve_exp_name(args):
    if args.exp_name:
        return args.exp_name
    if args.lora_path:
        return args.lora_path.rstrip("/").split("/")[-1]
    return "base"


def phase_generate(args, raw_file):
    """Run generation; write raw_samples_<exp>.jsonl."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True,
    )
    if args.lora_path:
        print(f"加载 LoRA adapter: {args.lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("  LoRA 已合并")
    model.eval()

    problems = read_problems()
    print(f"HumanEval 题目数: {len(problems)}")
    print(f"每题生成 {args.n_samples} 个样本, temperature={args.temperature}")

    raw_samples = []
    total = len(problems) * args.n_samples
    count = 0
    for task_id, problem in problems.items():
        for _ in range(args.n_samples):
            raw = generate_one_raw(model, tokenizer, problem["prompt"], args)
            raw_samples.append({"task_id": task_id, "raw_completion": raw})
            count += 1
            if count % 20 == 0 or count == total:
                print(f"  进度: {count}/{total}")

    with open(raw_file, "w") as f:
        for s in raw_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"原始生成结果已保存: {raw_file}")


def phase_postprocess_and_judge(args, raw_file, samples_file, results_file, k_values):
    """Read raw_samples → _extract_code → samples → judge."""
    print(f"读取原始样本: {raw_file}")
    raw_samples = [json.loads(l) for l in open(raw_file)]
    samples = [
        {"task_id": r["task_id"], "completion": _extract_code(r["raw_completion"])}
        for r in raw_samples
    ]
    empty = sum(1 for s in samples if not s["completion"].strip())
    print(f"post-process: {len(samples)} 样本，空 completion = {empty}")

    write_jsonl(samples_file, samples)
    print(f"后处理结果已保存: {samples_file}")

    print("\n开始评测...")
    results = evaluate_functional_correctness(samples_file, k=k_values, n_workers=4)

    print("\n" + "=" * 50)
    print(f"  模型: {args.model}")
    if args.lora_path:
        print(f"  LoRA: {args.lora_path}")
    print(f"  n_samples={args.n_samples}, temperature={args.temperature}")
    print("-" * 50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}  ({v*100:.1f}%)")
    print("=" * 50)

    with open(results_file, "w") as f:
        json.dump({
            "model": args.model,
            "lora_path": args.lora_path,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            **results,
        }, f, indent=2)
    print(f"评测结果已保存: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="HumanEval 评测")
    parser.add_argument("--stage", choices=["all", "generate", "rejudge"], default="all",
                        help="all=生成+评测; generate=只生成; rejudge=只后处理+评测（用已有 raw）")
    parser.add_argument("--model", type=str, default=None,
                        help="基础模型路径或 HuggingFace ID（rejudge 阶段非必需）")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter 路径")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--pass_k", type=str, default="1")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="输出文件前缀；缺省从 lora_path 末段推断")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    k_values = [int(x) for x in args.pass_k.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    exp_name = resolve_exp_name(args)
    raw_file = os.path.join(args.output_dir, f"raw_samples_{exp_name}.jsonl")
    samples_file = os.path.join(args.output_dir, f"samples_{exp_name}.jsonl")
    results_file = os.path.join(args.output_dir, f"results_{exp_name}.json")

    if args.stage in ("all", "generate"):
        if args.model is None:
            print("ERROR: --model is required for generate/all", file=sys.stderr)
            sys.exit(1)
        phase_generate(args, raw_file)

    if args.stage in ("all", "rejudge"):
        if not os.path.exists(raw_file):
            print(f"ERROR: raw samples not found: {raw_file}", file=sys.stderr)
            sys.exit(1)
        phase_postprocess_and_judge(args, raw_file, samples_file, results_file, k_values)


if __name__ == "__main__":
    main()

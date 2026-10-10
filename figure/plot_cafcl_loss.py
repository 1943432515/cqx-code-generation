"""Pull CAFCL stage-level training curves from wandb and render a paper figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

OUT_DIR = Path(__file__).parent
ENTITY_PROJECT = "username/cafcl-code-gen" # 需要替换

STAGE_META = [
    ("b4-cafcl-stage1", "Stage 1: Easy",   "#4C78A8"),
    ("b4-cafcl-stage2", "Stage 2: +Medium", "#F58518"),
    ("b4-cafcl-stage3", "Stage 3: +Hard",   "#54A24B"),
]


def fetch_history():
    api = wandb.Api()
    # 同一个 stage 名可能有多次 run（1-epoch 旧版、3-epoch 新版、失败重跑等），
    # 挑 state=finished 里最新创建的那个。
    all_runs = list(api.runs(ENTITY_PROJECT))
    runs = {}
    for r in all_runs:
        if r.state != "finished":
            continue
        prev = runs.get(r.name)
        if prev is None or r.created_at > prev.created_at:
            runs[r.name] = r
    series = []
    offset = 0
    for name, label, color in STAGE_META:
        run = runs[name]
        print(f"{name}: run_id={run.id}, created={run.created_at}, steps={run.summary.get('_step')}")
        hist = run.history(keys=["train/loss", "train/lr", "train/grad_norm"], samples=1000)
        hist = hist.dropna(subset=["train/loss"]).reset_index(drop=True)
        local_step = np.arange(1, len(hist) + 1)
        global_step = local_step + offset
        series.append({
            "name": name,
            "label": label,
            "color": color,
            "local_step": local_step,
            "global_step": global_step,
            "loss": hist["train/loss"].to_numpy(),
            "lr": hist["train/lr"].to_numpy() if "train/lr" in hist else None,
            "grad_norm": hist["train/grad_norm"].to_numpy() if "train/grad_norm" in hist else None,
        })
        offset += len(hist)
    return series


def smooth(x, w=5):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(x)]


def plot_loss(series, out_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    boundaries = []
    for s in series:
        ax.plot(s["global_step"], s["loss"], color=s["color"], alpha=0.35, linewidth=1.0)
        ax.plot(s["global_step"], smooth(s["loss"], w=5),
                color=s["color"], linewidth=2.0, label=s["label"])
        boundaries.append(s["global_step"][-1])

    for b in boundaries[:-1]:
        ax.axvline(b + 0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)

    for s in series:
        ax.scatter([s["global_step"][-1]], [s["loss"][-1]],
                   color=s["color"], s=35, zorder=5, edgecolor="white", linewidth=1.0)
        ax.annotate(f"{s['loss'][-1]:.3f}",
                    xy=(s["global_step"][-1], s["loss"][-1]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color=s["color"])

    ax.set_xlabel("Training Step (cumulative across stages)")
    ax.set_ylabel("Training Loss")
    ax.set_title("CAFCL Curriculum Training Loss across Three Stages")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=400, bbox_inches="tight")
        print(f"saved {p}")
    plt.close(fig)


def dump_csv(series, out_path):
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "local_step", "global_step", "train/loss", "train/lr", "train/grad_norm"])
        for s in series:
            for i in range(len(s["loss"])):
                w.writerow([
                    s["name"],
                    s["local_step"][i],
                    s["global_step"][i],
                    s["loss"][i],
                    s["lr"][i] if s["lr"] is not None else "",
                    s["grad_norm"][i] if s["grad_norm"] is not None else "",
                ])
    print(f"saved {out_path}")


if __name__ == "__main__":
    series = fetch_history()
    dump_csv(series, OUT_DIR / "cafcl_history.csv")
    plot_loss(series, OUT_DIR / "cafcl_loss_curve")

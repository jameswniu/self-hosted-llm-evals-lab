#!/usr/bin/env python3
"""Generate figures referenced by README.md into docs/figures/."""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, "docs", "figures")
ABLATION_PATH = os.path.join(REPO_ROOT, "ablation", "results", "ablation_results.json")
METRICS_PATH = os.path.join(REPO_ROOT, "perf", "metrics.csv")


def wilson_ci(p, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return centre - spread, centre + spread


# -- Figure 1: accuracy_by_strategy.png ----------------------------------------

def fig_accuracy_by_strategy(data):
    strategies = [
        ("template_baseline", "Baseline"),
        ("template_instruction", "Instruction"),
        ("template_cot", "Chain-of-Thought"),
        ("template_fewshot_cot", "Few-shot + CoT"),
        ("self_consistency_k5", "Self-consistency k=5"),
    ]
    names = [s[1] for s in strategies]
    accs = [data[s[0]]["accuracy"] for s in strategies]
    n = 20

    lo = [wilson_ci(a, n)[0] for a in accs]
    hi = [wilson_ci(a, n)[1] for a in accs]
    err_lo = [a - l for a, l in zip(accs, lo)]
    err_hi = [h - a for a, h in zip(accs, hi)]

    best_idx = int(np.argmax(accs))
    worst_idx = int(np.argmin(accs))
    colors = []
    for i in range(len(accs)):
        if i == best_idx:
            colors.append("#2ecc71")
        elif i == worst_idx:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(names, accs, yerr=[err_lo, err_hi], fmt="none", ecolor="black",
                capsize=4, linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{acc:.0%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Prompting Strategy (n=20, Wilson 95% CI)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(axis="x", rotation=15)
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_by_strategy.png"), dpi=150)
    plt.close(fig)
    print("  accuracy_by_strategy.png")


# -- Figure 2: confidence_routing.png ------------------------------------------

def fig_confidence_routing(data):
    examples = data["self_consistency_k5"]["examples"]
    conf = [e["confidence"] for e in examples]
    correct = [e["correct"] for e in examples]

    # Bucket by confidence
    buckets = {"<= 0.6": [], "0.8": [], "1.0": []}
    for c, cor in zip(conf, correct):
        if c <= 0.6:
            buckets["<= 0.6"].append(cor)
        elif c < 1.0:
            buckets["0.8"].append(cor)
        else:
            buckets["1.0"].append(cor)

    bucket_names = list(buckets.keys())
    bucket_accs = [np.mean(v) if v else 0 for v in buckets.values()]
    bucket_ns = [len(v) for v in buckets.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # Left: strip plot of confidence vs correctness
    jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(conf))
    colors_strip = ["#2ecc71" if c else "#e74c3c" for c in correct]
    x_pos = [1 if c else 0 for c in correct]
    ax1.scatter([x + j for x, j in zip(x_pos, jitter)], conf, c=colors_strip,
                s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Incorrect", "Correct"])
    ax1.set_ylabel("Vote Confidence")
    ax1.set_title("Confidence vs Correctness")
    ax1.set_ylim(0.3, 1.1)
    ax1.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(1.05, 0.8, "threshold", fontsize=8, color="gray", va="center")

    # Right: accuracy by confidence bucket
    bar_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax2.bar(bucket_names, bucket_accs, color=bar_colors, edgecolor="white", linewidth=0.8)
    for bar, acc, n in zip(bars, bucket_accs, bucket_ns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"{acc:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Vote Confidence")
    ax2.set_title("Accuracy by Confidence Bucket")
    ax2.set_ylim(0, 1.2)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confidence_routing.png"), dpi=150)
    plt.close(fig)
    print("  confidence_routing.png")


# -- Figure 3: latency_throughput.png ------------------------------------------

def fig_latency_throughput(metrics_path):
    df = pd.read_csv(metrics_path)
    df = df[df["success"] == True].copy()
    # Exclude cold-start first row
    df = df.iloc[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: TTFT boxplot by batch_label
    order = sorted(df["batch_label"].unique())
    sns.boxplot(data=df, x="batch_label", y="ttft_ms", hue="batch_label", order=order, ax=ax1,
                palette="Blues_d", fliersize=3, legend=False)
    ax1.set_xlabel("Batch Configuration")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token by Configuration")
    ax1.tick_params(axis="x", rotation=30)

    # Right: throughput barplot by batch_label
    throughput = df.groupby("batch_label")["tokens_per_sec"].mean().reindex(order)
    ax2.bar(order, throughput.values, color=sns.color_palette("Blues_d", len(order)),
            edgecolor="white", linewidth=0.8)
    for i, (lbl, val) in enumerate(zip(order, throughput.values)):
        ax2.text(i, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Batch Configuration")
    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_title("Mean Throughput by Configuration")
    ax2.tick_params(axis="x", rotation=30)

    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "latency_throughput.png"), dpi=150)
    plt.close(fig)
    print("  latency_throughput.png")


# -- Main ---------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Generating figures...")

    with open(ABLATION_PATH) as f:
        ablation = json.load(f)

    fig_accuracy_by_strategy(ablation)
    fig_confidence_routing(ablation)
    fig_latency_throughput(METRICS_PATH)

    print(f"Done. Figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

"""Regenerate README charts with the darker Mermaid-aligned palette."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
})

# Palette
NAVY = "#2D5FAF"
TERRA = "#C44B3B"
TEAL = "#278F84"
PURPLE = "#7B52C7"
AMBER = "#C4890C"

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def chart_accuracy():
    strategies = [
        "Baseline\n(minimal)",
        "Instruction\ntemplate",
        "Chain-of-\nthought",
        "Few-shot\n+ CoT",
        "Self-consistency\n(k=5)",
    ]
    values = [60, 50, 35, 55, 70]
    colors = [NAVY, AMBER, TERRA, PURPLE, TEAL]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(strategies, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Baseline dashed line
    ax.axhline(60, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Bold percentage labels above bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val}%",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    # Annotation arrow from baseline to CoT
    ax.annotate(
        "-25pp",
        xy=(2, 35), xytext=(2, 55),
        fontsize=11, fontweight="bold", color=TERRA,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=TERRA, lw=2),
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_title("HellaSwag Accuracy by Prompting Strategy (Llama 3.1 8B)", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_by_strategy.png"), dpi=180)
    plt.close(fig)
    print("  -> accuracy_by_strategy.png")


def chart_confidence():
    buckets = ["1.0\n(unanimous)", "0.8", "0.6", "0.4"]
    correct_pct = [100, 86, 50, 33]
    ns = [5, 7, 4, 3]
    colors = [TEAL, TEAL, AMBER, TERRA]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(buckets, correct_pct, color=colors, width=0.55, edgecolor="white", linewidth=0.5)

    # Random baseline
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Labels
    for bar, val, n in zip(bars, correct_pct, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val}%  (n={n})",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_xlabel("Vote Confidence")
    ax.set_ylabel("Correct (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Self-Consistency: Vote Confidence as Routing Signal", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confidence_routing.png"), dpi=180)
    plt.close(fig)
    print("  -> confidence_routing.png")


def chart_latency_throughput():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: TTFT vs Concurrency (grouped bar) ---
    concurrency = ["c=1", "c=3", "c=5"]
    short_ttft = [130, 3200, 5800]
    long_ttft = [300, 4100, None]  # c=5 missing for long

    x = np.arange(len(concurrency))
    w = 0.35

    bars1 = ax1.bar(x - w / 2, short_ttft, w, label="Short prompts", color=NAVY, edgecolor="white", linewidth=0.5)
    long_vals = [v if v is not None else 0 for v in long_ttft]
    bars2 = ax1.bar(x + w / 2, long_vals, w, label="Long prompts", color=AMBER, edgecolor="white", linewidth=0.5)
    # Hide the missing bar
    bars2[2].set_visible(False)

    for bar, val in zip(bars1, short_ttft):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 80, str(val),
                 ha="center", va="bottom", fontweight="bold", fontsize=9)
    for bar, val in zip(bars2, long_ttft):
        if val is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 80, str(val),
                     ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(concurrency)
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token vs Concurrency", fontweight="bold", fontsize=12)
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right: Throughput (single-color bars) ---
    labels = ["short\nc=1", "short\nc=3", "short\nc=5", "long\nc=1", "long\nc=3"]
    throughput = [65, 62, 58, 64, 60]

    bars3 = ax2.bar(labels, throughput, color=TEAL, width=0.55, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars3, throughput):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val),
                 ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax2.set_ylabel("tok/s")
    ax2.set_ylim(0, 80)
    ax2.set_title("Throughput (tok/s)", fontweight="bold", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "latency_throughput.png"), dpi=180)
    plt.close(fig)
    print("  -> latency_throughput.png")


if __name__ == "__main__":
    print("Generating figures...")
    chart_accuracy()
    chart_confidence()
    chart_latency_throughput()
    print("Done.")

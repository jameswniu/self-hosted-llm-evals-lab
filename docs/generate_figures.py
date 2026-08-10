#!/usr/bin/env python3
"""Generate every figure in docs/figures/ as SVG, straight from the results files.

Stdlib only: no matplotlib, no seaborn, no pandas. The figures are emitted as SVG
so they stay sharp at any zoom and readable at 75% browser zoom, and so a reader
can diff them in a pull request instead of eyeballing a re-rendered PNG.

Sources of truth:
  ablation/results/ablation_results.json   accuracy per strategy, per-item votes
  perf/metrics.csv                         one row per load-test request
"""
import csv, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import *  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
ABLATION = os.path.join(ROOT, "ablation", "results", "ablation_results.json")
METRICS = os.path.join(ROOT, "perf", "metrics.csv")
N = 20


def fig_accuracy(data):
    H = 545
    rows = [("Baseline", "template_baseline", BLUE), ("Instruction", "template_instruction", BLUE),
            ("Chain-of-Thought", "template_cot", ORANGE), ("Few-shot + CoT", "template_fewshot_cot", BLUE),
            ("Self-consistency", "self_consistency_k5", AQUA)]
    s = head(H, "f1", "Accuracy by prompting strategy on 20 items, with Wilson 95 percent confidence intervals")
    s += title_block("f1", "ABLATION RESULT", "Accuracy by prompting strategy")
    x0, x1, ytop, ybot = 96, 864, 140, 400
    for pct in (0, 25, 50, 75, 100):
        y = ybot - (ybot - ytop) * pct / 100
        s += f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>\n'
        s += txt(x0 - 14, y + 5, f"{pct}%", 15, MUTE, anchor="end")
    slot = (x1 - x0) / len(rows)
    for i, (label, key, col) in enumerate(rows):
        acc = data[key]["accuracy"]
        lo, hi = wilson(acc, N)
        cx = x0 + slot * (i + 0.5)
        bw = 96
        by = ybot - (ybot - ytop) * acc
        s += f'<rect x="{cx-bw/2:.1f}" y="{by:.1f}" width="{bw}" height="{ybot-by:.1f}" fill="{col}" rx="4"/>\n'
        ylo, yhi = ybot - (ybot - ytop) * lo, ybot - (ybot - ytop) * hi
        s += (f'<line x1="{cx}" y1="{yhi:.1f}" x2="{cx}" y2="{ylo:.1f}" stroke="{INK2}" stroke-width="2"/>\n'
              f'<line x1="{cx-13}" y1="{yhi:.1f}" x2="{cx+13}" y2="{yhi:.1f}" stroke="{INK2}" stroke-width="2"/>\n'
              f'<line x1="{cx-13}" y1="{ylo:.1f}" x2="{cx+13}" y2="{ylo:.1f}" stroke="{INK2}" stroke-width="2"/>\n')
        s += txt(cx, yhi - 14, f"{acc:.0%}", 19, INK, weight="700")
        s += txt(cx, ybot + 28, label, 15, INK3)
        s += txt(cx, ybot + 50, f"{round(acc*N)}/{N}", 15, FAINT)
    s += f'<line x1="{x0}" y1="{ybot}" x2="{x1}" y2="{ybot}" stroke="#46525f" stroke-width="1.5"/>\n'
    s += caption(["Whiskers are Wilson 95% intervals on n=20. They overlap across most of their range, which is the honest",
                  "read of a 20-item run: the ordering is suggestive, and no single pairwise gap here is established."], 496)
    return s + "</svg>\n"


def fig_confidence(data):
    H = 520
    ex = data["self_consistency_k5"]["examples"]
    levels = sorted({e["confidence"] for e in ex})
    s = head(H, "f2", "Vote confidence against correctness for all fifteen logged items, and accuracy grouped by confidence bucket")
    s += title_block("f2", "ROUTING SIGNAL", "Vote confidence against correctness")

    x0, ybot, ytop = 92, 372, 150
    s += txt(x0, 126, "one dot per logged item", 15, INK3, anchor="start")
    slot = (470 - x0) / len(levels)
    for i, lv in enumerate(levels):
        cx = x0 + slot * (i + 0.5)
        items = [e for e in ex if e["confidence"] == lv]
        for j, e in enumerate(items):
            col, r = (AQUA, 11) if e["correct"] else (ORANGE, 11)
            cy = ybot - 26 - j * 26
            s += f'<circle cx="{cx:.1f}" cy="{cy}" r="{r}" fill="{col}" stroke="{BG1}" stroke-width="2"/>\n'
        ncor = sum(1 for e in items if e["correct"])
        s += txt(cx, ybot + 26, f"{lv:.1f}", 16, INK3)
        s += txt(cx, ybot + 48, f"{ncor}/{len(items)}", 14, FAINT)
    s += f'<line x1="{x0}" y1="{ybot}" x2="470" y2="{ybot}" stroke="#46525f" stroke-width="1.5"/>\n'
    s += txt(281, ybot + 76, "vote confidence", 15, MUTE)
    s += f'<circle cx="344" cy="122" r="8" fill="{AQUA}"/>\n' + txt(360, 127, "correct", 14, MUTE, anchor="start")
    s += f'<circle cx="430" cy="122" r="8" fill="{ORANGE}"/>\n' + txt(446, 127, "wrong", 14, MUTE, anchor="start")

    bx0 = 540
    s += txt(bx0, 126, "accuracy by bucket", 15, INK3, anchor="start")
    buckets = [("&lt;= 0.6", [e for e in ex if e["confidence"] <= 0.6], ORANGE),
               ("&gt;= 0.8", [e for e in ex if e["confidence"] >= 0.8], AQUA)]
    bslot = (864 - bx0) / len(buckets)
    for i, (name, items, col) in enumerate(buckets):
        acc = sum(1 for e in items if e["correct"]) / len(items)
        cx = bx0 + bslot * (i + 0.5)
        by = ybot - (ybot - ytop) * acc
        s += f'<rect x="{cx-64:.1f}" y="{by:.1f}" width="128" height="{ybot-by:.1f}" fill="{col}" rx="4"/>\n'
        s += txt(cx, by - 14, f"{acc:.0%}", 19, INK, weight="700")
        s += txt(cx, ybot + 26, name, 16, INK3)
        s += txt(cx, ybot + 48, f"{sum(1 for e in items if e['correct'])}/{len(items)} items", 14, FAINT)
    s += f'<line x1="{bx0}" y1="{ybot}" x2="864" y2="{ybot}" stroke="#46525f" stroke-width="1.5"/>\n'
    s += caption(["Confident votes are right far more often than unsure ones, which is what makes the cascade worth building.",
                  "With 15 logged items the split is a usable signal, not a calibrated threshold: treat 0.8 as a starting knob."], 470)
    return s + "</svg>\n"


def fig_latency(rows):
    H = 520
    ok = [r for r in rows if r["success"] == "True"][1:]
    by = {}
    for r in ok:
        by.setdefault(r["batch_label"], []).append((float(r["ttft_ms"]), float(r["tokens_per_sec"])))
    labels = sorted(by, key=lambda k: sorted(x[0] for x in by[k])[len(by[k]) // 2])
    s = head(H, "f3", "Median time to first token and mean throughput for each load-test configuration")
    s += title_block("f3", "SERVING PROFILE", "Concurrency costs latency and buys nothing")
    s += txt(300, 128, "median TTFT (ms, log scale)", 15, INK3, anchor="middle")
    s += txt(700, 128, "mean throughput (tok/s)", 15, INK3, anchor="middle")
    ax0, ax1 = 196, 500
    bx0, bx1 = 596, 864
    maxtps = max(sum(x[1] for x in v) / len(v) for v in by.values())
    for i, lab in enumerate(labels):
        vals = sorted(x[0] for x in by[lab])
        p50 = vals[len(vals) // 2]
        tps = sum(x[1] for x in by[lab]) / len(by[lab])
        y = 156 + i * 40
        conc = lab.split("_c")[1][0]
        col = ORANGE if p50 > 1000 else BLUE
        s += txt(184, y + 15, lab, 15, INK3, anchor="end")
        lw = (ax1 - ax0) * (math.log10(p50) - 2) / (math.log10(9000) - 2)
        s += f'<rect x="{ax0}" y="{y}" width="{max(lw,3):.1f}" height="22" fill="{col}" rx="3"/>\n'
        s += txt(ax0 + max(lw, 3) + 10, y + 16, f"{p50:,.0f}", 15, INK2, anchor="start")
        tw = (bx1 - bx0) * tps / maxtps
        s += f'<rect x="{bx0}" y="{y}" width="{tw:.1f}" height="22" fill="{AQUA}" rx="3"/>\n'
        s += txt(bx0 + tw + 10, y + 16, f"{tps:.0f}", 15, INK2, anchor="start")
    yb = 156 + len(labels) * 40
    s += f'<line x1="{ax0}" y1="{yb}" x2="{ax1}" y2="{yb}" stroke="#46525f" stroke-width="1.5"/>\n'
    s += f'<line x1="{bx0}" y1="{yb}" x2="{bx1}" y2="{yb}" stroke="#46525f" stroke-width="1.5"/>\n'
    s += caption(["Time to first token climbs by an order of magnitude once concurrency rises, while throughput stays flat near 65 tok/s.",
                  "That is the signature of a backend that queues rather than batches: the work is serialized, so parallel callers only wait."], yb + 44)
    return s + "</svg>\n"


def main():
    os.makedirs(FIG, exist_ok=True)
    data = json.load(open(ABLATION))
    rows = list(csv.DictReader(open(METRICS)))
    for name, svg in [("accuracy_by_strategy", fig_accuracy(data)),
                      ("confidence_routing", fig_confidence(data)),
                      ("latency_throughput", fig_latency(rows))]:
        p = os.path.join(FIG, f"{name}.svg")
        open(p, "w").write(svg)
        print(f"  docs/figures/{name}.svg  {os.path.getsize(p):,} bytes")


if __name__ == "__main__":
    main()

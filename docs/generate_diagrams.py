#!/usr/bin/env python3
"""Emit the three concept diagrams in assets/ as SVG.

These describe structure rather than data, so they are generated once here
instead of being derived from a results file. Type is sized for 75% browser zoom.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import *  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")


def architecture():
    H = 738
    s = head(H, "a", "Call graph: four Makefile targets drive four pipelines that reach one Ollama endpoint over two API surfaces, with a standalone cached wrapper parked off the path")
    s += title_block("a", "CALL GRAPH", "Four pipelines, two API surfaces, one served model")

    lanes = [(32, "make eval", "run_eval.py", ["subprocess to", "python -m lm_eval"], BLUE),
             (246, "make ablation", "optimize_prompt.py", ["infer.py", "4 templates, k=5"], AQUA),
             (460, "make perf", "load_test.py", ["ThreadPoolExecutor", "streaming TTFT"], AQUA),
             (674, "make validate", "validate.py", ["5 prompts x 5 trials", "greedy, seed 42"], AQUA)]
    for x, tgt, mod, subs, col in lanes:
        cx = x + 99
        s += box(x, 118, 198, 36, "a", stroke="#46525f", fill="#141d27")
        s += txt(cx, 142, tgt, 17, INK2)
        s += f'<line x1="{cx}" y1="154" x2="{cx}" y2="184" stroke="#5a6673" stroke-width="1.6" marker-end="url(#ara)"/>\n'
        s += box(x, 186, 198, 96, "a")
        s += txt(cx, 214, mod, 17, INK, weight="700")
        for i, sub in enumerate(subs):
            s += txt(cx, 242 + i * 22, sub, 15, MUTE)
        mk = "arba" if col == BLUE else "arga"
        s += f'<line x1="{cx}" y1="282" x2="{cx}" y2="314" stroke="{col}" stroke-width="1.8" marker-end="url(#{mk})"/>\n'

    s += box(32, 316, 198, 62, "a", stroke=BLUE, sw=1.8, fill="#0e1620")
    s += txt(131, 342, "/v1/chat/completions", 15, BLUE_T, weight="700")
    s += txt(131, 364, "OpenAI-compatible", 15, MUTE)
    s += box(246, 316, 626, 62, "a", stroke=AQUA, sw=1.8, fill="#0d1a16")
    s += txt(559, 342, "/api/generate", 17, AQUA_T, weight="700")
    s += txt(559, 364, "Ollama native", 15, MUTE)

    for cx in (131, 559):
        s += f'<line x1="{cx}" y1="378" x2="{cx}" y2="410" stroke="#5a6673" stroke-width="1.6" marker-end="url(#ara)"/>\n'

    s += box(32, 412, 840, 74, "a", stroke="#8f9aa6", sw=1.8, fill="#141d27")
    s += txt(452, 444, "Ollama", 23, INK, mono=False, weight="700")
    s += txt(452, 470, "llama3.1:8b     Q4_0     ~4.7 GB     port 11434", 16, INK3)
    s += txt(36, 512, "process lifecycle owned by serve/serve.py, started with make serve, health-checked on /api/tags", 15, FAINT, anchor="start")

    s += f'<rect x="32" y="534" width="840" height="122" fill="#0a0e12" stroke="#4a5663" stroke-width="1.4" stroke-dasharray="6 5" rx="4"/>\n'
    s += txt(56, 566, "eval_runner/model.py", 17, INK2, anchor="start", weight="700")
    s += txt(56, 594, "OllamaEvalModel + PromptCache, SHA-256 keyed", 15, MUTE, anchor="start")
    s += txt(56, 616, "writes .cache/{hash}.json, speaks both surfaces above", 15, MUTE, anchor="start")
    s += txt(56, 638, "no module imports it: every pipeline calls requests.post", 15, MUTE, anchor="start")
    s += box(666, 566, 182, 58, "a", stroke="#8a6a3c", sw=1.5, fill="#1c150c")
    s += txt(757, 590, "NOT ON ANY", 15, "#c9985e", weight="700")
    s += txt(757, 612, "MAKEFILE PATH", 15, "#c9985e", weight="700")

    s += caption(["Three of the four pipelines speak Ollama's native API. Only lm-eval-harness uses the OpenAI-compatible",
                  "surface, which is why the eval lane runs as a subprocess instead of through the shared wrapper below."], 692)
    return s + "</svg>\n"


def ablation():
    H = 500
    s = head(H, "b", "Ablation pipeline: four prompt templates scored under greedy decoding, the winner carried into five-sample self-consistency voting, then Wilson intervals and McNemar tests")
    s += title_block("b", "EXPERIMENTAL DESIGN", "What the ablation actually runs")
    steps = [(30, 132, "4 templates", ["baseline, instruction", "CoT, few-shot + CoT"], STROKE, INK),
             (340, 132, "greedy decode", ["temp 0, top_k 1", "seed 42"], STROKE, INK),
             (650, 132, "best template", ["highest accuracy", "carried forward"], STROKE, INK),
             (30, 300, "self-consistency", ["k=5, temp 0.7", "seeds 42 to 46"], AQUA, AQUA_T),
             (340, 300, "majority vote", ["confidence =", "winning count / k"], AQUA, AQUA_T),
             (650, 300, "Wilson + McNemar", ["95% intervals", "paired test"], "#8f9aa6", INK)]
    for i, (x, y, t, subs, stroke, tc) in enumerate(steps):
        cx = x + 110
        s += box(x, y, 220, 110, "b", stroke=stroke, sw=1.6)
        s += txt(cx, y + 36, t, 17, tc, weight="700")
        for j, sub in enumerate(subs):
            s += txt(cx, y + 66 + j * 24, sub, 15, MUTE)
        if i in (0, 1, 3, 4):
            col = AQUA if i >= 3 else "#5a6673"
            mk = "argb" if i >= 3 else "arb"
            s += f'<line x1="{x+220}" y1="{y+55}" x2="{x+304}" y2="{y+55}" stroke="{col}" stroke-width="1.8" marker-end="url(#{mk})"/>\n'
    s += f'<path d="M 870 242 L 884 242 L 884 272 L 16 272 L 16 320 L 24 320" fill="none" stroke="#5a6673" stroke-width="1.8" marker-end="url(#arb)"/>\n'
    s += caption(["Every strategy is scored on the identical 20-item subset, so a difference between two bars is a difference in the",
                  "prompt, not in the sample. Self-consistency is the only stage that leaves greedy decoding, and the only one that helped."], 452)
    return s + "</svg>\n"


def routing():
    H = 540
    s = head(H, "c", "Confidence routing: five samples are extracted by a three-tier regex and voted, then high-confidence answers are served from the small model while low-confidence answers cascade to a larger one")
    s += title_block("c", "PRODUCTION CASCADE", "Spending the big model only where the small one is unsure")
    row = [(20, "generate k=5", ["temp 0.7, top_p 0.95", "seeds 42 to 46"], BLUE, BLUE_T),
           (320, "extract answers", ["3-tier regex normalizes", "A, A., (A), answer is A"], STROKE, INK),
           (620, "majority vote", ["confidence =", "winning count / k"], "#8f9aa6", INK)]
    for i, (x, ti, subs, stroke, tc) in enumerate(row):
        cx = x + 130
        s += box(x, 130, 260, 108, "c", stroke=stroke, sw=1.6)
        s += txt(cx, 166, ti, 17, tc, weight="700")
        for j, sub in enumerate(subs):
            s += txt(cx, 196 + j * 22, sub, 15, MUTE)
        if i < 2:
            s += f'<line x1="{x+260}" y1="184" x2="{x+292}" y2="184" stroke="#5a6673" stroke-width="1.8" marker-end="url(#arc)"/>\n'
    s += txt(20, 118, "one item in", 15, FAINT, anchor="start")

    s += f'<path d="M 750 238 L 750 286 L 232 286 L 232 344" fill="none" stroke="{AQUA}" stroke-width="1.9" marker-end="url(#argc)"/>\n'
    s += f'<path d="M 750 238 L 750 312 L 650 312 L 650 344" fill="none" stroke="{ORANGE}" stroke-width="1.9" marker-end="url(#aroc)"/>\n'
    s += box(42, 346, 380, 88, "c", stroke=AQUA, sw=1.8, fill="#0d1a16")
    s += txt(232, 382, "confidence &gt;= 0.8", 19, AQUA_T, weight="700")
    s += txt(232, 410, "serve straight from the 8B", 16, INK3)
    s += box(460, 346, 380, 88, "c", stroke=ORANGE, sw=1.8, fill="#1c130e")
    s += txt(650, 382, "confidence &lt;= 0.6", 19, ORANGE_T, weight="700")
    s += txt(650, 410, "cascade to a larger model", 16, INK3)
    s += caption(["The vote is not only an accuracy trick. How lopsided the five samples were is itself a signal, and it separates the",
                  "answers worth trusting from the minority worth paying for. That is the part of this repo that survives a budget."], 476)
    return s + "</svg>\n"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("architecture", architecture), ("ablation-pipeline", ablation), ("confidence-routing", routing)]:
        p = os.path.join(OUT, f"{name}.svg")
        open(p, "w").write(fn())
        print(f"  assets/{name}.svg  {os.path.getsize(p):,} bytes")

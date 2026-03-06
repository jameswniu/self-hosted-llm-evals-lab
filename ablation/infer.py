#!/usr/bin/env python3
"""
Run inference with the optimized configuration.

Applies the best prompt template and decoding settings found during
optimization, then evaluates against the full benchmark subset.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from optimize_prompt import (
    generate,
    extract_answer,
    self_consistency_vote,
    build_prompt,
)


IMPROVE_DIR = Path(__file__).parent
DATA_DIR = IMPROVE_DIR / "data"
RESULTS_DIR = IMPROVE_DIR / "results"

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"


def load_best_config() -> dict:
    """Load ablation results and determine best config."""
    results_path = RESULTS_DIR / "ablation_results.json"
    if not results_path.exists():
        print("[infer] No ablation results found. Run optimize_prompt.py first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    best = max(results.items(), key=lambda x: x[1]["accuracy"])
    return {"name": best[0], "accuracy": best[1]["accuracy"], "config": best[1]}


def run_baseline(
    examples: list,
    model: str,
    base_url: str,
    templates: dict,
    seed: int = 42,
) -> dict:
    """Run baseline evaluation with default template."""
    correct = 0
    total = 0
    predictions = []

    for item in examples:
        prompt = build_prompt(item["context"], item["endings"], "baseline", templates)
        response = generate(prompt, model, base_url, temperature=0, seed=seed)
        predicted = extract_answer(response)
        is_correct = predicted == item["label"]
        if is_correct:
            correct += 1
        total += 1
        predictions.append({
            "label": item["label"],
            "predicted": predicted,
            "correct": is_correct,
            "response": response.strip()[:200],
        })

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def run_improved(
    examples: list,
    model: str,
    base_url: str,
    templates: dict,
    best_config_name: str,
    use_self_consistency: bool = False,
    k: int = 5,
    seed: int = 42,
) -> dict:
    """Run improved evaluation with optimized template."""
    template_name = best_config_name.replace("template_", "")
    if template_name not in templates:
        template_name = "fewshot_cot"

    correct = 0
    total = 0
    predictions = []

    for item in examples:
        prompt = build_prompt(item["context"], item["endings"], template_name, templates)

        if use_self_consistency:
            sc_result = self_consistency_vote(prompt, model, base_url, k=k)
            predicted = sc_result["answer"]
            extra = {"confidence": sc_result["confidence"], "votes": sc_result["votes"]}
        else:
            response = generate(prompt, model, base_url, temperature=0, seed=seed)
            predicted = extract_answer(response)
            extra = {"response": response.strip()[:200]}

        is_correct = predicted == item["label"]
        if is_correct:
            correct += 1
        total += 1
        predictions.append({
            "label": item["label"],
            "predicted": predicted,
            "correct": is_correct,
            **extra,
        })

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "template": template_name,
        "self_consistency": use_self_consistency,
        "predictions": predictions,
    }


def compute_confidence_interval(accuracy: float, n: int, confidence: float = 0.95) -> tuple:
    """Compute Wilson score confidence interval for proportion."""
    if n == 0:
        return (0, 0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (accuracy + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((accuracy * (1 - accuracy) + z**2 / (4 * n)) / n) / denominator
    return (max(0, center - spread), min(1, center + spread))


def statistical_test(baseline_preds: list, improved_preds: list) -> dict:
    """McNemar's test for paired proportions."""
    b_correct = [p["correct"] for p in baseline_preds]
    i_correct = [p["correct"] for p in improved_preds]

    n = min(len(b_correct), len(i_correct))
    b_correct = b_correct[:n]
    i_correct = i_correct[:n]

    # b_wrong, i_right vs b_right, i_wrong
    b01 = sum(1 for b, i in zip(b_correct, i_correct) if not b and i)
    b10 = sum(1 for b, i in zip(b_correct, i_correct) if b and not i)

    if b01 + b10 == 0:
        return {"test": "mcnemar", "p_value": 1.0, "significant": False}

    # McNemar's chi-squared
    chi2 = (abs(b01 - b10) - 1)**2 / (b01 + b10) if (b01 + b10) > 0 else 0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "test": "mcnemar",
        "chi2": chi2,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "improved_only": b01,
        "regressed_only": b10,
    }


def main():
    parser = argparse.ArgumentParser(description="Run improved inference")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--self-consistency", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    templates_path = DATA_DIR / "prompt_templates.json"
    with open(templates_path) as f:
        templates = json.load(f)

    fewshot_path = DATA_DIR / "fewshot_pool.json"
    with open(fewshot_path) as f:
        examples = json.load(f)

    examples = examples[:args.max_examples]
    best = load_best_config()

    print(f"[infer] Model: {args.model}")
    print(f"[infer] Best config from ablation: {best['name']} ({best['accuracy']:.4f})")
    print(f"[infer] Examples: {len(examples)}")
    print(f"[infer] Self-consistency: {args.self_consistency} (k={args.k})")

    print("\n[infer] Running baseline...")
    baseline = run_baseline(examples, args.model, args.base_url, templates, args.seed)
    print(f"  Baseline accuracy: {baseline['accuracy']:.4f}")

    print("\n[infer] Running improved...")
    improved = run_improved(
        examples, args.model, args.base_url, templates,
        best["name"], args.self_consistency, args.k, args.seed,
    )
    print(f"  Improved accuracy: {improved['accuracy']:.4f}")

    lift = improved["accuracy"] - baseline["accuracy"]
    baseline_ci = compute_confidence_interval(baseline["accuracy"], baseline["total"])
    improved_ci = compute_confidence_interval(improved["accuracy"], improved["total"])

    stat_test = statistical_test(baseline["predictions"], improved["predictions"])

    report = {
        "model": args.model,
        "seed": args.seed,
        "n_examples": len(examples),
        "baseline": {
            "accuracy": baseline["accuracy"],
            "ci_95": baseline_ci,
        },
        "improved": {
            "accuracy": improved["accuracy"],
            "ci_95": improved_ci,
            "template": improved["template"],
            "self_consistency": improved["self_consistency"],
        },
        "lift": lift,
        "statistical_test": stat_test,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline:  {baseline['accuracy']:.4f}  CI: [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
    print(f"  Improved:  {improved['accuracy']:.4f}  CI: [{improved_ci[0]:.4f}, {improved_ci[1]:.4f}]")
    print(f"  Lift:      {lift:+.4f}")
    print(f"  p-value:   {stat_test['p_value']:.4f} ({'significant' if stat_test['significant'] else 'not significant'})")


if __name__ == "__main__":
    main()

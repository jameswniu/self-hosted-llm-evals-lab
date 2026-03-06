#!/usr/bin/env python3
"""
Prompt Optimization via Ablation

Implements inference-time optimization strategies:
1. Prompt template rewriting (instruction design)
2. Few-shot example selection
3. Chain-of-thought prompting
4. Self-consistency (k-sample decoding + majority voting)
5. Output normalization and answer extraction

No finetuning or parameter updates. Same model, same Ollama config.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import requests


DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
IMPROVE_DIR = Path(__file__).parent
DATA_DIR = IMPROVE_DIR / "data"
RESULTS_DIR = IMPROVE_DIR / "results"

ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "a": 0, "b": 1, "c": 2, "d": 3}


def generate(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0,
    max_tokens: int = 128,
    seed: Optional[int] = 42,
    top_p: float = 1.0,
    top_k: int = 1,
) -> str:
    """Generate a single completion."""
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
    }
    if seed is not None:
        options["seed"] = seed

    resp = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def extract_answer(response: str) -> Optional[int]:
    """
    Extract answer choice (0-3) from model response.
    Handles multiple formats: "A", "A.", "(A)", "Answer: A", etc.
    """
    text = response.strip()

    # Direct letter at start
    match = re.match(r"^[(\s]*([A-Da-d])[).\s,:]", text)
    if match:
        return ANSWER_MAP.get(match.group(1))

    # "Answer: X" or "answer is X"
    match = re.search(r"(?:answer|choice|option)[\s:]+(?:is\s+)?[(\s]*([A-Da-d])", text, re.I)
    if match:
        return ANSWER_MAP.get(match.group(1))

    # Standalone letter
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return ANSWER_MAP.get(match.group(1))

    return None


def self_consistency_vote(
    prompt: str,
    model: str,
    base_url: str,
    k: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 128,
) -> dict:
    """
    Self-consistency: generate k responses and take majority vote.
    Uses different seeds for diversity.
    """
    answers = []
    raw_responses = []

    for i in range(k):
        resp = generate(
            prompt, model, base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=42 + i,  # different seed each sample
            top_p=0.95,
            top_k=40,
        )
        raw_responses.append(resp)
        answer = extract_answer(resp)
        if answer is not None:
            answers.append(answer)

    if not answers:
        return {"answer": None, "confidence": 0, "k": k, "votes": {}}

    vote_counts = Counter(answers)
    winner, winner_count = vote_counts.most_common(1)[0]

    return {
        "answer": winner,
        "confidence": winner_count / len(answers),
        "k": k,
        "valid_votes": len(answers),
        "votes": dict(vote_counts),
        "raw_responses": raw_responses,
    }


def build_prompt(
    context: str,
    endings: list,
    template_name: str = "baseline",
    templates: dict = None,
) -> str:
    """Build a prompt from template and HellaSwag item."""
    if templates is None:
        templates_path = DATA_DIR / "prompt_templates.json"
        with open(templates_path) as f:
            templates = json.load(f)

    template = templates[template_name]
    return template.format(
        context=context,
        ending_0=endings[0],
        ending_1=endings[1],
        ending_2=endings[2],
        ending_3=endings[3],
    )


def run_ablation(
    examples: list,
    model: str,
    base_url: str,
    templates: dict,
    max_examples: int = 50,
) -> dict:
    """
    Run ablation study across template variants and decoding configs.
    Returns results for each configuration.
    """
    configs_path = DATA_DIR / "decoding_configs.json"
    with open(configs_path) as f:
        decoding_configs = json.load(f)

    results = {}
    subset = examples[:max_examples]

    # Test each template with greedy decoding
    for template_name in templates:
        print(f"\n[optimize] Testing template: {template_name}")
        correct = 0
        total = 0
        before_after = []

        for item in subset:
            prompt = build_prompt(
                item["context"], item["endings"], template_name, templates
            )
            response = generate(prompt, model, base_url, temperature=0, max_tokens=128)
            predicted = extract_answer(response)
            is_correct = predicted == item["label"]
            if is_correct:
                correct += 1
            total += 1

            before_after.append({
                "context": item["context"][:100],
                "label": item["label"],
                "predicted": predicted,
                "correct": is_correct,
                "response": response.strip()[:200],
            })

        accuracy = correct / total if total > 0 else 0
        results[f"template_{template_name}"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "examples": before_after[:15],
        }
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Test self-consistency on best template
    best_template = max(
        [(k, v["accuracy"]) for k, v in results.items()],
        key=lambda x: x[1],
    )[0]
    best_template_name = best_template.replace("template_", "")
    print(f"\n[optimize] Self-consistency (k=5) with best template: {best_template_name}")

    correct_sc = 0
    total_sc = 0
    sc_examples = []

    for item in subset[:30]:  # fewer for self-consistency (5x cost)
        prompt = build_prompt(
            item["context"], item["endings"], best_template_name, templates
        )
        sc_result = self_consistency_vote(prompt, model, base_url, k=5)
        predicted = sc_result["answer"]
        is_correct = predicted == item["label"]
        if is_correct:
            correct_sc += 1
        total_sc += 1

        sc_examples.append({
            "context": item["context"][:100],
            "label": item["label"],
            "predicted": predicted,
            "correct": is_correct,
            "confidence": sc_result["confidence"],
            "votes": sc_result["votes"],
        })

    sc_accuracy = correct_sc / total_sc if total_sc > 0 else 0
    results["self_consistency_k5"] = {
        "accuracy": sc_accuracy,
        "correct": correct_sc,
        "total": total_sc,
        "base_template": best_template_name,
        "examples": sc_examples[:15],
    }
    print(f"  Accuracy: {sc_accuracy:.4f} ({correct_sc}/{total_sc})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prompt Optimizer")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    templates_path = DATA_DIR / "prompt_templates.json"
    with open(templates_path) as f:
        templates = json.load(f)

    fewshot_path = DATA_DIR / "fewshot_pool.json"
    with open(fewshot_path) as f:
        fewshot_pool = json.load(f)

    print(f"[optimize] Model: {args.model}")
    print(f"[optimize] Templates: {list(templates.keys())}")
    print(f"[optimize] Few-shot pool: {len(fewshot_pool)} examples")
    print(f"[optimize] Max eval examples: {args.max_examples}")

    results = run_ablation(
        fewshot_pool, args.model, args.base_url,
        templates, args.max_examples,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[optimize] Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for config_name, data in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"  {config_name}: {data['accuracy']:.4f} ({data['correct']}/{data['total']})")


if __name__ == "__main__":
    main()

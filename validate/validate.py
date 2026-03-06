#!/usr/bin/env python3
"""
Determinism & Output Validation

Tests:
1. Deterministic mode: seed + temperature=0 + top_p=1
2. Identical prompts yield identical responses across N runs
3. Lightweight validation logic (regex/schema) for custom task outputs

Usage:
    python validate/validate.py --model llama3.1:8b --seed 42
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from typing import Optional

import requests


DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
NUM_DETERMINISM_TRIALS = 5


# ─── Output Validators ──────────────────────────────────────────────

class OutputValidator:
    """Base class for output validation."""

    def validate(self, prompt: str, response: str) -> dict:
        raise NotImplementedError


class NumericValidator(OutputValidator):
    """Validates that response contains a numeric answer."""
    PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

    def validate(self, prompt: str, response: str) -> dict:
        matches = self.PATTERN.findall(response.strip())
        return {
            "valid": len(matches) > 0,
            "extracted": matches[0] if matches else None,
            "rule": "numeric_answer",
            "raw": response.strip()[:100],
        }


class SingleWordValidator(OutputValidator):
    """Validates single-word or short-phrase responses."""
    MAX_WORDS = 5

    def validate(self, prompt: str, response: str) -> dict:
        clean = response.strip().rstrip(".")
        word_count = len(clean.split())
        return {
            "valid": 0 < word_count <= self.MAX_WORDS,
            "word_count": word_count,
            "rule": f"max_{self.MAX_WORDS}_words",
            "raw": clean[:100],
        }


class ClassificationValidator(OutputValidator):
    """Validates classification output against allowed labels."""

    def __init__(self, allowed_labels: list):
        self.allowed_labels = [l.lower() for l in allowed_labels]

    def validate(self, prompt: str, response: str) -> dict:
        clean = response.strip().lower().rstrip(".")
        matched = any(label in clean for label in self.allowed_labels)
        return {
            "valid": matched,
            "extracted": clean[:50],
            "allowed": self.allowed_labels,
            "rule": "classification_label",
        }


class SchemaValidator(OutputValidator):
    """Validates JSON-structured output against expected keys."""

    def __init__(self, required_keys: list):
        self.required_keys = required_keys

    def validate(self, prompt: str, response: str) -> dict:
        try:
            data = json.loads(response.strip())
            missing = [k for k in self.required_keys if k not in data]
            return {
                "valid": len(missing) == 0,
                "missing_keys": missing,
                "rule": "json_schema",
            }
        except json.JSONDecodeError:
            return {
                "valid": False,
                "error": "not_valid_json",
                "rule": "json_schema",
            }


# ─── Test Cases ──────────────────────────────────────────────────────

DETERMINISM_PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 7 * 8? Answer with just the number.",
    "List the first 5 prime numbers separated by commas.",
    "Is water wet? Answer yes or no.",
    "What programming language was created by Guido van Rossum?",
]

VALIDATION_TESTS = [
    {
        "prompt": "What is 15 * 23? Answer with just the number.",
        "validator": NumericValidator(),
        "expected_pattern": r"345",
    },
    {
        "prompt": "What is the capital of Japan? Answer in one word.",
        "validator": SingleWordValidator(),
    },
    {
        "prompt": 'Classify this sentiment as positive, negative, or neutral: "I love this product!"',
        "validator": ClassificationValidator(["positive", "negative", "neutral"]),
    },
    {
        "prompt": "What is the chemical symbol for gold? Answer with just the symbol.",
        "validator": SingleWordValidator(),
    },
    {
        "prompt": "What is 100 divided by 4? Answer with just the number.",
        "validator": NumericValidator(),
        "expected_pattern": r"25",
    },
]


def generate_deterministic(
    prompt: str,
    model: str,
    base_url: str,
    seed: int = 42,
    max_tokens: int = 128,
) -> str:
    """Generate with full deterministic settings."""
    resp = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "seed": seed,
                "num_predict": max_tokens,
                "repeat_penalty": 1.0,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def test_determinism(model: str, base_url: str, seed: int, trials: int) -> dict:
    """
    Verify identical prompts yield identical responses across N trials.
    Returns per-prompt consistency results.
    """
    print(f"\n{'='*70}")
    print(f"DETERMINISM TEST (seed={seed}, trials={trials})")
    print(f"{'='*70}")

    results = {}
    all_consistent = True

    for prompt in DETERMINISM_PROMPTS:
        responses = []
        for trial in range(trials):
            resp = generate_deterministic(prompt, model, base_url, seed)
            responses.append(resp.strip())

        unique_responses = set(responses)
        is_consistent = len(unique_responses) == 1
        if not is_consistent:
            all_consistent = False

        freq = Counter(responses)
        results[prompt[:50]] = {
            "consistent": is_consistent,
            "unique_count": len(unique_responses),
            "trials": trials,
            "most_common": freq.most_common(1)[0] if freq else None,
        }

        status = "PASS" if is_consistent else "FAIL"
        print(f"\n  [{status}] \"{prompt[:60]}...\"")
        print(f"    Unique responses: {len(unique_responses)}/{trials}")
        if not is_consistent:
            for resp, count in freq.most_common():
                print(f"    [{count}x] \"{resp[:80]}\"")

    print(f"\n  Overall: {'ALL CONSISTENT' if all_consistent else 'NONDETERMINISM DETECTED'}")
    return {"prompts_tested": len(DETERMINISM_PROMPTS), "all_consistent": all_consistent, "details": results}


def test_validation(model: str, base_url: str, seed: int) -> dict:
    """
    Run output validation tests against custom task responses.
    """
    print(f"\n{'='*70}")
    print("OUTPUT VALIDATION TEST")
    print(f"{'='*70}")

    results = []
    pass_count = 0

    for test in VALIDATION_TESTS:
        response = generate_deterministic(test["prompt"], model, base_url, seed, max_tokens=64)
        validation = test["validator"].validate(test["prompt"], response)

        passed = validation["valid"]
        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {test['prompt'][:60]}")
        print(f"    Response: \"{response.strip()[:80]}\"")
        print(f"    Validation: {validation}")

        results.append({
            "prompt": test["prompt"],
            "response": response.strip(),
            "validation": validation,
            "passed": passed,
        })

    total = len(VALIDATION_TESTS)
    print(f"\n  Results: {pass_count}/{total} passed")

    return {"total": total, "passed": pass_count, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Guardrails & Determinism Validator")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=NUM_DETERMINISM_TRIALS)
    parser.add_argument("--output", default="validate/results.json")
    args = parser.parse_args()

    print(f"[validate] Model: {args.model}")
    print(f"[validate] Seed: {args.seed}")
    print(f"[validate] Deterministic settings: temperature=0, top_p=1, top_k=1")

    determinism_results = test_determinism(args.model, args.base_url, args.seed, args.trials)
    validation_results = test_validation(args.model, args.base_url, args.seed)

    report = {
        "model": args.model,
        "seed": args.seed,
        "determinism": determinism_results,
        "validation": validation_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[validate] Full report saved to {args.output}")


if __name__ == "__main__":
    main()

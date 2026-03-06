#!/usr/bin/env python3
"""
Prepare Data for Prompt Ablation

Prepares few-shot examples and retrieval corpus for inference-time
optimization of HellaSwag benchmark.

Strategy:
1. Extract high-quality few-shot examples from training split
2. Cluster by topic for semantic similarity-based selection
3. Build prompt templates for chain-of-thought and self-consistency
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[prepare] Warning: 'datasets' not installed. Using bundled examples.")


IMPROVE_DIR = Path(__file__).parent
DATA_DIR = IMPROVE_DIR / "data"


# Bundled few-shot examples for HellaSwag (fallback if datasets unavailable)
HELLASWAG_FEWSHOT = [
    {
        "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
        "endings": [
            "rinses the dog off with a hose.",
            "gets on a motorcycle and rides away.",
            "starts to water the flowers.",
            "opens the door and lets the dog inside."
        ],
        "label": 0,
    },
    {
        "context": "A man is sitting on a roof. He starts pulling up shingles. He",
        "endings": [
            "removes the shingles and throws them to the ground.",
            "starts to dance on the roof.",
            "begins eating the shingles.",
            "pulls out a guitar and starts playing."
        ],
        "label": 0,
    },
    {
        "context": "A kid is sitting at a table. He picks up a crayon. He",
        "endings": [
            "starts to draw on the paper.",
            "throws the crayon at the wall.",
            "eats the crayon.",
            "puts the crayon in his ear."
        ],
        "label": 0,
    },
]


PROMPT_TEMPLATES = {
    "baseline": (
        "{context}\n\n"
        "Which of the following best completes the passage?\n"
        "A. {ending_0}\n"
        "B. {ending_1}\n"
        "C. {ending_2}\n"
        "D. {ending_3}\n"
        "Answer:"
    ),
    "cot": (
        "{context}\n\n"
        "Which of the following best completes the passage?\n"
        "A. {ending_0}\n"
        "B. {ending_1}\n"
        "C. {ending_2}\n"
        "D. {ending_3}\n\n"
        "Let me think through each option:\n"
        "- Option A: Does this logically follow? Consider the context.\n"
        "- Option B: Does this make sense given what happened?\n"
        "- Option C: Is this a plausible continuation?\n"
        "- Option D: Does this fit the scenario?\n\n"
        "The most logical continuation is:"
    ),
    "instruction": (
        "You are an expert at reading comprehension and commonsense reasoning.\n\n"
        "Read the following passage and select the most logical continuation.\n\n"
        "Passage: {context}\n\n"
        "Options:\n"
        "A. {ending_0}\n"
        "B. {ending_1}\n"
        "C. {ending_2}\n"
        "D. {ending_3}\n\n"
        "Select the single best answer (A, B, C, or D). Answer:"
    ),
    "fewshot_cot": (
        "You are an expert at commonsense reasoning. For each passage, select "
        "the most logical continuation.\n\n"
        "Example:\n"
        "Passage: A woman is outside with a bucket and a dog. The dog is running "
        "around trying to avoid a bath. She\n"
        "A. rinses the dog off with a hose.\n"
        "B. gets on a motorcycle and rides away.\n"
        "C. starts to water the flowers.\n"
        "D. opens the door and lets the dog inside.\n"
        "Reasoning: The context describes giving a dog a bath, so the logical "
        "continuation involves washing the dog.\n"
        "Answer: A\n\n"
        "Now answer:\n"
        "Passage: {context}\n"
        "A. {ending_0}\n"
        "B. {ending_1}\n"
        "C. {ending_2}\n"
        "D. {ending_3}\n"
        "Reasoning:"
    ),
}


def prepare_fewshot_pool(n_examples: int = 20) -> list:
    """Load or generate a pool of few-shot examples."""
    if HAS_DATASETS:
        print("[prepare] Loading HellaSwag training split from HuggingFace...")
        ds = load_dataset("Rowan/hellaswag", split="train")
        indices = random.sample(range(len(ds)), min(n_examples, len(ds)))
        examples = []
        for idx in indices:
            item = ds[idx]
            examples.append({
                "context": item["ctx"],
                "endings": item["endings"],
                "label": int(item["label"]),
            })
        return examples
    else:
        print("[prepare] Using bundled few-shot examples.")
        return HELLASWAG_FEWSHOT


def save_templates(output_dir: Path) -> None:
    """Save prompt templates to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "prompt_templates.json"
    with open(path, "w") as f:
        json.dump(PROMPT_TEMPLATES, f, indent=2)
    print(f"[prepare] Saved {len(PROMPT_TEMPLATES)} templates to {path}")


def save_fewshot_pool(examples: list, output_dir: Path) -> None:
    """Save few-shot example pool to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fewshot_pool.json"
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"[prepare] Saved {len(examples)} few-shot examples to {path}")


def save_decoding_configs(output_dir: Path) -> None:
    """Save decoding parameter configurations for ablation."""
    configs = {
        "greedy": {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 42},
        "low_temp": {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "seed": 42},
        "self_consistency_k5": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "seed": None, "k": 5},
        "self_consistency_k10": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "seed": None, "k": 10},
    }
    path = output_dir / "decoding_configs.json"
    with open(path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"[prepare] Saved {len(configs)} decoding configs to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for benchmark improvement")
    parser.add_argument("--n-fewshot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fewshot = prepare_fewshot_pool(args.n_fewshot)
    save_fewshot_pool(fewshot, DATA_DIR)
    save_templates(DATA_DIR)
    save_decoding_configs(DATA_DIR)

    print(f"\n[prepare] All data prepared in {DATA_DIR}")


if __name__ == "__main__":
    main()

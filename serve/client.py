#!/usr/bin/env python3
"""
Sample Client for Ollama Inference Endpoint

Demonstrates prompt generation via both native Ollama API and
OpenAI-compatible API (used by lm-evaluation-harness).
"""

import argparse
import json
import time
import requests
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"

SAMPLE_PROMPTS = [
    {
        "label": "factual_qa",
        "prompt": "What is the capital of France? Answer in one word.",
    },
    {
        "label": "reasoning",
        "prompt": "If a train travels 60 mph for 2.5 hours, how far does it go? Show your work briefly.",
    },
    {
        "label": "code_gen",
        "prompt": "Write a Python function that checks if a string is a palindrome.",
    },
    {
        "label": "classification",
        "prompt": 'Classify the sentiment of this review as positive, negative, or neutral: "The food was okay but the service was terrible." Respond with only the classification.',
    },
    {
        "label": "instruction_following",
        "prompt": "List exactly 3 benefits of exercise. Use numbered format. Do not include any other text.",
    },
]


def generate_ollama_native(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: Optional[int] = 42,
) -> dict:
    """Generate using Ollama native API (non-streaming)."""
    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
                "top_p": 1.0,
            },
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    return {
        "response": data.get("response", ""),
        "model": data.get("model", model),
        "total_duration_ms": data.get("total_duration", 0) / 1e6,
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
        "tokens_per_sec": (
            data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9)
            if data.get("eval_duration", 0) > 0
            else 0
        ),
        "wall_time_sec": elapsed,
    }


def generate_openai_compat(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: Optional[int] = 42,
) -> dict:
    """Generate using OpenAI-compatible /v1/chat/completions endpoint."""
    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "top_p": 1.0,
            "stream": False,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "response": content,
        "model": data.get("model", model),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_time_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Ollama Client Demo")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--api", choices=["native", "openai", "both"], default="both")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"Ollama Client Demo | Model: {args.model} | API: {args.api}")
    print(f"{'='*70}\n")

    for i, sample in enumerate(SAMPLE_PROMPTS, 1):
        print(f"--- [{i}/{len(SAMPLE_PROMPTS)}] {sample['label']} ---")
        print(f"Prompt: {sample['prompt'][:80]}...")

        if args.api in ("native", "both"):
            result = generate_ollama_native(
                sample["prompt"], args.model, args.base_url
            )
            print(f"\n[Native API]")
            print(f"  Response: {result['response'][:200]}")
            print(f"  Tokens/sec: {result['tokens_per_sec']:.1f}")
            print(f"  Wall time: {result['wall_time_sec']:.2f}s")

        if args.api in ("openai", "both"):
            result = generate_openai_compat(
                sample["prompt"], args.model, args.base_url
            )
            print(f"\n[OpenAI-compat API]")
            print(f"  Response: {result['response'][:200]}")
            print(f"  Tokens: {result['prompt_tokens']}p + {result['completion_tokens']}c")
            print(f"  Wall time: {result['wall_time_sec']:.2f}s")

        print()

    print(f"{'='*70}")
    print("All sample generations complete.")


if __name__ == "__main__":
    main()

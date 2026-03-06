#!/usr/bin/env python3
"""
Performance Load Generator

Sends concurrent requests to the Ollama endpoint and collects:
- Time-to-first-token (TTFT)
- Tokens per second (TPOT)
- P50 / P95 / P99 latency
- GPU utilization (if available via nvidia-smi)

Compares across batch sizes, caching, and stop-sequence settings.

Usage:
    python perf/load_test.py --model llama3.1:8b --output perf/metrics.csv
"""

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests


DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"

SHORT_PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of France.",
    "What color is the sky?",
    "Say hello in Spanish.",
    "What is Python?",
]

LONG_PROMPTS = [
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational lensing, and time dilation. Include historical context and modern applications.",
    "Write a comprehensive comparison of relational databases and NoSQL databases. Cover ACID properties, CAP theorem, use cases, scaling strategies, and when to choose each.",
    "Describe the complete lifecycle of a machine learning project from data collection to production deployment. Include data preprocessing, feature engineering, model selection, training, evaluation, and monitoring.",
    "Explain how modern CPU architectures work, covering instruction pipelining, branch prediction, cache hierarchy, out-of-order execution, and SIMD operations.",
    "Provide a detailed overview of distributed systems consensus algorithms including Paxos, Raft, and Byzantine fault tolerance. Explain the tradeoffs between consistency, availability, and partition tolerance.",
]


@dataclass
class RequestMetrics:
    prompt_type: str        # short or long
    prompt_length: int      # character count
    concurrency: int        # concurrent request count
    batch_label: str        # config label
    ttft_ms: float          # time to first token
    total_time_ms: float    # total request time
    tokens_generated: int   # output token count
    tokens_per_sec: float   # throughput
    success: bool
    error: Optional[str] = None
    stop_sequence: Optional[str] = None
    cached: bool = False


def send_request_streaming(
    prompt: str,
    model: str,
    base_url: str,
    max_tokens: int = 256,
    stop: list = None,
    seed: int = 42,
) -> dict:
    """Send a streaming request to measure TTFT accurately."""
    start = time.perf_counter()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0,
                    "num_predict": max_tokens,
                    "seed": seed,
                    "top_p": 1.0,
                    "stop": stop or [],
                },
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                if chunk.get("response"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response += chunk["response"]
                    token_count += 1
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", token_count)
                    break

        end = time.perf_counter()

        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        total = (end - start) * 1000
        tps = eval_count / ((end - (first_token_time or start))) if first_token_time else 0

        return {
            "ttft_ms": ttft,
            "total_time_ms": total,
            "tokens_generated": eval_count,
            "tokens_per_sec": tps,
            "success": True,
            "error": None,
        }
    except Exception as e:
        end = time.perf_counter()
        return {
            "ttft_ms": 0,
            "total_time_ms": (end - start) * 1000,
            "tokens_generated": 0,
            "tokens_per_sec": 0,
            "success": False,
            "error": str(e),
        }


def send_request_non_streaming(
    prompt: str,
    model: str,
    base_url: str,
    max_tokens: int = 256,
    stop: list = None,
    seed: int = 42,
) -> dict:
    """Send a non-streaming request (used for batch comparison)."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": max_tokens,
                    "seed": seed,
                    "top_p": 1.0,
                    "stop": stop or [],
                },
            },
            timeout=120,
        )
        end = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()

        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1)
        prompt_eval_duration = data.get("prompt_eval_duration", 0)

        return {
            "ttft_ms": prompt_eval_duration / 1e6,
            "total_time_ms": (end - start) * 1000,
            "tokens_generated": eval_count,
            "tokens_per_sec": eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0,
            "success": True,
            "error": None,
        }
    except Exception as e:
        end = time.perf_counter()
        return {
            "ttft_ms": 0,
            "total_time_ms": (end - start) * 1000,
            "tokens_generated": 0,
            "tokens_per_sec": 0,
            "success": False,
            "error": str(e),
        }


def get_gpu_utilization() -> Optional[dict]:
    """Query nvidia-smi for GPU utilization if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu_util_pct": float(parts[0]),
                "mem_util_pct": float(parts[1]),
                "mem_used_mb": float(parts[2]),
                "mem_total_mb": float(parts[3]),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def run_load_test(
    prompts: list,
    prompt_type: str,
    concurrency: int,
    model: str,
    base_url: str,
    batch_label: str,
    max_tokens: int = 256,
    stop: list = None,
    use_streaming: bool = True,
) -> list:
    """Run concurrent requests and collect metrics."""
    metrics = []
    send_fn = send_request_streaming if use_streaming else send_request_non_streaming

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for prompt in prompts:
            future = executor.submit(
                send_fn, prompt, model, base_url, max_tokens, stop
            )
            futures[future] = prompt

        for future in as_completed(futures):
            prompt = futures[future]
            result = future.result()
            metrics.append(RequestMetrics(
                prompt_type=prompt_type,
                prompt_length=len(prompt),
                concurrency=concurrency,
                batch_label=batch_label,
                ttft_ms=result["ttft_ms"],
                total_time_ms=result["total_time_ms"],
                tokens_generated=result["tokens_generated"],
                tokens_per_sec=result["tokens_per_sec"],
                success=result["success"],
                error=result.get("error"),
                stop_sequence=str(stop) if stop else None,
            ))

    return metrics


def compute_percentiles(values: list) -> dict:
    """Compute P50, P95, P99 percentiles."""
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def print_analysis(all_metrics: list) -> None:
    """Print analysis summary grouped by configuration."""
    configs = {}
    for m in all_metrics:
        key = (m.prompt_type, m.concurrency, m.batch_label)
        if key not in configs:
            configs[key] = []
        configs[key].append(m)

    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*80}")

    for (ptype, conc, label), group in sorted(configs.items()):
        successful = [m for m in group if m.success]
        if not successful:
            continue

        ttfts = [m.ttft_ms for m in successful]
        totals = [m.total_time_ms for m in successful]
        tps_vals = [m.tokens_per_sec for m in successful if m.tokens_per_sec > 0]

        ttft_pct = compute_percentiles(ttfts)
        total_pct = compute_percentiles(totals)

        print(f"\n[{label}] {ptype} prompts | concurrency={conc}")
        print(f"  Requests: {len(group)} total, {len(successful)} success, {len(group)-len(successful)} failed")
        print(f"  TTFT:     P50={ttft_pct['p50']:.1f}ms  P95={ttft_pct['p95']:.1f}ms  P99={ttft_pct['p99']:.1f}ms")
        print(f"  Latency:  P50={total_pct['p50']:.1f}ms  P95={total_pct['p95']:.1f}ms  P99={total_pct['p99']:.1f}ms")
        if tps_vals:
            print(f"  Tokens/s: mean={np.mean(tps_vals):.1f}  min={np.min(tps_vals):.1f}  max={np.max(tps_vals):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Load Test for Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", default="perf/metrics.csv")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    all_metrics = []

    test_configs = [
        {"prompts": SHORT_PROMPTS, "type": "short", "concurrency": 1, "label": "short_c1", "stop": None},
        {"prompts": SHORT_PROMPTS, "type": "short", "concurrency": 3, "label": "short_c3", "stop": None},
        {"prompts": SHORT_PROMPTS, "type": "short", "concurrency": 5, "label": "short_c5", "stop": None},
        {"prompts": LONG_PROMPTS, "type": "long", "concurrency": 1, "label": "long_c1", "stop": None},
        {"prompts": LONG_PROMPTS, "type": "long", "concurrency": 3, "label": "long_c3", "stop": None},
        {"prompts": SHORT_PROMPTS, "type": "short", "concurrency": 1, "label": "short_c1_stop", "stop": ["\n", "."]},
        {"prompts": LONG_PROMPTS, "type": "long", "concurrency": 1, "label": "long_c1_stop", "stop": ["\n\n"]},
    ]

    print(f"[perf] Starting load tests against {args.base_url}")
    print(f"[perf] Model: {args.model}")
    print(f"[perf] Max tokens: {args.max_tokens}")
    print(f"[perf] Configurations: {len(test_configs)}\n")

    for i, config in enumerate(test_configs, 1):
        print(f"[perf] Running {i}/{len(test_configs)}: {config['label']}...")

        gpu_before = get_gpu_utilization()

        metrics = run_load_test(
            prompts=config["prompts"],
            prompt_type=config["type"],
            concurrency=config["concurrency"],
            model=args.model,
            base_url=args.base_url,
            batch_label=config["label"],
            max_tokens=args.max_tokens,
            stop=config["stop"],
        )

        gpu_after = get_gpu_utilization()
        all_metrics.extend(metrics)

        if gpu_after:
            print(f"  GPU: {gpu_after['gpu_util_pct']}% util, {gpu_after['mem_used_mb']:.0f}MB used")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in asdict(all_metrics[0]).keys()])
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(asdict(m))

    print(f"\n[perf] Metrics saved to {args.output}")

    print_analysis(all_metrics)


if __name__ == "__main__":
    main()

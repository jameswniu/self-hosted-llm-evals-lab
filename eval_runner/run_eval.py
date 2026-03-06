#!/usr/bin/env python3
"""
Evaluation Runner

Runs lm-evaluation-harness benchmarks against the Ollama-served model.
Supports official tasks (MMLU, HellaSwag) and custom JSON-based benchmarks.

Usage:
    python eval_runner/run_eval.py --tasks hellaswag,mmlu --limit 100
    python eval_runner/run_eval.py --tasks custom_bench --include-path eval_runner/custom_task
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
RESULTS_DIR = Path(__file__).parent / "results"


def run_lm_eval(
    model: str,
    base_url: str,
    tasks: str,
    limit: int = 100,
    num_fewshot: int = 0,
    output_dir: str = str(RESULTS_DIR),
    include_path: str = None,
    batch_size: str = "1",
    seed: int = 42,
    extra_args: list = None,
) -> dict:
    """
    Execute lm-eval-harness via CLI against Ollama's OpenAI-compatible endpoint.

    Uses the local-chat-completions backend which hits /v1/chat/completions.
    This is the recommended integration path for Ollama.
    """
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", (
            f"model={model},"
            f"base_url={base_url}/v1/chat/completions,"
            f"num_concurrent=1,"
            f"max_retries=3,"
            f"tokenized_requests=False"
        ),
        "--apply_chat_template",
        "--tasks", tasks,
        "--limit", str(limit),
        "--batch_size", batch_size,
        "--seed", str(seed),
        "--output_path", output_dir,
        "--log_samples",
        "--cache_requests", "true",
    ]

    if num_fewshot > 0:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if include_path:
        cmd.extend(["--include_path", include_path])

    if extra_args:
        cmd.extend(extra_args)

    print(f"[eval] Running: {' '.join(cmd)}")
    print(f"[eval] Tasks: {tasks}")
    print(f"[eval] Limit: {limit} samples per task")
    print(f"[eval] Output: {output_dir}")
    print()

    start = datetime.now()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n[eval] Completed in {elapsed:.1f}s (exit code: {result.returncode})")

    return {
        "tasks": tasks,
        "limit": limit,
        "elapsed_sec": elapsed,
        "exit_code": result.returncode,
    }


def parse_results(results_dir: str) -> pd.DataFrame:
    """
    Parse lm-eval output JSON files into a summary DataFrame.
    """
    results_path = Path(results_dir)
    rows = []

    for json_file in results_path.rglob("results_*.json"):
        with open(json_file) as f:
            data = json.load(f)

        results = data.get("results", {})
        config = data.get("config", {})

        for task_name, metrics in results.items():
            row = {
                "task": task_name,
                "model": config.get("model", "unknown"),
                "num_fewshot": config.get("num_fewshot", 0),
                "limit": config.get("limit", None),
            }
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[metric_name] = value
                elif metric_name.endswith(",none"):
                    clean_name = metric_name.replace(",none", "")
                    row[clean_name] = value
            rows.append(row)

    if not rows:
        print("[eval] No results found to parse.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a formatted summary of evaluation results."""
    if df.empty:
        print("[eval] No results to display.")
        return

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*70}")

    score_cols = [c for c in df.columns if "acc" in c.lower() or "score" in c.lower()]

    for _, row in df.iterrows():
        print(f"\nTask: {row['task']}")
        print(f"  Model: {row.get('model', 'unknown')}")
        print(f"  Few-shot: {row.get('num_fewshot', 0)}")
        for col in score_cols:
            if col in row and pd.notna(row[col]):
                print(f"  {col}: {row[col]:.4f}")

    print(f"\n{'='*70}")


def save_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Save summary table as CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[eval] Summary saved to {csv_path}")

    json_path = output_path / "summary.json"
    df.to_json(json_path, orient="records", indent=2)


def main():
    parser = argparse.ArgumentParser(description="LM Evaluation Runner")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tasks", default="hellaswag,mmlu", help="Comma-separated task list")
    parser.add_argument("--limit", type=int, default=100, help="Samples per task")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--include-path", default=None, help="Path for custom tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_info = run_lm_eval(
        model=args.model,
        base_url=args.base_url,
        tasks=args.tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        output_dir=args.output_dir,
        include_path=args.include_path,
        seed=args.seed,
    )

    if run_info["exit_code"] == 0:
        df = parse_results(args.output_dir)
        print_summary_table(df)
        save_summary(df, args.output_dir)
    else:
        print("[eval] Evaluation failed. Check output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

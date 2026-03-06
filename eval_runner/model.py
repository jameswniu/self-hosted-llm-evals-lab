#!/usr/bin/env python3
"""
Custom Model Wrapper for lm-evaluation-harness

Wraps an Ollama endpoint as an LM backend compatible with EleutherAI's
evaluation harness. Supports both loglikelihood and generate_until tasks,
with built-in prompt caching for determinism and efficiency.

Usage with lm-eval CLI:
    lm_eval --model local-chat-completions \
        --model_args model=llama3.1:8b,base_url=http://localhost:11434/v1 \
        --tasks hellaswag

Or programmatically via OllamaEvalModel in this file.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests


CACHE_DIR = Path(__file__).parent / ".cache"


class PromptCache:
    """
    Disk-backed prompt cache for deterministic, efficient re-runs.

    Cache keys are SHA-256 hashes of (model, prompt, params) tuples.
    This ensures repeated evaluations with identical settings return
    cached results without hitting the inference endpoint.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, model: str, prompt: str, params: dict) -> str:
        blob = json.dumps(
            {"model": model, "prompt": prompt, **params},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, model: str, prompt: str, params: dict) -> Optional[dict]:
        key = self._key(model, prompt, params)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            with open(path) as f:
                return json.load(f)
        self.misses += 1
        return None

    def put(self, model: str, prompt: str, params: dict, result: dict) -> None:
        key = self._key(model, prompt, params)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(result, f)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
        }


class OllamaEvalModel:
    """
    Programmatic wrapper around Ollama for evaluation tasks.

    This class provides a clean interface for:
    - Non-streaming completions with caching
    - Deterministic generation (seed + temperature=0)
    - Token-level metrics (tokens/sec, latency)

    For lm-eval-harness integration, use the CLI with --model local-chat-completions
    which hits Ollama's OpenAI-compatible /v1 endpoint directly. This class
    is for custom evaluation scripts that need more control.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
        max_tokens: int = 512,
        use_cache: bool = True,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cache = PromptCache() if use_cache else None

    def _params(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> dict:
        """
        Generate a completion for the given prompt.

        Returns dict with keys: response, tokens_per_sec, wall_time_sec,
        eval_count, cached.
        """
        params = self._params()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if self.cache:
            cached = self.cache.get(self.model, prompt, params)
            if cached:
                cached["cached"] = True
                return cached

        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": params["temperature"],
                    "num_predict": params["max_tokens"],
                    "seed": params["seed"],
                    "top_p": params["top_p"],
                },
            },
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()

        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)

        result = {
            "response": data.get("response", ""),
            "eval_count": eval_count,
            "tokens_per_sec": (
                eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
            ),
            "wall_time_sec": elapsed,
            "cached": False,
        }

        if self.cache:
            self.cache.put(self.model, prompt, params, result)

        return result

    def chat(self, messages: list[dict], max_tokens: Optional[int] = None) -> dict:
        """
        Chat completion via OpenAI-compatible endpoint.
        """
        params = self._params()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        prompt_key = json.dumps(messages, sort_keys=True)

        if self.cache:
            cached = self.cache.get(self.model, prompt_key, params)
            if cached:
                cached["cached"] = True
                return cached

        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": params["temperature"],
                "max_tokens": params["max_tokens"],
                "seed": params["seed"],
                "top_p": params["top_p"],
                "stream": False,
            },
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        result = {
            "response": data["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "wall_time_sec": elapsed,
            "cached": False,
        }

        if self.cache:
            self.cache.put(self.model, prompt_key, params, result)

        return result

    def health_check(self) -> bool:
        """Verify Ollama endpoint is reachable and model is loaded."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False


if __name__ == "__main__":
    model = OllamaEvalModel()
    if model.health_check():
        print("[model] Ollama endpoint healthy, model available.")
        result = model.generate("What is 2 + 2? Answer with just the number.")
        print(f"[model] Test response: {result['response']}")
        print(f"[model] Cached: {result['cached']}")
    else:
        print("[model] ERROR: Ollama not reachable or model not loaded.")

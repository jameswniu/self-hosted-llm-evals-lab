#!/usr/bin/env python3
"""
Ollama Inference Server Manager

Starts Ollama, pulls the specified model, and verifies the endpoint is live.
One-line startup: python serve/serve.py --model llama3.1:8b
"""

import argparse
import subprocess
import sys
import time
import requests
import signal
import os


DEFAULT_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
HEALTH_TIMEOUT = 60  # seconds to wait for Ollama to start


def check_ollama_running(base_url: str) -> bool:
    """Check if Ollama server is already running."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def start_ollama_server() -> subprocess.Popen:
    """Start Ollama server as a subprocess."""
    print("[serve] Starting Ollama server...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def wait_for_health(base_url: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    """Poll the Ollama health endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        if check_ollama_running(base_url):
            return True
        time.sleep(1)
    return False


def pull_model(model: str) -> None:
    """Pull model if not already available locally."""
    print(f"[serve] Ensuring model '{model}' is available...")
    result = subprocess.run(
        ["ollama", "pull", model],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[serve] Error pulling model: {result.stderr}")
        sys.exit(1)
    print(f"[serve] Model '{model}' ready.")


def list_models(base_url: str) -> list:
    """List available models from Ollama API."""
    resp = requests.get(f"{base_url}/api/tags")
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def warm_up(model: str, base_url: str) -> None:
    """Send a warmup request to load model into memory."""
    print(f"[serve] Warming up model '{model}'...")
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 5},
            },
            timeout=120,
        )
        resp.raise_for_status()
        print(f"[serve] Warmup complete. Model loaded in memory.")
    except Exception as e:
        print(f"[serve] Warmup warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ollama Inference Server Manager")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to serve")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warmup")
    args = parser.parse_args()

    proc = None

    if not check_ollama_running(args.base_url):
        proc = start_ollama_server()
        if not wait_for_health(args.base_url):
            print("[serve] ERROR: Ollama failed to start within timeout.")
            sys.exit(1)
        print(f"[serve] Ollama server is live at {args.base_url}")
    else:
        print(f"[serve] Ollama already running at {args.base_url}")

    pull_model(args.model)

    if not args.no_warmup:
        warm_up(args.model, args.base_url)

    models = list_models(args.base_url)
    print(f"[serve] Available models: {models}")
    print(f"[serve] OpenAI-compatible endpoint: {args.base_url}/v1")
    print(f"[serve] Ready for evaluation. Press Ctrl+C to stop.")

    def shutdown(sig, frame):
        print("\n[serve] Shutting down...")
        if proc:
            proc.terminate()
            proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if proc:
        try:
            proc.wait()
        except KeyboardInterrupt:
            shutdown(None, None)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            shutdown(None, None)


if __name__ == "__main__":
    main()

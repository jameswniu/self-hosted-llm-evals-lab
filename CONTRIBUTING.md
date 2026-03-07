# Contributing

Thanks for your interest in contributing to self-hosted-llm-evals-lab.

## Getting Started

```bash
make setup     # Create venv, install deps, pull model
make serve     # Start Ollama endpoint
make eval      # Run benchmarks
```

## How to Contribute

1. Fork the repo and create a feature branch
2. Make your changes
3. Run the relevant `make` targets to verify nothing breaks
4. Open a PR with a clear description of the change

## Areas for Contribution

- Additional benchmark tasks (coding, math, reasoning)
- Support for more inference backends (vLLM, TGI, llama.cpp)
- Extended ablation across model sizes (7B, 13B, 70B)
- Full-precision vs quantized comparisons
- Visualization improvements

## Code Style

- Python 3.9+ compatible
- No strict linting enforced, but keep it readable
- Include docstrings for public functions

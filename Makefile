.PHONY: setup serve test-serve eval eval-custom perf validate ablation clean

MODEL ?= llama3.1:8b
BASE_URL ?= http://localhost:11434
SEED ?= 42

# ─── Setup ───────────────────────────────────────────────────────────
setup:
	@echo "==> Creating virtual environment..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "==> Pulling Ollama model $(MODEL)..."
	ollama pull $(MODEL)
	@echo "==> Setup complete. Run: source .venv/bin/activate"

# ─── Serving ─────────────────────────────────────────────────────────
serve:
	python serve/serve.py --model $(MODEL)

test-serve:
	python serve/client.py --base-url $(BASE_URL)

# ─── Evaluation ──────────────────────────────────────────────────────
eval:
	python eval_runner/run_eval.py \
		--model $(MODEL) \
		--base-url $(BASE_URL) \
		--tasks hellaswag_generative,mmlu_all_generative \
		--limit 50 \
		--include-path eval_runner/custom_task \
		--output-dir eval_runner/results

eval-custom:
	python eval_runner/run_eval.py \
		--model $(MODEL) \
		--base-url $(BASE_URL) \
		--tasks custom_bench \
		--include-path eval_runner/custom_task \
		--output-dir eval_runner/results

eval-all: eval eval-custom

# ─── Performance ─────────────────────────────────────────────────────
perf:
	python perf/load_test.py \
		--model $(MODEL) \
		--base-url $(BASE_URL) \
		--output perf/metrics.csv

# ─── Validation ──────────────────────────────────────────────────────
validate:
	python validate/validate.py \
		--model $(MODEL) \
		--base-url $(BASE_URL) \
		--seed $(SEED)

# ─── Ablation ────────────────────────────────────────────────────────
ablation-prepare:
	python ablation/prepare_data.py

ablation-optimize:
	python ablation/optimize_prompt.py --model $(MODEL) --base-url $(BASE_URL)

ablation-eval:
	bash ablation/eval.sh

ablation: ablation-prepare ablation-optimize ablation-eval

# ─── Housekeeping ─────────────────────────────────────────────────────
clean:
	rm -rf .venv __pycache__ */__pycache__ */*/__pycache__
	rm -rf eval_runner/results/*.json perf/metrics.csv

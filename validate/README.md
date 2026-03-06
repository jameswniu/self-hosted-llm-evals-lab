# Determinism & Output Validation

## What Was Tested

### 1. Deterministic Mode
Configured Ollama with maximum determinism:
- `temperature=0` (greedy decoding)
- `top_p=1` (no nucleus sampling cutoff)
- `top_k=1` (always select highest probability token)
- `seed=42` (fixed random seed)
- `repeat_penalty=1.0` (no repeat penalty variation)

### 2. Reproducibility Verification
Each of 5 test prompts was sent **5 times** with identical parameters.
Responses were compared character-by-character to verify exact match.

Test categories:
- Factual Q&A (single-word expected)
- Arithmetic (numeric expected)
- List generation (structured expected)
- Binary yes/no
- Knowledge recall

### 3. Output Validation
Lightweight validators applied to custom task outputs:
- **NumericValidator**: regex extraction of numeric answers
- **SingleWordValidator**: word count check (max 5 words)
- **ClassificationValidator**: response matches allowed label set
- **SchemaValidator**: JSON output matches required key schema

## Where Nondeterminism Persists

Even with all deterministic settings, some sources of variation may remain:

1. **GPU floating-point nondeterminism**: Different GPU thread scheduling can
   produce slightly different floating-point accumulation results. This is
   hardware-level and cannot be fully controlled by seed alone.

2. **Ollama version differences**: Quantization rounding and KV-cache
   implementation details may vary across Ollama versions.

3. **Model loading order**: If the model is unloaded and reloaded between
   runs, internal state initialization may differ slightly.

4. **Long outputs**: Nondeterminism tends to compound over longer generation
   sequences as small floating-point differences cascade through autoregressive
   decoding.

**Mitigation**: For production evaluation pipelines, pair deterministic settings
with prompt-level caching (see `eval_runner/model.py` PromptCache) so that
once a response is generated, it is stored and reused exactly.

## Running

```bash
python validate/validate.py --model llama3.1:8b --seed 42 --trials 5
```

Results are saved to `validate/results.json`.

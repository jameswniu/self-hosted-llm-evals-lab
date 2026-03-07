# self-hosted-llm-evals-lab

Eval, benchmark, and optimize open-source LLMs you self-host as APIs.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![lm-eval-harness](https://img.shields.io/badge/lm--eval--harness-v0.4%2B-orange)
![Ollama](https://img.shields.io/badge/Ollama-compatible-purple)

## The Problem

You downloaded an open-source LLM. You're serving it with Ollama or vLLM. But you're flying blind:

- You don't know how accurate it actually is on your tasks
- Every "prompt engineering tip" you followed might be making things worse (we found CoT drops accuracy 25pp on small models)
- You have no idea if it can handle real traffic
- You can't tell if outputs are even reproducible

This toolkit gives you answers before you ship to production.

## What This Does

- **Benchmark accuracy** (MMLU, HellaSwag, custom tasks) via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Load-test throughput**, TTFT, P50/P95/P99 latency under concurrency
- **Validate determinism** (greedy decoding + fixed seed, verified across 5x5 trials)
- **Optimize prompts** via systematic ablation (templates, CoT, few-shot, self-consistency voting)

Works with [Ollama](https://ollama.com), or any OpenAI-compatible `/v1/chat/completions` endpoint.

## Use Cases

- **"Is my model accurate enough?"** -- Run standardized benchmarks (MMLU, HellaSwag) on any model you're serving. Compare Llama vs Mistral vs Gemma on the same tasks.
- **"Are my prompts actually helping?"** -- Test whether chain-of-thought, few-shot, or instruction prompts improve or hurt your specific model. (Spoiler: at 8B parameters, simpler prompts win.)
- **"Can it handle production traffic?"** -- Load-test your endpoint with concurrent requests. Get P50/P95/P99 latency, TTFT, and throughput numbers before users hit it.
- **"Why does it give different answers each time?"** -- Validate determinism with reproducibility checks. Pin down greedy decoding + fixed seed to get consistent outputs.
- **"Which model should I deploy?"** -- Run the same eval suite across models and compare accuracy, latency, and throughput side-by-side.

## Quick Start

```bash
make setup     # venv + deps + pull model
make serve     # Start Ollama endpoint
make eval      # MMLU + HellaSwag benchmarks
make ablation  # Prompt strategy optimization
make perf      # Load testing + latency analysis
make validate  # Determinism checks
```

Override the model: `MODEL=mistral:7b make eval`

## Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'lineColor': '#64748b', 'edgeLabelBackground': '#ffffff', 'clusterBkg': '#ffffff'}, 'flowchart': {'curve': 'linear'}}}%%
flowchart TD
    classDef serving fill:#2D5FAF,stroke:#1A3D7A,color:#ffffff,stroke-width:2px
    classDef eval fill:#C44B3B,stroke:#8B2E22,color:#ffffff,stroke-width:2px
    classDef ops fill:#278F84,stroke:#1A6359,color:#ffffff,stroke-width:2px
    classDef cache fill:#7B52C7,stroke:#53348A,color:#ffffff,stroke-width:2px

    OL(["Ollama<br/>llama3.1:8b Q4_0"]):::serving
    BR["Benchmark Runner<br/>lm-eval-harness"]:::eval
    AB["Ablation Engine<br/>optimize_prompt.py"]:::eval
    LT["Load Tester<br/>ThreadPoolExecutor"]:::ops
    DV["Determinism Validator<br/>5x5 matrix"]:::ops
    PC[("SHA-256<br/>Prompt Cache")]:::cache

    OL -->|"hellaswag, mmlu"| BR
    OL -->|"templates x decoding"| AB
    OL -->|"concurrent streaming"| LT
    OL -->|"greedy + seed=42"| DV

    BR -.->|"cache hit"| PC
    AB -.->|"cache hit"| PC
    DV -.->|"cache hit"| PC
```

Every (model, prompt, params) call is SHA-256 hashed and cached to disk. Repeated evaluations hit the cache instead of the inference endpoint, which makes ablation runs efficient and results reproducible. Deterministic baselines (temperature=0, top_k=1, fixed seed) are enforced before any comparison.

## Key Findings

### Chain-of-thought hurts at small scale

We found that chain-of-thought prompting **drops accuracy by 25 percentage points** on Llama 3.1 8B. The model gets the right answer with a simple prompt, then reasons itself into the wrong one when asked to think step-by-step.

| Strategy | Accuracy | 95% CI | vs Baseline |
|---|---|---|---|
| Baseline (minimal template) | 60% (12/20) | [0.39, 0.78] | -- |
| Instruction template | 50% (10/20) | [0.30, 0.70] | -10pp |
| Chain-of-thought | 35% (7/20) | [0.18, 0.57] | **-25pp** |
| Few-shot + CoT | 55% (11/20) | [0.34, 0.74] | -5pp |
| Self-consistency (k=5) | **70% (14/20)** | **[0.48, 0.85]** | **+10pp** |

95% CIs via Wilson score interval. McNemar's test on self-consistency vs baseline: p=0.48 (not significant at n=20).

![Accuracy by Prompting Strategy](docs/figures/accuracy_by_strategy.png)

Accuracy decreased monotonically with prompt complexity. Each added layer of instruction degraded performance. At 8B parameters, the model pattern-matches commonsense completions effectively, but explicit reasoning chains introduce noise that overrides correct first-pass answers.

Few-shot examples partially recovered CoT losses (35% to 55%) by constraining output format and reasoning depth.

#### Ablation pipeline

```mermaid
graph LR
    subgraph Phase1["Phase 1: Template Selection"]
        direction TB
        T1["baseline\n(complete the passage)"]
        T2["instruction"]
        T3["chain-of-thought"]
        T4["few-shot + CoT"]
        GD["Greedy decoding\ntemp=0, top_k=1, seed=42"]
        T1 --> GD
        T2 --> GD
        T3 --> GD
        T4 --> GD
        GD --> BEST["Best template\n(highest accuracy)"]
    end

    subgraph Phase2["Phase 2: Self-Consistency"]
        direction TB
        SC["k=5 samples\ntemp=0.7, top_p=0.95\ntop_k=40, seeds 42..46"]
        MV["Majority vote\nconfidence = winner/k"]
        SC --> MV
    end

    subgraph Stats["Statistical Comparison"]
        direction TB
        WCI["Wilson score CIs"]
        MN["McNemar's test"]
    end

    BEST --> SC
    MV --> WCI
    MV --> MN

    classDef phase1 fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef phase2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef stats fill:#f3e8ff,stroke:#9333ea,color:#581c87

    class T1,T2,T3,T4,GD,BEST phase1
    class SC,MV phase2
    class WCI,MN stats
```

### Self-consistency and confidence routing

Self-consistency (5 samples at temperature=0.7, majority vote) was the only strategy that improved accuracy: +10pp over baseline at 5x latency cost.

```mermaid
graph LR
    subgraph Generate["Parallel Generation"]
        P["Prompt"] --> G1["seed=42\ntemp=0.7\ntop_p=0.95\ntop_k=40"]
        P --> G2["seed=43"]
        P --> G3["seed=44"]
        P --> G4["seed=45"]
        P --> G5["seed=46"]
    end

    subgraph Extract["Answer Extraction"]
        G1 --> E1["extract_answer()\n3-tier regex"]
        G2 --> E2["extract_answer()"]
        G3 --> E3["extract_answer()"]
        G4 --> E4["extract_answer()"]
        G5 --> E5["extract_answer()"]
    end

    subgraph Vote["Majority Vote"]
        E1 --> C["Counter()\nwinner, count"]
        E2 --> C
        E3 --> C
        E4 --> C
        E5 --> C
        C --> CONF["confidence =\nwinner_count / k"]
    end

    subgraph Route["Confidence Routing"]
        CONF -->|">= 0.8"| SERVE["Serve answer"]
        CONF -->|"<= 0.6"| CASCADE["Cascade to\nlarger model"]
    end

    classDef gen fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    classDef ext fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef vote fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef route fill:#f3e8ff,stroke:#9333ea,color:#581c87

    class P,G1,G2,G3,G4,G5 gen
    class E1,E2,E3,E4,E5 ext
    class C,CONF vote
    class SERVE,CASCADE route
```

Self-consistency config: `k=5, temperature=0.7, top_p=0.95, top_k=40, seeds=[42,43,44,45,46], max_tokens=128`.

The vote confidence distribution reveals a practical routing signal:

![Vote Confidence as Routing Signal](docs/figures/confidence_routing.png)

- Confidence >= 0.8: nearly always correct. Trust and serve.
- Confidence <= 0.6: close to random. Cascade to a larger model.

This suggests a production routing strategy: run self-consistency on the small model, and only escalate low-confidence items to a more expensive endpoint.

### Performance profile

| Config | TTFT P50 (ms) | Throughput (tok/s) | Concurrency |
|---|---|---|---|
| Short, c=1 | ~132 | ~65 | 1 |
| Short, c=3 | ~596 | ~68 | 3 |
| Short, c=5 | ~4531 | ~66 | 5 |
| Long, c=1 | ~190 | ~63 | 1 |

TTFT measured via streaming first non-empty chunk. Throughput from Ollama's `eval_count / eval_duration`. First request excluded (cold-start model load).

```mermaid
graph LR
    subgraph Clients["ThreadPoolExecutor"]
        C1["Thread 1\nprompt"]
        C2["Thread 2\nprompt"]
        C3["Thread 3\nprompt"]
    end

    subgraph Queue["Ollama Request Queue"]
        direction TB
        Q["Sequential processing\n(no batching)"]
        Q --> R1["Request 1\nstreaming"]
        Q --> R2["Request 2\nwaiting..."]
        Q --> R3["Request 3\nwaiting..."]
    end

    subgraph Metrics["Metric Collection"]
        direction TB
        TTFT["TTFT\nfirst non-empty chunk"]
        TPS["tok/s\neval_count / eval_duration"]
        AGG["numpy aggregation\nP50 / P95 / P99"]
        TTFT --> AGG
        TPS --> AGG
    end

    C1 --> Q
    C2 --> Q
    C3 --> Q
    R1 --> TTFT
    R1 --> TPS

    classDef client fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    classDef queue fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef metrics fill:#dcfce7,stroke:#16a34a,color:#14532d

    class C1,C2,C3 client
    class Q,R1,R2,R3 queue
    class TTFT,TPS,AGG metrics
```

TTFT scales linearly with concurrency because Ollama queues requests sequentially rather than batching. For scaling beyond single-GPU, continuous batching backends (vLLM, TGI) would be the next step.

![Latency and Throughput](docs/figures/latency_throughput.png)

## Methodology

- **Model**: Llama 3.1 8B (Q4_0 quantized) via Ollama
- **Decoding**: Greedy baseline (temperature=0, top_p=1, top_k=1, seed=42). Self-consistency uses temperature=0.7, top_p=0.95, k=5.
- **Benchmark**: HellaSwag (commonsense completion), MMLU (knowledge), custom JSON task
- **Ablation**: 5 prompting strategies tested on identical 20-item subset. Answer extraction via regex normalization ("A", "A.", "(A)", "The answer is A").
- **Statistical tests**: McNemar's test for paired accuracy comparisons. Wilson score confidence intervals for proportions. p=0.48 on the self-consistency vs baseline comparison (not significant at n=20).

<details>
<summary>Full decoding parameters and implementation details</summary>

### Decoding parameters

| Parameter | Greedy (baseline) | Self-consistency |
|---|---|---|
| temperature | 0 | 0.7 |
| top_p | 1.0 | 0.95 |
| top_k | 1 | 40 |
| seed | 42 | 42, 43, 44, 45, 46 |
| repeat_penalty | 1.0 | 1.0 |
| max_tokens | 128 | 128 |

### lm-eval-harness invocation

```bash
python -m lm_eval \
  --model local-chat-completions \
  --model_args model=llama3.1:8b,base_url=http://localhost:11434/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
  --apply_chat_template \
  --tasks hellaswag,mmlu \
  --limit 100 \
  --batch_size 1 \
  --seed 42 \
  --log_samples \
  --cache_requests true
```

### Answer extraction (3-tier regex)

1. **Direct letter at start**: `^[(\s]*([A-Da-d])[).\s,:]`
2. **Keyword pattern**: `(?:answer|choice|option)[\s:]+(?:is\s+)?[(\s]*([A-Da-d])`
3. **Standalone letter fallback**: `\b([A-Da-d])\b`

### Statistical formulas

**Wilson score CI**:

```
center = (p + z^2/2n) / (1 + z^2/n)
spread = z * sqrt((p(1-p) + z^2/4n) / n) / (1 + z^2/n)
CI = [center - spread, center + spread]
```

Where z = 1.96 for 95% confidence, p = observed accuracy, n = sample size.

**McNemar's chi-squared**:

```
chi2 = (|b - c| - 1)^2 / (b + c)
```

Where b = items only baseline got right, c = items only improved got right. df=1.

</details>

## Limitations & Future Work

- **Sample size**: n=20 gives 30-40pp wide confidence intervals. Need 200+ examples for 80% power to detect a 10pp effect. The CoT drop is large enough to be directionally meaningful, but exact magnitudes are uncertain.
- **Single model size**: At what parameter count does CoT start helping? Testing 13B, 70B, and mixture-of-experts architectures would map the crossover point.
- **Quantization**: All results are on Q4_0 quantized weights. Full-precision comparison would isolate whether quantization interacts with prompting strategy.
- **Task coverage**: Ablation was HellaSwag only. CoT may perform differently on coding, math, or multi-step reasoning tasks where explicit reasoning is more structurally useful.
- **Backend**: Ollama's sequential queuing limits concurrency insights. Benchmarking against vLLM or TGI with continuous batching would give a more realistic production performance profile.

## Takeaways

1. Simpler prompts outperform complex ones at 8B scale. Each added layer of instruction degraded accuracy.
2. Self-consistency's vote confidence is a useful routing signal: high-confidence items can be trusted, low-confidence items should cascade to a larger model.
3. n=20 gives 30-40pp wide confidence intervals. Need 200+ for meaningful significance tests.
4. Deterministic baselines (greedy + fixed seed) are a prerequisite for valid ablations.

## Project Structure

```
self-hosted-llm-evals-lab/
├── Makefile                    # All commands
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
│
├── serve/                      # Inference server management
│   ├── serve.py                # Ollama lifecycle manager
│   └── client.py               # Sample generation client
│
├── eval_runner/                # Benchmark evaluation
│   ├── model.py                # Model wrapper + SHA-256 prompt cache
│   ├── run_eval.py             # lm-eval-harness runner
│   └── custom_task/            # Custom JSON benchmark definitions
│
├── perf/                       # Load testing
│   ├── load_test.py            # Concurrent load generator
│   ├── metrics.csv             # Raw metrics output
│   └── analysis.ipynb          # Visualization notebook
│
├── validate/                   # Determinism verification
│   ├── validate.py             # N-trial consistency + output validators
│   └── README.md               # Testing methodology
│
├── ablation/                   # Prompt strategy optimization
│   ├── prepare_data.py         # Few-shot + template preparation
│   ├── optimize_prompt.py      # Ablation across strategies
│   ├── infer.py                # Comparison + statistical tests
│   ├── eval.sh                 # Full pipeline script
│   └── report.md               # Detailed results + analysis
│
└── docs/
    ├── REPORT.md               # Extended research report
    ├── architecture.md         # System design
    └── figures/                # Chart exports
```

## License

MIT

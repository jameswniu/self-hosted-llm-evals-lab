# Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'lineColor': '#64748b', 'edgeLabelBackground': '#ffffff', 'clusterBkg': '#ffffff'}, 'flowchart': {'curve': 'linear'}}}%%
graph TD
    classDef serving fill:#2D5FAF,stroke:#1A3D7A,color:#ffffff,stroke-width:2px
    classDef eval fill:#C44B3B,stroke:#8B2E22,color:#ffffff,stroke-width:2px
    classDef ops fill:#278F84,stroke:#1A6359,color:#ffffff,stroke-width:2px
    classDef cache fill:#7B52C7,stroke:#53348A,color:#ffffff,stroke-width:2px
    classDef result fill:#1E3A5F,stroke:#142740,color:#ffffff,stroke-width:2px

    A["Ollama / OpenAI-compatible endpoint"]:::serving --> B["Benchmark Runner"]:::eval
    A --> C["Prompt Ablation Engine"]:::eval
    A --> D["Load Tester"]:::ops
    A --> E["Determinism Validator"]:::ops

    B -->|"lm-eval-harness"| F["MMLU, HellaSwag, Custom Tasks"]:::result
    C -->|"templates x decoding"| G["Accuracy by Strategy"]:::result
    D -->|"concurrent requests"| H["TTFT, P50/P95/P99, tok/s"]:::result
    E -->|"greedy + fixed seed"| I["5x5 Reproducibility Matrix"]:::result

    B -.-> J[("SHA-256 Prompt Cache")]:::cache
    C -.-> J
    E -.-> J
```

## Components

### Benchmark Runner (`eval_runner/`)
Wraps EleutherAI's lm-evaluation-harness with a custom model adapter. Supports MMLU, HellaSwag, and custom JSON-based benchmarks. Results cached via SHA-256 hashed (model, prompt, params) keys.

### Prompt Ablation Engine (`ablation/`)
Systematic comparison of prompting strategies: baseline, instruction template, chain-of-thought, few-shot + CoT, and self-consistency voting. Measures accuracy deltas and statistical significance (McNemar's test, Wilson confidence intervals).

### Load Tester (`perf/`)
Concurrent request generator measuring TTFT, total latency (P50/P95/P99), and throughput (tok/s). Tests across prompt lengths, concurrency levels, and stop-sequence configurations.

### Determinism Validator (`validate/`)
Verifies reproducibility under greedy decoding (temperature=0, top_k=1, fixed seed). Runs N-trial consistency checks and applies output validators (regex, schema, classification).

# LinkedIn Post Draft

## Main Post

I told an 8B model to "think step by step" and its accuracy dropped by 25 percentage points.

This is why you need evals before deploying. Most people self-host an open-source LLM, follow a few prompt engineering tips, and ship -- with no idea whether their model is accurate, whether their prompts help or hurt, or whether the endpoint can handle real traffic.

I built an eval toolkit for self-hosted LLMs and ran a systematic ablation across 5 prompting strategies on Llama 3.1 8B:

| Strategy | Accuracy |
|---|---|
| Simple prompt | 60% |
| Instruction template | 50% |
| Chain-of-thought | 35% |
| Few-shot + CoT | 55% |
| Self-consistency (k=5) | 70% |

The model gets commonsense questions right with a minimal prompt. But when you ask it to reason step-by-step, it talks itself into the wrong answer. At 8B parameters, CoT adds noise, not signal.

The one thing that worked: self-consistency. Generate 5 answers at higher temperature, take the majority vote. +10pp over baseline. And the vote confidence cleanly separates reliable predictions (>=0.8 confidence, nearly always correct) from unreliable ones (<=0.6, basically random).

That confidence score is a practical routing signal. Trust high-confidence answers from the small model. Cascade low-confidence items to a larger one.

The practical takeaway for anyone serving open-source models as APIs: eval before you prompt-engineer. Simpler prompts can outperform complex ones at small scale. And without running evals, you're just guessing.

Repo: [link]

Built as an eval toolkit -- if you're self-hosting LLMs, you can use this to benchmark, eval, and optimize your setup out of the box.

#LLM #OpenSource #PromptEngineering #AIEngineering #MachineLearning #Evals

---

## First Comment (Engineering Depth)

Some engineering details for the curious:

- SHA-256 prompt caching: every (model, prompt, params) tuple is hashed and cached to disk, so ablation runs over the same data are instant and perfectly reproducible
- Deterministic baselines: greedy decoding + fixed seed, verified identical across 5x5 trials before running any comparison
- Load testing with streaming TTFT measurement, percentile latency (P50/P95/P99), throughput scaling under concurrency
- Statistical rigor: McNemar's test for paired comparisons, Wilson score confidence intervals. The self-consistency gain isn't statistically significant at n=20 (p=0.48) -- need 200+ samples for 80% power

Built with lm-eval-harness, Ollama, and Python. Works with any OpenAI-compatible endpoint.

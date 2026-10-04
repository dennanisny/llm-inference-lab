# Costs

Every GPU boot gets a row here. Per the plan's GPU discipline rule, the row is written
**before** the machine starts, and the `Figure` column names the file that session will
produce — if it can't be named, the session doesn't boot.

Budget for the 24 weeks: **$300–500** across roughly **20 hours** of machine time.
Multiple GPUs are only needed in weeks 18–19.

| Date | Week / Session | GPU | Rate | Hours | Cost | Figure |
|---|---|---|---|---|---|---|
| 2026-09-06 | W0 · Session 1 | RunPod A100 SXM | $1.59/hr | 0.46 | $0.73 | — (smoke test, no figure by design) |

**Spent to date: $0.73 of $300–500. Machine time to date: 0.46 of ~20 hours.**

## Notes on the rows

- **W0 · Session 1** — bring-up only: `vllm serve Qwen/Qwen2.5-7B-Instruct`, verify
  `/v1/models` and `/health`, send one chat completion, shut down. No figure was expected;
  the deliverable is the served endpoint itself. Full record in
  [`experiments/week00-vllm-serve-qwen/`](experiments/week00-vllm-serve-qwen/).
  The 0.46 GPU-hours is the $0.73 charge at the $1.59/hr A100 SXM rate.

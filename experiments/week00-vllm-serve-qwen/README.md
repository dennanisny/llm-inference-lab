# Week 0 — Use vLLM to serve Qwen model

**Question:** Can vLLM serve Qwen2.5-7B-Instruct on a single rented A100 and return a valid chat completion end to end?

---

## Hypothesis

I expect to use vLLM to serve a LLM model on a pod managed by RunPod, and get a valid response for text generation.

---

## Setup

| | |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Hardware | RunPod A100 SXM |
| Engine | vLLM 0.28.0 (`vllm-0.28.0-de1ecaf1`) |
| Workload | `chat` — one single-turn request, no concurrency |
| SLO | Not measured this week. TTFT / TPOT targets are set once the load-test harness lands (Session 3) |
| Varied | Nothing — smoke test at a single fixed configuration |
| Held fixed | vLLM server defaults (single GPU, `max_model_len` 32768) |

---

## Result

No figure this week — the artifact of a smoke test is the response itself, not a curve.

Observed from the run below: the OpenAI-compatible endpoint came up and listed
`Qwen/Qwen2.5-7B-Instruct` with `max_model_len` 32768; a single chat request returned
`finish_reason: "stop"` with 33 prompt tokens and 49 completion tokens (82 total).

<!-- TODO: no latency numbers were captured this session. First TTFT / TPOT
     measurements come from the Session 3 load-test script; add result.png then. -->

**Against the hypothesis:** Matched. vLLM served the model on a RunPod A100 and returned a
valid generation on the first request, with no fallback or degraded path involved. Nothing
was learned about latency or throughput — that was not what this week tested.

---

## Reproduce

Start serving the model

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Validate the expected model is present

```bash
curl localhost:8000/v1/models
{"object":"list","data":[{"id":"Qwen/Qwen2.5-7B-Instruct","object":"model","created":1788735762,"owned_by":"vllm","root":"Qwen/Qwen2.5-7B-Instruct","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-ba83413358dd8c4d","object":"model_permission","created":1788735762,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

Validate the service is healthy

```bash
until curl -sf localhost:8000/health; do sleep 5; done && echo READY
READY
```

Text generation

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Who are you?"}]}'

{
  "id": "chatcmpl-b76a63d231831106",
  "object": "chat.completion",
  "created": 1788735780,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I am Qwen, a large language model created by Alibaba Cloud. I'm here to assist with a wide variety of tasks and answer any questions you might have to the best of my knowledge and ability. How can I help you today?",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "reasoning": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null,
      "routed_experts": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": "vllm-0.28.0-de1ecaf1",
  "usage": {
    "prompt_tokens": 33,
    "total_tokens": 82,
    "completion_tokens": 49,
    "prompt_tokens_details": null,
    "completion_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "prompt_text": null,
  "kv_transfer_params": null,
  "ec_transfer_params": null,
  "metrics": null
}
```

### Notes

- First start downloads ~15GB of weights; 3–10 minutes of apparently-stuck logs is normal
  (weight load plus CUDA graph capture). Poll `/health` instead of watching the log.
- Run all of the commands above **on the GPU host itself**. Reaching the server from a laptop
  requires exposing port 8000 in the RunPod console and using the public proxy address rather
  than `localhost`.

---

## Cost

Approximately `$0.73` on `A100 SXM` at `$1.59/hr` — about **0.46 GPU-hours** (~28 minutes).

<!-- TODO: confirm actual GPU-hours against the RunPod billing page; 0.46 is back-computed
     from the $0.73 / $1.59-per-hour figures, not read off the invoice. -->

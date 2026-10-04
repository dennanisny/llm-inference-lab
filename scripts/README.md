# scripts

Load-testing harness. Built and validated **locally against a mock**, per the plan's GPU
discipline rule: nothing here needs a GPU to debug.

| File | What it is |
|---|---|
| `profiles.py` | Workload profiles. Defined once, imported everywhere, not retuned per experiment |
| `mock_server.py` | OpenAI-compatible streaming mock with fixed, known timings |
| `loadgen.py` | N concurrent streaming requests; records TTFT and TPOT per request |

Install: `pip install -r requirements.txt` (aiohttp only).

## Validate locally — no GPU

```bash
python3 scripts/mock_server.py --port 8001 &
python3 scripts/loadgen.py --base-url http://localhost:8001 --model mock-model \
    --concurrency 1,10 --rounds 3 --assert-ttft-ms 80 --assert-tpot-ms 20
```

The mock's delays are fixed, so the expected answer is known in advance: TTFT 80 ms,
TPOT 20 ms, **at every concurrency level**. If TTFT grows with concurrency the requests
are being serialized — a harness bug, and one that would otherwise be discovered at
$1.59/hr. Exit code is non-zero when an assertion fails, so this works in CI or a
pre-boot check.

## Real run — on the GPU host

```bash
until curl -sf localhost:8000/health; do sleep 5; done && echo READY
python3 scripts/loadgen.py --calibrate                     # once, per model
python3 scripts/loadgen.py --concurrency 1,4,16 --rounds 8 --csv results.csv
```

Run this **on the GPU machine itself**. Driving it from a laptop through the RunPod proxy
puts WAN round-trip inside the TTFT measurement with no way to subtract it afterward.

## Measurement definitions

- **TTFT** — first chunk with *non-empty content*, minus request send. The role-only chunk
  vLLM emits first is deliberately skipped; counting it reports single-digit milliseconds
  regardless of prefill cost.
- **TPOT** — (last content chunk − first content chunk) / (output_tokens − 1). Decode only;
  prefill is excluded by construction. `nan` when fewer than 2 output tokens.

Output length is pinned with `max_tokens` + `ignore_eos` so every request emits exactly
`profile.output_tokens`. Without that, TPOT spread measures response-length variance
instead of engine behaviour.

## Two flags that change what you are measuring

- `--unique-prefix` prepends a random string per request to **defeat prefix caching**.
  Prefix caching is on by default in vLLM V1, so identical prompts mean request 1 fills
  the cache and the rest hit it — a ~100% hit-rate number, not a cold baseline. Either
  flag is defensible; record which one the run used. The prefix is itself tokens, so
  re-run `--calibrate` with the flag on.
- `--no-ignore-eos` lets the model stop early. Only for checking natural output lengths,
  never for a TPOT comparison.

`loadgen.py` warns when observed `prompt_tokens` / `output_tokens` drift from the profile,
and when the sample is too small for the p95/p99 it just printed (under 100 requests).

#!/usr/bin/env python3
"""Load generator: N concurrent streaming requests, records TTFT and TPOT per request.

Definitions used here, so later experiments can't quietly drift:

  TTFT = (first chunk carrying NON-EMPTY content) - (request send)
         The role-only chunk that vLLM sends first is skipped on purpose; counting it
         would report a TTFT of a few milliseconds regardless of prefill cost.
  TPOT = (last content chunk - first content chunk) / (output_tokens - 1)
         Inter-token time across the decode phase only; prefill is excluded by
         construction. Undefined (nan) when output_tokens < 2.

Output length is pinned with max_tokens + ignore_eos so every request emits exactly
profile.output_tokens tokens. Without that, TPOT spread measures response-length
variance rather than engine behaviour.

Local validation against the mock (no GPU):
    python3 scripts/mock_server.py --port 8001 &
    python3 scripts/loadgen.py --base-url http://localhost:8001 --model mock-model \
        --concurrency 10 --assert-ttft-ms 80 --assert-tpot-ms 20

Real run — ON THE GPU HOST, not over a proxy from a laptop:
    python3 scripts/loadgen.py --concurrency 1,4,16 --rounds 8 --csv results.csv
"""

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, asdict

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profiles  # noqa: E402

FILLER_WORD = "token"
PERCENTILE_MIN_SAMPLES = 100


@dataclass
class Result:
    concurrency: int
    round_idx: int
    ok: bool
    status: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    prompt_tokens: int
    output_tokens: int
    error: str = ""


def pct(xs, p: float) -> float:
    """Nearest-rank percentile. Returns nan on an empty sample."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, math.ceil(p / 100.0 * len(s)) - 1)
    return s[k]


def build_prompt(filler_words: int, unique_prefix: bool) -> str:
    """Fixed prompt of `filler_words` words, optionally with a unique prefix.

    The unique prefix exists to defeat prefix caching: with an identical prompt on every
    request, request 1 populates the cache and the rest hit it, so TTFT reflects a ~100%
    hit rate rather than a cold path. Pass --unique-prefix for a cold baseline, or leave
    it off and label the run as a cache-hit measurement — but say which one you did.

    Caveat: the prefix is itself tokens. A 32-char hex string is one whitespace word but
    several tokens to a real tokenizer, so turning this on inflates prompt_tokens above
    the profile target. Re-run --calibrate WITH --unique-prefix and use the filler count
    it reports, or the cold and warm runs won't be comparing the same input length.
    """
    head = f"{uuid.uuid4().hex} " if unique_prefix else ""
    return head + " ".join([FILLER_WORD] * max(1, filler_words))


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    concurrency: int,
    round_idx: int,
) -> Result:
    blank = dict(
        concurrency=concurrency,
        round_idx=round_idx,
        ok=False,
        status=0,
        ttft_ms=float("nan"),
        tpot_ms=float("nan"),
        total_ms=float("nan"),
        prompt_tokens=0,
        output_tokens=0,
    )
    t0 = time.perf_counter()
    first_content = None
    last_content = None
    chunks = 0
    usage = {}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = (await resp.text())[:200]
                return Result(**{**blank, "status": resp.status, "error": text})
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content") or ""
                    if not content:
                        continue  # role chunk / empty keepalive — not a token
                    now = time.perf_counter()
                    if first_content is None:
                        first_content = now
                    last_content = now
                    chunks += 1
            total = time.perf_counter() - t0
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return Result(**{**blank, "error": f"{type(exc).__name__}: {exc}"})

    if first_content is None:
        return Result(**{**blank, "status": 200, "error": "no content chunks received"})

    out_tokens = int(usage.get("completion_tokens") or chunks)
    if out_tokens > 1 and last_content > first_content:
        tpot = (last_content - first_content) / (out_tokens - 1) * 1000.0
    else:
        tpot = float("nan")

    return Result(
        concurrency=concurrency,
        round_idx=round_idx,
        ok=True,
        status=200,
        ttft_ms=(first_content - t0) * 1000.0,
        tpot_ms=tpot,
        total_ms=total * 1000.0,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=out_tokens,
    )


def make_payload(args, profile, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": profile.output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not args.no_ignore_eos:
        payload["ignore_eos"] = True
    return payload


async def run_level(session, url, args, profile, concurrency: int) -> list[Result]:
    out: list[Result] = []
    for r in range(args.rounds):
        payloads = [
            make_payload(args, profile, build_prompt(args.filler_words, args.unique_prefix))
            for _ in range(concurrency)
        ]
        out.extend(await asyncio.gather(*(one_request(session, url, p, concurrency, r) for p in payloads)))
    return out


async def calibrate(session, url, args, profile) -> None:
    """Find the filler-word count whose server-reported prompt_tokens hits the target.

    Avoids depending on a local tokenizer: the server is the authority on token counts.
    """
    target = profile.input_tokens
    print(f"calibrating filler words against prompt_tokens target {target} ...")
    seen: dict[int, int] = {}
    lo, hi = 1, max(8, target * 2)
    best = None
    for _ in range(12):
        mid = (lo + hi) // 2
        payload = make_payload(args, profile, build_prompt(mid, False))
        payload["max_tokens"] = 1
        res = await one_request(session, url, payload, 1, 0)
        if not res.ok or not res.prompt_tokens:
            print(f"  filler={mid:4d} -> no usage reported ({res.error or 'none'}); cannot calibrate")
            return
        seen[mid] = res.prompt_tokens
        print(f"  filler={mid:4d} -> prompt_tokens={res.prompt_tokens}")
        if best is None or abs(res.prompt_tokens - target) < abs(seen[best] - target):
            best = mid
        if res.prompt_tokens == target:
            break
        if res.prompt_tokens < target:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    print(f"\nclosest: --filler-words {best}  (prompt_tokens={seen[best]}, target={target})")
    if seen[best] != target:
        print(
            f"note: exact {target} not reachable by whole filler words — the chat template "
            f"contributes a fixed overhead. Either accept {seen[best]} and record it, or "
            f"adjust the profile."
        )


def summarize(results: list[Result], label: str, profile=None) -> dict:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    ttft = [r.ttft_ms for r in ok]
    tpot = [r.tpot_ms for r in ok if not math.isnan(r.tpot_ms)]
    print(f"\n=== {label} ===")
    print(f"requests: {len(results)}  ok: {len(ok)}  failed: {failed}")
    if failed:
        for r in results:
            if not r.ok:
                print(f"  FAIL status={r.status} {r.error}")
    if not ok:
        return {}
    pt = {r.prompt_tokens for r in ok}
    ot = {r.output_tokens for r in ok}
    print(f"prompt_tokens: {sorted(pt)}   output_tokens: {sorted(ot)}")
    if profile is not None:
        off = [n for n in pt if n and n != profile.input_tokens]
        if off:
            print(f"  WARNING: prompt_tokens {sorted(off)} != profile {profile.name} target "
                  f"{profile.input_tokens}. Re-run --calibrate (with --unique-prefix if you "
                  f"are using it) or this run is not on-profile.")
        if ot and profile.output_tokens not in ot:
            print(f"  WARNING: output_tokens {sorted(ot)} != profile target "
                  f"{profile.output_tokens}. Is ignore_eos supported by this server?")
    if len(ot) > 1:
        print("  WARNING: output length varies across requests — TPOT spread is partly "
              "length variance. Is ignore_eos being honoured?")
    for name, xs in (("TTFT", ttft), ("TPOT", tpot)):
        if not xs:
            continue
        print(
            f"{name:4s} ms   mean {statistics.fmean(xs):8.2f}   p50 {pct(xs,50):8.2f}   "
            f"p95 {pct(xs,95):8.2f}   p99 {pct(xs,99):8.2f}   max {max(xs):8.2f}"
        )
    if len(ok) < PERCENTILE_MIN_SAMPLES:
        print(
            f"NOTE: {len(ok)} samples — p95/p99 above are not meaningful tail estimates. "
            f"Raise --rounds toward {PERCENTILE_MIN_SAMPLES}+ before quoting a p99 anywhere."
        )
    return {"ttft": ttft, "tpot": tpot, "ok": len(ok)}


async def main_async(args) -> int:
    profile = profiles.get(args.profile)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conns = max(args.concurrency_levels) if args.concurrency_levels else 1

    print(f"profile {profile.name}: {profile.input_tokens} in / {profile.output_tokens} out  ({profile.note})")
    print(f"target {url}   model={args.model}")
    print(f"prefix caching: {'DEFEATED via unique prefix per request' if args.unique_prefix else 'NOT defeated — identical prompts, expect cache hits'}")

    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=conns + 4)
    ) as session:
        if args.calibrate:
            await calibrate(session, url, args, profile)
            return 0

        all_results: list[Result] = []
        summaries = {}
        for c in args.concurrency_levels:
            res = await run_level(session, url, args, profile, c)
            all_results.extend(res)
            summaries[c] = summarize(res, f"concurrency {c}  (rounds={args.rounds})", profile)

        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(asdict(all_results[0]).keys()))
                w.writeheader()
                for r in all_results:
                    w.writerow(asdict(r))
            print(f"\nwrote {len(all_results)} rows to {args.csv}")

        if args.assert_ttft_ms is not None or args.assert_tpot_ms is not None:
            return check_assertions(args, summaries)
    return 0


def check_assertions(args, summaries) -> int:
    print("\n=== assertions (mock validation) ===")
    bad = 0
    for c, s in summaries.items():
        if not s:
            continue
        for name, want, xs in (
            ("TTFT", args.assert_ttft_ms, s["ttft"]),
            ("TPOT", args.assert_tpot_ms, s["tpot"]),
        ):
            if want is None or not xs:
                continue
            got = statistics.fmean(xs)
            tol = want * args.assert_tol
            ok = abs(got - want) <= tol
            bad += 0 if ok else 1
            print(
                f"  c={c:3d} {name}: mean {got:7.2f} ms vs expected {want:7.2f} "
                f"(+/-{tol:.1f})  {'PASS' if ok else 'FAIL'}"
            )
    if bad:
        print(
            f"\n{bad} assertion(s) FAILED. Fix the harness locally before booting a GPU.\n"
            "A TTFT that scales with concurrency against the mock means your requests are "
            "serialized, not concurrent."
        )
    else:
        print("\nall assertions passed — harness timing math is sound.")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--profile", default="W1-chat")
    ap.add_argument("--concurrency", default="1", help="comma-separated levels, e.g. 1,4,16")
    ap.add_argument("--rounds", type=int, default=1, help="repeat each level this many times (more samples)")
    ap.add_argument("--filler-words", type=int, default=9, help="see --calibrate")
    ap.add_argument("--unique-prefix", action="store_true", help="defeat prefix caching (cold baseline)")
    ap.add_argument("--no-ignore-eos", action="store_true", help="let the model stop early (TPOT gets noisy)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--csv", default="")
    ap.add_argument("--calibrate", action="store_true", help="find --filler-words for the profile's input_tokens")
    ap.add_argument("--assert-ttft-ms", type=float, default=None)
    ap.add_argument("--assert-tpot-ms", type=float, default=None)
    ap.add_argument("--assert-tol", type=float, default=0.35, help="relative tolerance for assertions")
    args = ap.parse_args()
    args.concurrency_levels = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

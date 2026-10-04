#!/usr/bin/env python3
"""Minimal OpenAI-compatible streaming mock. No GPU, no model, no weights.

Why this exists: its timings are FIXED and KNOWN, so if loadgen.py measures anything
other than --ttft-ms and --tpot-ms, the bug is in loadgen.py. You cannot get that check
on a rented GPU, where you have no idea what the true answer is supposed to be.

It deliberately reproduces one piece of real vLLM behaviour: the first SSE chunk carries
`delta.role` with EMPTY content, and the first actual token arrives later. A harness that
starts its TTFT clock on the role chunk will report a far-too-low TTFT, so the mock makes
that mistake visible.

    python3 scripts/mock_server.py --port 8001
    python3 scripts/loadgen.py --base-url http://localhost:8001 --model mock-model \
        --concurrency 10 --assert-ttft-ms 80 --assert-tpot-ms 20
"""

import argparse
import asyncio
import json
import time
import uuid

from aiohttp import web

ROLE_CHUNK_DELAY_S = 0.005  # vLLM-like: role chunk lands well before the first token


def _count_prompt_tokens(messages) -> int:
    """Crude stand-in for a tokenizer: whitespace words + a fixed template overhead.

    Only ever used by the mock, so it just has to be deterministic — it is NOT a model
    of how Qwen tokenizes anything.
    """
    words = sum(len(str(m.get("content", "")).split()) for m in messages)
    return words + 24


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": request.app["cfg"].model,
                    "object": "model",
                    "owned_by": "mock",
                    "max_model_len": 32768,
                }
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="")


async def handle_chat(request: web.Request) -> web.StreamResponse:
    cfg = request.app["cfg"]
    body = await request.json()

    max_tokens = int(body.get("max_tokens") or 16)
    prompt_tokens = _count_prompt_tokens(body.get("messages", []))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    created = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"

    ttft_s = cfg.ttft_ms / 1000.0
    tpot_s = cfg.tpot_ms / 1000.0

    if not body.get("stream"):
        await asyncio.sleep(ttft_s + tpot_s * max(0, max_tokens - 1))
        return web.json_response(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": cfg.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "tok " * max_tokens},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_tokens + max_tokens,
                },
            }
        )

    resp = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    await resp.prepare(request)

    async def send(payload: dict) -> None:
        await resp.write(f"data: {json.dumps(payload)}\n\n".encode())

    def envelope(delta: dict, finish=None) -> dict:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": cfg.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    # Role chunk with no content — the decoy. TTFT must NOT be measured here.
    await asyncio.sleep(ROLE_CHUNK_DELAY_S)
    await send(envelope({"role": "assistant", "content": ""}))

    # First real token after the remaining TTFT, then one every TPOT.
    await asyncio.sleep(max(0.0, ttft_s - ROLE_CHUNK_DELAY_S))
    for i in range(max_tokens):
        if i:
            await asyncio.sleep(tpot_s)
        await send(envelope({"content": f"tok{i} "}))

    await send(envelope({}, finish="length"))
    if include_usage:
        await send(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": cfg.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_tokens + max_tokens,
                },
            }
        )
    await resp.write(b"data: [DONE]\n\n")
    return resp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--ttft-ms", type=float, default=80.0, help="delay before the first CONTENT chunk")
    ap.add_argument("--tpot-ms", type=float, default=20.0, help="delay between content chunks")
    cfg = ap.parse_args()

    app = web.Application()
    app["cfg"] = cfg
    app.add_routes(
        [
            web.get("/health", handle_health),
            web.get("/v1/models", handle_models),
            web.post("/v1/chat/completions", handle_chat),
        ]
    )
    print(f"mock: http://{cfg.host}:{cfg.port}  ttft={cfg.ttft_ms}ms tpot={cfg.tpot_ms}ms model={cfg.model}")
    web.run_app(app, host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()

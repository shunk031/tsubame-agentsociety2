#!/usr/bin/env python3
"""Sweep concurrency against a vLLM server and report where it stops helping.

Why: every throughput number this repository has is entangled with
agentsociety2 -- its routers, its env actor, its retries. None of them says
what the hardware can do, so there is no denominator to compare against. A run
that achieves 300 tokens a second means something very different if the ceiling
is 400 than if it is 4000.

Two regimes are measured separately because they are bound by different things.
Decode (short prompt, long output) rereads every weight per token and is bound
by memory bandwidth; on an H100 the arithmetic says roughly 300 concurrent
sequences are needed before the arithmetic units matter at all. Prefill (long
prompt, short output) processes a whole prompt at once and is compute-bound
from the start. Reporting one number for both hides which one a workload is in.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request


def request_once(
    endpoint: str, model: str, prompt: str, max_tokens: int, timeout: float
) -> tuple[int, int, float]:
    """Return (prompt tokens, generated tokens, seconds) for one completion."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            # Without this the model may stop early and the levels stop being
            # comparable: a level that happened to emit shorter replies would
            # look faster for a reason that has nothing to do with concurrency.
            "ignore_eos": True,
            "temperature": 0.0,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    elapsed = time.monotonic() - started
    usage = payload.get("usage", {})
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), elapsed


async def run_level(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    timeout: float,
) -> dict:
    # The executor is sized to the level, not left to asyncio's default. That
    # default is min(32, cpu_count + 4), so an unsized pool silently caps every
    # level above it: a sweep to 256 then reports identical throughput from 32
    # upward and wall times that double at 64, quadruple at 128 and so on --
    # the signature of the client queueing, read as the server saturating. This
    # measurement made exactly that mistake before the pool was sized here.
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                pool, request_once, endpoint, model, prompt, max_tokens, timeout
            )
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.monotonic() - started

    ok = [r for r in results if not isinstance(r, BaseException)]
    prompt_tokens = sum(r[0] for r in ok)
    generated = sum(r[1] for r in ok)
    latencies = [r[2] for r in ok]
    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        # Throughput is over wall time for the whole level, which is what a
        # server's capacity means; per-request latency is reported beside it
        # because the two move in opposite directions as concurrency rises.
        "generated_per_s": generated / wall if wall else 0.0,
        "prompt_per_s": prompt_tokens / wall if wall else 0.0,
        "latency_p50": statistics.median(latencies) if latencies else float("nan"),
        "latency_max": max(latencies) if latencies else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument(
        "--levels",
        default="1,2,4,8,16,32,64,128,256",
        help="Concurrency levels to sweep, lowest first.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--label", default="decode")
    args = parser.parse_args()

    if not args.endpoint or not args.model:
        parser.error("--endpoint and --model are required (or ENDPOINT/MODEL)")

    # A word repeated to length. Real text would be compressed differently by
    # the tokenizer and the prompt length is the variable under control here.
    prompt = "token " * args.prompt_tokens
    endpoint = args.endpoint.rstrip("/") + "/v1/completions"

    print(f"# {args.label}: prompt≈{args.prompt_tokens} tok, output={args.max_tokens} tok")
    print(
        f"{'conc':>5} {'ok':>4} {'fail':>4} {'wall_s':>8} "
        f"{'gen_tok/s':>10} {'prompt_tok/s':>12} {'p50_s':>7} {'max_s':>7}"
    )
    for level in [int(x) for x in args.levels.split(",")]:
        result = asyncio.run(
            run_level(
                endpoint,
                args.model,
                prompt,
                args.max_tokens,
                level,
                args.timeout,
            )
        )
        print(
            f"{result['concurrency']:>5} {result['ok']:>4} {result['failed']:>4} "
            f"{result['wall_s']:>8.1f} {result['generated_per_s']:>10.1f} "
            f"{result['prompt_per_s']:>12.1f} "
            f"{result['latency_p50']:>7.1f} {result['latency_max']:>7.1f}"
        )
        # A level that could not complete says the ceiling was passed; going
        # higher only measures the timeout.
        if result["failed"] and result["ok"] == 0:
            print(f"# level {level} failed entirely; stopping the sweep")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""P2 load harness: TTFT / ITL / throughput vs concurrency.

Sends streaming /v1/completions requests in concurrent waves and records
one CSV row per request in the shared schema (libs/metrics.py).

Scenarios (trimmed for the $128 envelope):
  prompt_lens = 512 / 2048 / 8192 tokens, gen = 128, concs = 1 / 8 / 32.

Usage (from repo root):
  python3 p02-bench/bench.py --base-url http://<GPU>:8000 --quick   # smoke
  python3 p02-bench/bench.py --base-url http://<GPU>:8000           # full sweep

Outputs to p02-bench/results/<run-id>/: results.csv, summary.json, manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import append_csv, compute_stats, row, summarize  # noqa: E402
from prompts import make_prompt, prompt_len_tokens  # noqa: E402
from vllm_client import health  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


async def server_version(base_url: str) -> str:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/version", timeout=10) as r:
                return (await r.text()).strip()[:80]
    except Exception:
        return "unknown"


async def one_request(session: aiohttp.ClientSession, base_url: str, model: str,
                      prompt: str, max_tokens: int, timeout_s: float,
                      retries: int = 2):
    """Single streaming request -> (RequestStats, ok, err).

    Retries connection-level failures (SYN burst vs graph capture can RST a
    whole wave) with backoff — but only while zero tokens arrived, so a retry
    never double-counts a partially-served request.
    """
    url = f"{base_url}/v1/completions"
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    attempt_err: Exception | None = None
    for attempt in range(retries):
        send_t = time.perf_counter()
        token_times: list[float] = []
        usage_tokens: int | None = None
        try:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                resp.raise_for_status()
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    # Authoritative token count: spec decoding batches multiple
                    # tokens per SSE event, so chunk-counting lies (P6 lesson).
                    if '"usage"' in data:
                        try:
                            import json as _json
                            usage_tokens = _json.loads(data)["usage"].get(
                                "completion_tokens", usage_tokens)
                        except Exception:
                            pass
                        continue
                    token_times.append(time.perf_counter())
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            if token_times:  # partial stream: record, don't retry
                break
            attempt_err = e
            await asyncio.sleep(2.0 * (attempt + 1))
            continue
        except Exception as e:  # record failures as rows, don't kill the wave
            stats = compute_stats(send_t, token_times, len(token_times))
            return stats, False, f"{type(e).__name__}: {e}"[:200]
        n_tok = usage_tokens if usage_tokens else len(token_times)
        stats = compute_stats(send_t, token_times, n_tok)
        return stats, True, ""
    stats = compute_stats(time.perf_counter(), [], 0)
    return stats, False, f"connect-failed-x{retries}: {type(attempt_err).__name__}"[:200]


async def wave(session: aiohttp.ClientSession, base_url: str, model: str,
               prompt: str, max_tokens: int, concurrency: int,
               timeout_s: float):
    sem = asyncio.Semaphore(max(concurrency, 1))

    async def bounded(i: int):
        # 50ms stagger: 32 simultaneous SYNs can RST against a server in
        # graph-capture; stagger costs nothing vs TTFT (timed per request).
        await asyncio.sleep(0.05 * i)
        async with sem:
            return await one_request(session, base_url, model, prompt,
                                     max_tokens, timeout_s)

    return await asyncio.gather(*[bounded(i) for i in range(concurrency)])


async def amain(a: argparse.Namespace) -> int:
    run_id = a.run_id or datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"run_id={run_id} model={a.model} git={git_sha()}")
    if not await health(a.base_url):
        print(f"FATAL: {a.base_url}/health unreachable. Is vLLM up?", file=sys.stderr)
        return 1
    out = Path(a.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    csv_path = str(out / "results.csv")
    version = await server_version(a.base_url)

    manifest = {
        "run_id": run_id, "git_sha": git_sha(), "model": a.model,
        "base_url_host": a.base_url.rsplit("@", 1)[-1],
        "server_version": version,
        "prompt_lens": a.prompt_lens, "gen_lens": a.gen_lens,
        "concs": a.concs, "repeats": a.repeats, "started_utc": run_id,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary: dict = {}
    async with aiohttp.ClientSession() as session:
        for plen in a.prompt_lens:
            prompt = make_prompt(plen)
            assert abs(prompt_len_tokens(prompt) - plen) <= 2, "prompt size drift"
            for gen in a.gen_lens:
                for conc in a.concs:
                    key = f"p{plen}/g{gen}/c{conc}"
                    # Warmup (unrecorded) shakes out compile/graph capture.
                    await wave(session, a.base_url, a.model, prompt, gen, 1,
                               a.timeout_s)
                    rows, ttfts, itls, tpss = [], [], [], []
                    for rep in range(a.repeats):
                        t0 = time.perf_counter()
                        results = await wave(session, a.base_url, a.model,
                                             prompt, gen, conc, a.timeout_s)
                        wall = time.perf_counter() - t0
                        for stats, ok, err in results:
                            rows.append(row(run_id, key, conc, plen, stats, ok, err))
                            if ok:
                                ttfts.append(stats.ttft_ms)
                                itls.append(stats.itl_p95_ms)
                                tpss.append(stats.toks_per_s)
                        # Wave throughput (server-side view).
                        ntok = sum(r["gen_len"] for r in rows[-conc:] if r["ok"])
                        summary.setdefault(key, {}).setdefault(
                            "wave_tok_s", []).append(ntok / wall if wall > 0 else 0)
                        time.sleep(a.cooldown_s)
                    append_csv(csv_path, rows)
                    s = summary[key]
                    s.update({"ttft_ms": summarize(ttfts), "itl_p95_ms": summarize(itls),
                              "toks_per_s": summarize(tpss), "errors": sum(1 for r in rows if not r["ok"])})
                    print(f"{key}: n={len(rows)} err={s['errors']} "
                          f"TTFT p50={s['ttft_ms']['p50']:.0f}/p95={s['ttft_ms']['p95']:.0f}ms "
                          f"ITL95 p50={s['itl_p95_ms']['p50']:.1f}ms "
                          f"wave {s['wave_tok_s'][-1]:.1f} tok/s")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWROTE {csv_path} + summary.json + manifest.json")
    print("Plot with: python3 p02-bench/plot.py", csv_path)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="P2 TTFT/ITL load harness")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "p02-bench" / "results"))
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--gen-lens", type=int, nargs="+", default=[128])
    ap.add_argument("--concs", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quick", action="store_true",
                    help="smoke: p512/g64, concs 1+4, 1 repeat")
    a = ap.parse_args()
    if a.quick:
        a.prompt_lens, a.gen_lens, a.concs, a.repeats = [512], [64], [1, 4], 1
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()

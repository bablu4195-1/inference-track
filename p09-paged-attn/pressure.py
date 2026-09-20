#!/usr/bin/env python3
"""P9 memory-pressure driver: push KV usage to 100%, record the distress timeline.

Sends repeated high-concurrency, long-context waves (default 32 x 6k tokens
~ 10.7 GB KV against ~6.4 GB headroom -> guaranteed pressure on A10G) and
scrapes /metrics after every wave. Detects the canonical vLLM distress
sequence: kv% saturates -> waiting queue grows -> swapped (preemption/
recompute) -> request errors.

Output: results.csv (shared schema, `label:pressure`) + timeline.json with
first_wait_wave, first_swap_wave, totals. analyze.py joins block-size configs.

Usage:
  python3 p09-paged-attn/pressure.py --base-url http://<GPU>:8000 --label bs16
  # monitor.py in a second terminal recommended for the full-resolution trace
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT / "p02-bench"))
sys.path.insert(0, str(ROOT / "p03-kvcalc"))

import bench  # noqa: E402
import monitor  # noqa: E402
from metrics import append_csv, row  # noqa: E402
from prompts import make_prompt  # noqa: E402
from vllm_client import health  # noqa: E402


def snap(base_url: str) -> dict[str, float]:
    try:
        m = monitor.fetch_metrics(base_url)
    except Exception:
        return {}
    return {k: float(m.get(k, 0.0)) for k in
            ("vllm:kv_cache_usage_perc", "vllm:num_requests_running",
             "vllm:num_requests_waiting", "vllm:num_requests_swapped")}


async def amain(a: argparse.Namespace) -> int:
    if not await health(a.base_url):
        print(f"FATAL: {a.base_url}/health unreachable", file=sys.stderr)
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_dir) / f"{a.label}-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    csv_path = str(out / "results.csv")

    # DISTINCT prompts per request: identical prompts share one prefix block
    # under prefix-caching and never pressurize memory (P3 lesson 2026-09-19).
    prompts = [make_prompt(a.prompt_tokens, seed=5000 + i) for i in range(a.conc)]
    timeline, first_wait, first_swap = [], None, None
    err_total, tok_total, wall_total = 0, 0, 0.0

    async def distinct_wave():
        sem = asyncio.Semaphore(max(a.conc, 1))

        async def bounded(i: int):
            await asyncio.sleep(0.05 * i)
            async with sem:
                return await bench.one_request(session, a.base_url, a.model,
                                               prompts[i], a.gen, a.timeout_s)
        return await asyncio.gather(*[bounded(i) for i in range(a.conc)])

    async with aiohttp.ClientSession() as session:
        # Small warmup so compile cost doesn't pollute wave 0.
        await bench.wave(session, a.base_url, a.model, make_prompt(512),
                         32, 1, a.timeout_s)
        for w in range(a.waves):
            t0 = time.perf_counter()
            sampler_stop: list[bool] = [False]
            sampler_peak = {"waiting": 0.0, "swapped": 0.0, "kv": 0.0}

            async def sampler() -> None:
                import urllib.request
                while not sampler_stop[0]:
                    try:
                        def _get():
                            with urllib.request.urlopen(
                                    f"{a.base_url}/metrics", timeout=5) as r:
                                return r.read().decode()
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, _get)
                        for line in text.splitlines():
                            if line.startswith("#") or " " not in line:
                                continue
                            name, _, val = line.partition(" ")
                            try:
                                v = float(val)
                            except ValueError:
                                continue
                            name = name.split("{")[0]
                            if name == "vllm:num_requests_waiting":
                                sampler_peak["waiting"] = max(
                                    sampler_peak["waiting"], v)
                            elif name == "vllm:num_requests_swapped":
                                sampler_peak["swapped"] = max(
                                    sampler_peak["swapped"], v)
                            elif name == "vllm:kv_cache_usage_perc":
                                sampler_peak["kv"] = max(sampler_peak["kv"], v)
                    except Exception:
                        pass
                    await asyncio.sleep(2.0)

            task = asyncio.create_task(sampler())
            res = await distinct_wave()
            sampler_stop[0] = True
            await task
            wall = time.perf_counter() - t0
            errs = sum(1 for _, ok, _ in res if not ok)
            toks = sum(s.gen_tokens for s, ok, _ in res if ok)
            err_total += errs
            tok_total += toks
            wall_total += wall
            rows = [row(stamp, f"{a.label}:pressure", a.conc, a.prompt_tokens,
                        s, ok, err) for s, ok, err in res]
            append_csv(csv_path, rows)
            m = {"vllm:num_requests_waiting": sampler_peak["waiting"],
                 "vllm:num_requests_swapped": sampler_peak["swapped"],
                 "vllm:kv_cache_usage_perc": sampler_peak["kv"]}
            ev = {"wave": w, "t_s": round(time.perf_counter(), 1),
                  "errs": errs, "wave_tok_s": round(toks / wall, 1) if wall else 0,
                  **{k.split(":")[1]: v for k, v in m.items()}}
            timeline.append(ev)
            if first_wait is None and m.get("vllm:num_requests_waiting", 0) > 0:
                first_wait = w
                print(f"  wave{w}: QUEUEING started (waiting>0)")
            if first_swap is None and m.get("vllm:num_requests_swapped", 0) > 0:
                first_swap = w
                print(f"  wave{w}: EVICTION started (swapped>0)")
            print(f"wave{w}: errs={errs} kv={m.get('vllm:kv_cache_usage_perc', 0):.0%} "
                  f"wait={m.get('vllm:num_requests_waiting', 0):.0f} "
                  f"swapped={m.get('vllm:num_requests_swapped', 0):.0f} "
                  f"{ev['wave_tok_s']:.0f} tok/s")
            time.sleep(a.cooldown_s)

    summary = {
        "label": a.label, "run_utc": stamp, "model": a.model,
        "prompt_tokens": a.prompt_tokens, "conc": a.conc, "waves": a.waves,
        "first_wait_wave": first_wait, "first_swap_wave": first_swap,
        "total_errors": err_total,
        "error_rate": round(err_total / max(a.waves * a.conc, 1), 4),
        "mean_wave_tok_s": round(tok_total / wall_total, 1) if wall_total else 0,
        "peak_kv": max((e.get("kv_cache_usage_perc", 0) for e in timeline), default=0),
        "timeline": timeline,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nlabel={a.label}: first_wait=wave{first_wait} first_swap=wave{first_swap} "
          f"err_rate={summary['error_rate']:.1%} peak_kv={summary['peak_kv']:.0%}")
    print(f"WROTE {csv_path} + summary.json")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="P9 memory-pressure driver")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", required=True, help="e.g. bs16")
    ap.add_argument("--out-dir", default=str(ROOT / "p09-paged-attn" / "results"))
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--waves", type=int, default=6)
    ap.add_argument("--cooldown-s", type=float, default=10.0)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()

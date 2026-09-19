#!/usr/bin/env python3
"""P3 predictor validation: ramp concurrency until distress, compare vs kv_math.

For each seq_len: predict max_batch with libs/kv_math, then run single waves
of growing concurrency (bench.py machinery). 'Distress' = any ok=False row
OR waiting-queue spike observed in /metrics. Records predicted vs observed
max stable batch -> validation.json. The ±15% accuracy claim lives or dies here.

GPU procedure (one boot, ~30-45 min, ~$0.40):
  1. Terminal 1: python3 p03-kvcalc/monitor.py --base-url http://<GPU>:8000
  2. Terminal 2: python3 p03-kvcalc/pressure.py --base-url http://<GPU>:8000
  3. Paste the validation.json table into README.md run table.

Usage: pressure.py --base-url URL [--model ID] [--seq-lens 2048 4096 8192]
                   [--gen 128] [--step 2] [--cooldown-s 8]
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

import bench  # noqa: E402
from kv_math import fits, max_batch  # noqa: E402
from prompts import make_prompt, prompt_len_tokens  # noqa: E402
from vllm_client import health  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def waiting_sync(base_url: str) -> float:
    sys.path.insert(0, str(ROOT / "p03-kvcalc"))
    import monitor
    try:
        return float(monitor.fetch_metrics(base_url).get(
            "vllm:num_requests_waiting", 0.0))
    except Exception:
        return 0.0


async def sample_during(session: aiohttp.ClientSession, base_url: str,
                        stop: asyncio.Event, out: dict) -> None:
    """Scrape /metrics every 2s WHILE a wave runs. Post-wave scrapes always
    read an empty queue (P3 lesson 2026-09-19) — transient queueing is only
    visible mid-wave."""
    import urllib.request
    import json as _json

    peak = {"waiting": 0.0, "swapped": 0.0, "kv": 0.0, "samples": 0}
    while not stop.is_set():
        try:
            def _get():
                with urllib.request.urlopen(f"{base_url}/metrics",
                                            timeout=5) as r:
                    return r.read().decode()
            text = await asyncio.get_event_loop().run_in_executor(None, _get)
            m: dict[str, float] = {}
            for line in text.splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                name, _, val = line.partition(" ")
                try:
                    m[name.split("{")[0]] = float(val)
                except ValueError:
                    pass
            peak["waiting"] = max(peak["waiting"],
                                  m.get("vllm:num_requests_waiting", 0.0))
            peak["swapped"] = max(peak["swapped"],
                                  m.get("vllm:num_requests_swapped", 0.0))
            peak["kv"] = max(peak["kv"],
                             m.get("vllm:kv_cache_usage_perc", 0.0))
            peak["samples"] += 1
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    out.update(peak)


async def amain(a: argparse.Namespace) -> int:
    if not await health(a.base_url):
        print(f"FATAL: {a.base_url}/health unreachable", file=sys.stderr)
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"run_utc": stamp, "model": a.model, "label": a.label,
                    "kv_predictor": "libs/kv_math.py", "seqs": {}}

    async with aiohttp.ClientSession() as session:
        for seq in a.seq_lens:
            pred = max_batch(seq, model=a.kv_model, dtype=a.dtype)
            # DISTINCT prompts per request (seeded): identical prompts share
            # one prefix block under prefix-caching and measure ~nothing.
            prompts = [make_prompt(seq, seed=1000 + i)
                       for i in range(max(pred + a.overshoot, 1))]
            assert abs(prompt_len_tokens(prompts[0]) - seq) <= 2
            print(f"\n== seq={seq}: predicted max_batch={pred} ==")

            async def distinct_wave(n: int):
                sem = asyncio.Semaphore(max(n, 1))

                async def bounded(i: int):
                    await asyncio.sleep(0.05 * i)
                    async with sem:
                        return await bench.one_request(
                            session, a.base_url, a.model, prompts[i],
                            a.gen, a.timeout_s)
                return await asyncio.gather(*[bounded(i) for i in range(n)])

            # Warmup at conc 1 so compile cost doesn't masquerade as distress.
            await distinct_wave(1)
            observed, distress_at, note = 0, None, ""
            for conc in range(1, pred + a.overshoot + 1, 1):
                stop, mid = asyncio.Event(), {}
                sampler = asyncio.create_task(
                    sample_during(session, a.base_url, stop, mid))
                results = await distinct_wave(conc)
                stop.set()
                await sampler
                errs = sum(1 for _, ok, _ in results if not ok)
                q = mid.get("waiting", 0.0)  # PEAK mid-wave queue depth
                tag = (f"conc={conc} errs={errs} max_waiting={q:.0f} "
                       f"max_kv={mid.get('kv', 0.0):.0%}")
                if errs or q >= a.queue_limit:
                    distress_at = conc
                    note = f"errors={errs} waiting={q:.0f}"
                    print(f"  {tag}  <-- DISTRESS, stopping ramp")
                    break
                observed = conc
                print(f"  {tag}  ok")
                time.sleep(a.cooldown_s)
            err_pct = (observed - pred) / max(pred, 1) * 100
            verdict = "PASS" if abs(err_pct) <= 15 else "RECALIBRATE"
            report["seqs"][str(seq)] = {
                "predicted_max_batch": pred, "observed_stable": observed,
                "distress_at": distress_at, "distress_note": note,
                "err_pct": round(err_pct, 1), "verdict": verdict,
            }
            print(f"seq={seq}: pred={pred} observed={observed} "
                  f"err={err_pct:+.1f}% -> {verdict}")
    (out / f"validation-{a.label}-{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"\nWROTE {out}/validation-{a.label}-{stamp}.json — paste table into p03-kvcalc/README.md")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 pressure validation")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", default="run",
                    help="tag for output filename + row config")
    ap.add_argument("--out-dir", default=".",
                    help="directory for validation-<label>-<utc>.json")
    ap.add_argument("--kv-model", default="qwen2.5-7b",
                    help="key in libs/kv_math.MODEL_SPECS")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp8", "awq"])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--overshoot", type=int, default=4,
                    help="how far past prediction to probe")
    ap.add_argument("--queue-limit", type=float, default=4.0,
                    help="waiting-queue depth that counts as distress")
    ap.add_argument("--cooldown-s", type=float, default=8.0)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()

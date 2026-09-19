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


async def amain(a: argparse.Namespace) -> int:
    if not await health(a.base_url):
        print(f"FATAL: {a.base_url}/health unreachable", file=sys.stderr)
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"p03-kvcalc/validation-{stamp}.json")
    report: dict = {"run_utc": stamp, "model": a.model,
                    "kv_predictor": "libs/kv_math.py", "seqs": {}}

    async with aiohttp.ClientSession() as session:
        for seq in a.seq_lens:
            pred = max_batch(seq, model=a.kv_model, dtype=a.dtype)
            prompt = make_prompt(seq)
            assert abs(prompt_len_tokens(prompt) - seq) <= 2
            print(f"\n== seq={seq}: predicted max_batch={pred} ==")
            # Warmup at conc 1 so compile cost doesn't masquerade as distress.
            await bench.wave(session, a.base_url, a.model, prompt, a.gen, 1,
                             a.timeout_s)
            observed, distress_at, note = 0, None, ""
            for conc in range(1, pred + a.overshoot + 1, 1):
                results = await bench.wave(session, a.base_url, a.model,
                                           prompt, a.gen, conc, a.timeout_s)
                errs = sum(1 for _, ok, _ in results if not ok)
                q = waiting_sync(a.base_url)
                tag = f"conc={conc} errs={errs} waiting={q:.0f}"
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
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWROTE {out} — paste table into p03-kvcalc/README.md")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 pressure validation")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kv-model", default="qwen2.5-7b",
                    help="key in libs/kv_math.MODEL_SPECS")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp8"])
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

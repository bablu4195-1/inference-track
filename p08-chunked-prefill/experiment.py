#!/usr/bin/env python3
"""P8 mixed-load experiment: one long-context hog + N decode streams.

Two phases per run:
  baseline: D decode-only requests (short prompt, long gen) -> ITL_base
  mixed:    D decodes + 1 hog (8k prompt) launched together ->
            ITL_mixed (decode starvation signal) + hog TTFT

Key metric: starvation_x = ITL_mixed / ITL_base.
  ~1.0x = scheduler isolated decode (chunking works)
  >>1x  = hog's prefill blocked decodes (chunking off / budget too large)

Run once per server config (see run-matrix.sh); compare.py joins the labels.

Usage:
  python3 p08-chunked-prefill/experiment.py --base-url http://<GPU>:8000 \
      --label on-8192 --out-dir p08-chunked-prefill/results
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT / "p02-bench"))

import bench  # noqa: E402
from metrics import append_csv, row  # noqa: E402
from prompts import make_prompt  # noqa: E402
from vllm_client import health  # noqa: E402


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


async def amain(a: argparse.Namespace) -> int:
    if not await health(a.base_url):
        print(f"FATAL: {a.base_url}/health unreachable", file=sys.stderr)
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_dir) / f"{a.label}-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    csv_path = str(out / "results.csv")

    dec_prompt = make_prompt(512)
    hog_prompt = make_prompt(a.hog_tokens)
    summary: dict = {"label": a.label, "run_utc": stamp, "model": a.model,
                     "decode_conc": a.decode_conc, "hog_tokens": a.hog_tokens}
    base_itls, mixed_itls, hog_ttfts = [], [], []

    async with aiohttp.ClientSession() as session:
        for rep in range(a.repeats):
            # Phase 1: decode-only baseline.
            res = await bench.wave(session, a.base_url, a.model, dec_prompt,
                                   a.decode_gen, a.decode_conc, a.timeout_s)
            rows = [row(stamp, f"{a.label}:base:decode", a.decode_conc, 512,
                        s, ok, err) for s, ok, err in res]
            append_csv(csv_path, rows)
            base_itls += [s.itl_p95_ms for s, ok, _ in res if ok]
            time.sleep(a.cooldown_s)

            # Phase 2: mixed — D decodes + 1 hog, same start gun.
            async def decode_one():
                return await bench.one_request(
                    session, a.base_url, a.model, dec_prompt,
                    a.decode_gen, a.timeout_s)

            async def hog_one():
                return await bench.one_request(
                    session, a.base_url, a.model, hog_prompt,
                    a.hog_gen, a.timeout_s)

            mixed = await asyncio.gather(
                *[decode_one() for _ in range(a.decode_conc)], hog_one())
            *dec_res, hog_res = mixed
            rows = [row(stamp, f"{a.label}:mixed:decode", a.decode_conc + 1, 512,
                        s, ok, err) for s, ok, err in dec_res]
            hs, hok, herr = hog_res
            rows.append(row(stamp, f"{a.label}:mixed:hog", a.decode_conc + 1,
                            a.hog_tokens, hs, hok, herr))
            append_csv(csv_path, rows)
            mixed_itls += [s.itl_p95_ms for s, ok, _ in dec_res if ok]
            if hok:
                hog_ttfts.append(hs.ttft_ms)
            print(f"rep{rep}: base_ITL95={med([s.itl_p95_ms for s, ok, _ in res if ok]):.1f}ms "
                  f"mixed_ITL95={med([s.itl_p95_ms for s, ok, _ in dec_res if ok]):.1f}ms "
                  f"hog_TTFT={hs.ttft_ms:.0f}ms ok={hok}")
            time.sleep(a.cooldown_s)

    b, m = med(base_itls), med(mixed_itls)
    summary.update({
        "itl_base_ms": round(b, 2), "itl_mixed_ms": round(m, 2),
        "starvation_x": round(m / b, 2) if b else 0.0,
        "hog_ttft_ms": round(med(hog_ttfts), 1),
        "n_base": len(base_itls), "n_mixed": len(mixed_itls),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nlabel={a.label}: ITL {b:.1f} -> {m:.1f}ms "
          f"(starvation {summary['starvation_x']}x), hog TTFT {summary['hog_ttft_ms']:.0f}ms")
    print(f"WROTE {csv_path} + summary.json")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="P8 mixed prefill/decode experiment")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", required=True,
                    help="e.g. on-8192 (chunked on, budget 8192)")
    ap.add_argument("--out-dir", default=str(ROOT / "p08-chunked-prefill" / "results"))
    ap.add_argument("--decode-conc", type=int, default=4)
    ap.add_argument("--decode-gen", type=int, default=256)
    ap.add_argument("--hog-tokens", type=int, default=8192)
    ap.add_argument("--hog-gen", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cooldown-s", type=float, default=8.0)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()

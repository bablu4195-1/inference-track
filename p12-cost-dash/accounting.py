#!/usr/bin/env python3
"""P12 token accounting: results CSVs + GPU spend -> $/M tokens, MFU.

Cost model (documented, auditable):
  1. Total GPU cost for the run comes from --cost-usd, or from
     infra/aws/cost-log.csv filtered by --note-match.
  2. Cost is attributed to requests PROPORTIONAL TO e2e_ms (GPU-time
     consumed), not token count: a 8k-prefill request costs more than its
     tokens suggest. $/M = total_cost / total_tokens * 1e6 (blended + split).
  3. MFU per request: achieved = toks_per_s * 2 * params / 1e12 TFLOPS;
     MFU = achieved / peak_TFLOPS. Decode is memory-bound so expect 1-5% —
     the point is tracking it, not admiring it. Prefill-heavy rows overstate
     nothing: tok/s there includes prefill, formula still bounds correctly.

Usage:
  python3 p12-cost-dash/accounting.py results1.csv [results2.csv ...] \\
      --model qwen2.5-7b --tenant track --cost-usd 0.50 --out ledger.json
  python3 p12-cost-dash/accounting.py p02-bench/results/*/results.csv \\
      --cost-log-note "P2 full sweep" --report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Peak BF16/FP16 tensor TFLOPS + params for MFU math.
FLEET = {
    "g5.xlarge": {"gpu": "A10G", "peak_tflops": 125.0, "usd_per_hr_spot": 0.54},
}
MODELS = {
    "qwen2.5-7b": {"params_b": 7.6},
    "llama-3.1-8b": {"params_b": 8.0},
    "qwen2.5-0.5b": {"params_b": 0.5},
}


def read_cost_log(note_match: str) -> float:
    p = Path(__file__).resolve().parents[1] / "infra" / "aws" / "cost-log.csv"
    total = 0.0
    with open(p) as f:
        for r in csv.DictReader(f):
            if note_match.lower() in (r.get("note") or "").lower():
                total += float(r["cost_usd"])
    return total


def load_rows(paths: list[str]):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if str(r.get("ok")) != "True":
                    continue
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="P12 token accounting")
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--model", default="qwen2.5-7b", choices=list(MODELS))
    ap.add_argument("--tenant", default="track")
    ap.add_argument("--cost-usd", type=float, default=None)
    ap.add_argument("--cost-log-note", default=None,
                    help="sum cost-log.csv entries whose note matches")
    ap.add_argument("--gpu", default="g5.xlarge", choices=list(FLEET))
    ap.add_argument("--out", default="p12-cost-dash/ledger.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    cost = a.cost_usd
    if cost is None and a.cost_log_note:
        cost = read_cost_log(a.cost_log_note)
    if not cost:
        sys.exit("no cost: pass --cost-usd or --cost-log-note matching cost-log.csv")

    rows = load_rows(a.csvs)
    if not rows:
        sys.exit("no ok rows in input CSVs")
    params = MODELS[a.model]["params_b"] * 1e9
    peak = FLEET[a.gpu]["peak_tflops"]

    per_key: dict = defaultdict(lambda: {"n": 0, "in_tok": 0, "out_tok": 0,
                                         "e2e_ms": 0.0, "mfu": []})
    for r in rows:
        k = (a.tenant, a.model, r.get("config", ""))
        d = per_key[k]
        d["n"] += 1
        d["in_tok"] += int(float(r.get("prompt_len", 0)))
        d["out_tok"] += int(float(r.get("gen_len", 0)))
        d["e2e_ms"] += float(r.get("e2e_ms", 0))
        tps = float(r.get("toks_per_s", 0))
        d["mfu"].append(tps * 2 * params / 1e12 / peak * 100)

    total_e2e = sum(d["e2e_ms"] for d in per_key.values()) or 1.0
    ledger = {"tenant_default": a.tenant, "model": a.model, "gpu": a.gpu,
              "total_cost_usd": cost, "groups": []}
    tot_in = tot_out = 0
    for (tenant, model, config), d in sorted(per_key.items()):
        share = cost * d["e2e_ms"] / total_e2e
        tin, tou = d["in_tok"], d["out_tok"]
        tot_in += tin
        tot_out += tou
        mfus = sorted(d["mfu"])
        ledger["groups"].append({
            "tenant": tenant, "model": model, "config": config,
            "requests": d["n"], "in_tokens": tin, "out_tokens": tou,
            "cost_usd": round(share, 4),
            "usd_per_M_in": round(share / max(tin, 1) * 1e6, 2),
            "usd_per_M_out": round(share / max(tou, 1) * 1e6, 2),
            "usd_per_M_blended": round(share / max(tin + tou, 1) * 1e6, 2),
            "mfu_p50_pct": round(mfus[len(mfus) // 2], 2),
        })
    ledger["totals"] = {"in_tokens": tot_in, "out_tokens": tot_out,
                        "usd_per_M_blended": round(cost / max(tot_in + tot_out, 1) * 1e6, 2)}
    Path(a.out).write_text(json.dumps(ledger, indent=2))
    if a.report or True:
        print(f"{'tenant/model/config':>28} {'n':>4} {'$in/M':>8} {'$out/M':>8} {'$blend/M':>9} {'MFU%':>6}")
        for g in ledger["groups"]:
            print(f"{(g['tenant'] + '/' + g['model'] + '/' + g['config'])[-28:]:>28} "
                  f"{g['requests']:>4} {g['usd_per_M_in']:>8.2f} {g['usd_per_M_out']:>8.2f} "
                  f"{g['usd_per_M_blended']:>9.2f} {g['mfu_p50_pct']:>6.2f}")
        print(f"TOTAL: {tot_in + tot_out} tokens for ${cost:.2f} "
              f"= ${ledger['totals']['usd_per_M_blended']:.2f}/M blended")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()

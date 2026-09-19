#!/usr/bin/env python3
"""P14 SLO burn reporter: error-budget accounting from results CSVs.

SLOs (track defaults, tunable):
  --ttft-slo-ms 2000   per-request TTFT deadline
  --itl-slo-ms 500     per-request ITL-p95 deadline
  --avail-target 99.0  availability target %  (ok rows / all rows)

A request burns budget if ok=False OR ttft>slo OR itl>slo. Report:
  per-fault burn rate, error budget remaining, MTTR proxy (first ok row after
  last bad row), recovery curve (burn rate per minute bucket).

Usage: python3 p14-chaos/slo.py baseline.csv fault.csv [--ttft-slo-ms 2000 ...]
Compares fault window against baseline: burn delta is the chaos finding.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def load(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def burn(rows, ttft_slo: float, itl_slo: float):
    bad, per_min = 0, defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        ok = str(r.get("ok")) == "True"
        slow = float(r.get("ttft_ms", 0)) > ttft_slo or float(r.get("itl_p95_ms", 0)) > itl_slo
        if not ok or slow:
            bad += 1
        per_min[i // 10][1] += 1
        if not ok or slow:
            per_min[i // 10][0] += 1
    n = max(len(rows), 1)
    return {"n": n, "bad": bad, "burn_rate": bad / n,
            "availability": (n - sum(1 for r in rows if str(r.get("ok")) != "True")) / n * 100,
            "per_10": dict(sorted(per_min.items()))}


def mttr_proxy(rows):
    """Index distance from last bad row to end (rows are time-ordered)."""
    last_bad = max((i for i, r in enumerate(rows)
                    if str(r.get("ok")) != "True"
                    or float(r.get("ttft_ms", 0)) > 1e12), default=-1)
    return len(rows) - 1 - last_bad


def main() -> None:
    base_p, fault_p = sys.argv[1], sys.argv[2]
    ttft_slo = float(sys.argv[sys.argv.index("--ttft-slo-ms") + 1]) \
        if "--ttft-slo-ms" in sys.argv else 2000.0
    itl_slo = float(sys.argv[sys.argv.index("--itl-slo-ms") + 1]) \
        if "--itl-slo-ms" in sys.argv else 500.0
    avail = float(sys.argv[sys.argv.index("--avail-target") + 1]) \
        if "--avail-target" in sys.argv else 99.0
    base, fault = load(base_p), load(fault_p)
    bb, fb = burn(base, ttft_slo, itl_slo), burn(fault, ttft_slo, itl_slo)
    print(f"{'window':>9} {'n':>5} {'burn%':>7} {'avail%':>7}")
    print(f"{'baseline':>9} {bb['n']:>5} {bb['burn_rate']:>6.1%} {bb['availability']:>6.2f}")
    print(f"{'fault':>9} {fb['n']:>5} {fb['burn_rate']:>6.1%} {fb['availability']:>6.2f}")
    print(f"burn delta: {fb['burn_rate'] - bb['burn_rate']:+.1%}  "
          f"budget headroom: {fb['availability'] - avail:+.2f}pp vs {avail:.0f}% target  "
          f"MTTR-proxy: {mttr_proxy(fault)} rows")
    print("recovery curve (bad/total per 10-row bucket):",
          {k: f"{v[0]}/{v[1]}" for k, v in fb["per_10"].items()})


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: slo.py baseline.csv fault.csv [--ttft-slo-ms N ...]")
    main()

#!/usr/bin/env python3
"""P10 comparison: colocated vs disaggregated prefill/decode.

Usage: python3 p10-disagg/compare.py coloc-summary.json disagg-summary.json
  [--plot out.png] [--report]
Each summary.json is a P2 full-sweep summary. Reports per scenario key:
  TTFT p50/p95, ITL95 p50, wave tok/s + disagg delta% on each.
Disagg wins when: long-context TTFT drops (prefill pool uncontended) and
mixed-workload ITL flattens. It loses when: transfer overhead dominates
(short prompts) or decode pool saturates first.
"""

from __future__ import annotations

import json
import sys


def load(p: str):
    with open(p) as f:
        return json.load(f)


def main() -> None:
    coloc, disagg = load(sys.argv[1]), load(sys.argv[2])
    plot = sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None
    md = "--report" in sys.argv
    keys = sorted(k for k in coloc if isinstance(coloc[k], dict) and "ttft_ms" in coloc[k])
    if md:
        print("| scenario | TTFT Δ% | ITL95 Δ% | tok/s Δ% | verdict |")
        print("|---|---|---|---|---|")
    else:
        print(f"{'scenario':>14} {'TTFT_d%':>8} {'ITL_d%':>8} {'tps_d%':>8} verdict")
    tps_d, wins = [], 0
    for k in keys:
        if k not in disagg or not isinstance(disagg[k], dict):
            continue
        c, d = coloc[k], disagg[k]
        t = (d["ttft_ms"]["p50"] - c["ttft_ms"]["p50"]) / max(c["ttft_ms"]["p50"], 1e-9) * 100
        i = (d["itl_p95_ms"]["p50"] - c["itl_p95_ms"]["p50"]) / max(c["itl_p95_ms"]["p50"], 1e-9) * 100
        w = (d.get("wave_tok_s", [0])[-1] - c.get("wave_tok_s", [0])[-1]) / max(c.get("wave_tok_s", [1])[-1], 1e-9) * 100
        v = "disagg" if (t + i - w) < 0 else "coloc"
        wins += v == "disagg"
        if md:
            print(f"| {k} | {t:+.1f}% | {i:+.1f}% | {w:+.1f}% | {v} |")
        else:
            print(f"{k:>14} {t:>+7.1f}% {i:>+7.1f}% {w:>+7.1f}% {v}")
        tps_d.append(w)
    print(f"disagg wins {wins}/{len(tps_d)} scenarios "
          f"(mean tok/s delta {sum(tps_d)/max(len(tps_d),1):+.1f}%)")
    if plot and keys:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(keys, tps_d)
        ax.axhline(0, color="k")
        ax.set_ylabel("tok/s delta % (disagg - coloc)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: compare.py coloc-summary.json disagg-summary.json [--plot o.png] [--report]")
    main()

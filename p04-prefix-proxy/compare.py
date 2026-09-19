#!/usr/bin/env python3
"""P4 A/B comparison: cache-on vs cache-off workload CSVs -> TTFT-cut table.

Usage: python3 p04-prefix-proxy/compare.py on.csv off.csv [--plot out.png]
Prints per-turn median TTFT for both configs and the cut:
  cut% = 1 - TTFT_on / TTFT_off   (positive = caching helped)
Expect: turn0 ~ 0% (cold prefix), turn>=1 strongly positive on long prefixes.
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str):
    per_turn: dict = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if str(r.get("ok")) != "True":
                continue
            per_turn[int(r["turn"])].append(float(r["ttft_ms"]))
    return per_turn


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    plot = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--plot" else None
    on, off = load(a_path), load(b_path)
    print(f"{'turn':>4} {'n_on':>5} {'TTFT_on':>9} {'TTFT_off':>9} {'cut%':>7}")
    cuts = {}
    for t in sorted(set(on) | set(off)):
        mo = statistics.median(on[t]) if t in on else float("nan")
        mf = statistics.median(off[t]) if t in off else float("nan")
        cut = (1 - mo / mf) * 100 if mf else 0.0
        cuts[t] = cut
        print(f"{t:>4} {len(on.get(t, [])):>5} {mo:>9.0f} {mf:>9.0f} {cut:>6.1f}%")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = sorted(cuts)
        fig, ax = plt.subplots()
        ax.bar(ts, [cuts[t] for t in ts])
        ax.set_xlabel("turn (0 = cold prefix)")
        ax.set_ylabel("TTFT cut % (cache-on vs off)")
        ax.set_title("Prefix-caching TTFT win per multi-turn turn")
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: compare.py on.csv off.csv [--plot out.png]")
    main()

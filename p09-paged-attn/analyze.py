#!/usr/bin/env python3
"""P9 analysis: join per-block-size summaries -> eviction comparison + plots.

Usage: python3 p09-paged-attn/analyze.py results/ [--plot out.png] [--report]
Reads results/<label>-*/summary.json. Table:
  label | first_wait | first_swap | err_rate | mean_tok_s | peak_kv
--report prints a markdown table for pasting into REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(results_dir: Path):
    rows = []
    for s in sorted(results_dir.glob("*/summary.json")):
        try:
            d = json.loads(s.read_text())
        except Exception:
            continue
        if "first_swap_wave" not in d:
            continue
        rows.append(d)
    return sorted(rows, key=lambda d: d["label"])


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    plot = sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None
    md = "--report" in sys.argv
    rows = load(results)
    hdr = ["label", "wait@wave", "swap@wave", "err%", "tok/s", "peak_kv"]
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join(["---"] * len(hdr)) + "|")
    else:
        print(f"{'label':>10} {'wait@':>6} {'swap@':>6} {'err%':>6} {'tok/s':>7} {'peak':>6}")
    for d in rows:
        fw = "-" if d["first_wait_wave"] is None else f"w{d['first_wait_wave']}"
        fs = "-" if d["first_swap_wave"] is None else f"w{d['first_swap_wave']}"
        if md:
            print(f"| {d['label']} | {fw} | {fs} | {d['error_rate']:.1%} | "
                  f"{d['mean_wave_tok_s']:.0f} | {d['peak_kv']:.0%} |")
        else:
            print(f"{d['label']:>10} {fw:>6} {fs:>6} {d['error_rate']:>5.1%} "
                  f"{d['mean_wave_tok_s']:>7.0f} {d['peak_kv']:>5.0%}")
    if plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for d in rows:
            tl = d["timeline"]
            ax.plot([e["wave"] for e in tl],
                    [e.get("num_requests_swapped", 0) for e in tl],
                    marker="o", label=d["label"])
        ax.set_xlabel("pressure wave")
        ax.set_ylabel("swapped (evicted) requests")
        ax.set_title("Eviction onset per block-size config")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    main()

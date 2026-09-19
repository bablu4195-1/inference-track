#!/usr/bin/env python3
"""P8 comparison: join per-config summaries -> starvation/TTFT tradeoff table.

Usage: python3 p08-chunked-prefill/compare.py results/ [--plot starvation.png]
Reads results/<label>-*/summary.json. Prints:
  label | ITL_base | ITL_mixed | starvation_x | hog_TTFT
Expected shape: starvation_x falls as budget shrinks (on-*), spikes for
off-16384; hog TTFT rises as budget shrinks. The crossing point is the
tuning answer for the workload.
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
        if "starvation_x" not in d:
            continue
        rows.append(d)
    return rows


def budget(label: str) -> int:
    try:
        return int(label.rsplit("-", 1)[1])
    except Exception:
        return 0


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    plot = None
    if "--plot" in sys.argv:
        plot = sys.argv[sys.argv.index("--plot") + 1]
    rows = sorted(load(results), key=lambda d: (d["label"].startswith("off"), budget(d["label"])))
    print(f"{'label':>10} {'ITL_base':>9} {'ITL_mixed':>9} {'starv_x':>8} {'hog_TTFT':>9}")
    for d in rows:
        print(f"{d['label']:>10} {d['itl_base_ms']:>9.1f} {d['itl_mixed_ms']:>9.1f} "
              f"{d['starvation_x']:>8.2f} {d['hog_ttft_ms']:>9.0f}")
    if plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        on = [d for d in rows if not d["label"].startswith("off") and budget(d["label"]) > 0]
        off = [d for d in rows if d["label"].startswith("off")]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        if on:
            ax1.plot([budget(d["label"]) for d in on],
                     [d["starvation_x"] for d in on], marker="o", label="chunked on")
        for d in off:
            ax1.axhline(d["starvation_x"], linestyle="--", label=f"{d['label']}")
        ax1.set_xscale("log", base=2)
        ax1.set_xlabel("max-num-batched-tokens")
        ax1.set_ylabel("starvation_x (mixed/base ITL)")
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)
        ax2.bar([d["label"] for d in rows], [d["hog_ttft_ms"] for d in rows])
        ax2.set_ylabel("hog TTFT ms")
        ax2.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    main()

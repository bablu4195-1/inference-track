#!/usr/bin/env python3
"""Plot load curves from a P2 results.csv.

Usage: python3 p02-bench/plot.py p02-bench/results/<run-id>/results.csv
Writes ttft.png, itl.png, throughput.png next to the CSV (Agg backend).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str):
    per_cfg: dict = defaultdict(lambda: {"ttft": [], "itl": [], "tps": [], "wave": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("ok") != "True":
                continue
            key = (int(r["prompt_len"]), int(r["gen_len"]), int(r["concurrency"]))
            d = per_cfg[key]
            d["ttft"].append(float(r["ttft_ms"]))
            d["itl"].append(float(r["itl_p95_ms"]))
            d["tps"].append(float(r["toks_per_s"]))
    return per_cfg


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def curve_plot(per_cfg, field: str, title: str, ylabel: str, out: Path) -> None:
    prompt_lens = sorted({k[0] for k in per_cfg})
    fig, ax = plt.subplots()
    for plen in prompt_lens:
        pts = sorted((k[2], pct(v[field], 95)) for k, v in per_cfg.items() if k[0] == plen)
        if pts:
            ax.plot([c for c, _ in pts], [v for _, v in pts], marker="o",
                    label=f"prompt {plen}")
    ax.set_xscale("log", base=2)
    ax.set_title(title)
    ax.set_xlabel("concurrency")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


def main() -> None:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
    per_cfg = load(str(csv_path))
    d = csv_path.parent
    curve_plot(per_cfg, "ttft", "TTFT p95 vs concurrency", "ms", d / "ttft.png")
    curve_plot(per_cfg, "itl", "ITL p95 vs concurrency", "ms", d / "itl.png")
    curve_plot(per_cfg, "tps", "Per-request throughput vs concurrency",
               "tok/s", d / "throughput.png")


if __name__ == "__main__":
    main()

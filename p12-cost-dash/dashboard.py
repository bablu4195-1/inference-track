#!/usr/bin/env python3
"""P12 static dashboard: ledger.json + cost-log.csv -> report.png.

Four panels: spend over time vs $128 budget, $/M blended per config,
MFU p50 per config, requests per config. Static PNG (no server) — the
Grafana-live version is a P15 stretch; numbers are what matter.

Usage: python3 p12-cost-dash/dashboard.py ledger.json [--plot report.png]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

BUDGET = 128.0


def spend_series():
    p = Path(__file__).resolve().parents[1] / "infra" / "aws" / "cost-log.csv"
    ts, cum, tot = [], [], 0.0
    with open(p) as f:
        for i, r in enumerate(csv.DictReader(f)):
            try:
                tot += float(r["cost_usd"])
            except (ValueError, KeyError):
                continue
            ts.append(i + 1)
            cum.append(tot)
    return ts, cum, tot


def main() -> None:
    ledger = json.loads(open(sys.argv[1]).read())
    out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--plot" else "report.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ledger["groups"]
    labels = [g["config"] or g["model"] for g in groups]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    ts, cum, tot = spend_series()
    axs[0, 0].plot(ts or [0], cum or [0], marker="o")
    axs[0, 0].axhline(BUDGET, color="r", linestyle="--", label="$128 budget")
    axs[0, 0].set_title(f"Cumulative GPU spend (${tot:.2f})")
    axs[0, 0].set_xlabel("cost-log entry")
    axs[0, 0].legend()

    axs[0, 1].bar(labels, [g["usd_per_M_blended"] for g in groups])
    axs[0, 1].set_title("$/M tokens (blended)")
    axs[0, 1].tick_params(axis="x", rotation=30)

    axs[1, 0].bar(labels, [g["mfu_p50_pct"] for g in groups])
    axs[1, 0].set_title("MFU p50 % (decode-bound: low is honest)")
    axs[1, 0].tick_params(axis="x", rotation=30)

    axs[1, 1].bar(labels, [g["requests"] for g in groups])
    axs[1, 1].set_title("Requests accounted")
    axs[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Cost per token — {ledger['model']} on {ledger['gpu']}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: dashboard.py ledger.json [--plot report.png]")
    main()

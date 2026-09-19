#!/usr/bin/env python3
"""P15 teardown join: three serving configs, one workload, one verdict table.

Configs (same model, same P2 full-sweep workload, same GPU type):
  A  fp16-colocated   FP16 baseline, single replica (runs/20260919-p2)
  B  awq-prefix       AWQ INT4 + prefix caching (runs/20260919-p5-full)
  C  fp8-spec         FP8 + speculative decode (runs/20260919-p6-full)

Each config dir holds summary.json (+ppl.json/vram for B/C). Output: per-key
latency/throughput/cost table with B/C deltas vs A, plus curves.

Usage: python3 p15-teardown/teardown.py A B C [--plot out.png] [--report]
Cost source: --cost-per-config USD "0.5,0.5,0.5" (from cost-log.csv notes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_summary(d: Path):
    cands = sorted(d.glob("**/summary.json"))
    if not cands:
        sys.exit(f"no summary.json under {d}")
    with open(cands[0]) as f:
        return json.load(f)


def load_ppl(d: Path):
    cands = sorted(d.glob("**/ppl.json"))
    if not cands:
        return None
    with open(cands[0]) as f:
        return json.load(f)["perplexity"]


def main() -> None:
    names = sys.argv[1:4]
    plot = sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None
    md = "--report" in sys.argv
    costs = [0.0, 0.0, 0.0]
    if "--cost-per-config" in sys.argv:
        costs = [float(x) for x in
                 sys.argv[sys.argv.index("--cost-per-config") + 1].split(",")]
    cfgs = [(n, Path(p), c) for n, p, c in zip(("A", "B", "C"), names, costs)]
    sums = {n: load_summary(d) for n, d, _ in cfgs}
    ppls = {n: load_ppl(d) for n, d, _ in cfgs}
    keys = sorted(k for k in sums["A"]
                  if isinstance(sums["A"][k], dict) and "ttft_ms" in sums["A"][k])
    if md:
        print("| scenario | A TTFT | B TTFT (Δ) | C TTFT (Δ) | A tps | B tps (Δ) | C tps (Δ) |")
        print("|---|---|---|---|---|---|---|")
    else:
        print(f"{'scenario':>14} {'A_TTFT':>7} {'B_d%':>7} {'C_d%':>7} "
              f"{'A_tps':>7} {'B_d%':>7} {'C_d%':>7}")
    for k in keys:
        a = sums["A"].get(k, {})
        row = [k, a.get("ttft_ms", {}).get("p50", 0),
               a.get("wave_tok_s", [0])[-1] if a.get("wave_tok_s") else 0]
        cells = []
        for n in ("B", "C"):
            b = sums[n].get(k, {})
            if not b:
                cells += ["-", "-"]
                continue
            dt = (b["ttft_ms"]["p50"] - row[1]) / max(row[1], 1e-9) * 100
            at = a.get("wave_tok_s", [0])
            bt = b.get("wave_tok_s", [0])
            dw = (bt[-1] - at[-1]) / max(at[-1], 1e-9) * 100 if at and bt else 0.0
            cells += [f"{dt:+.1f}%", f"{dw:+.1f}%"]
        if md:
            b, c = sums["B"].get(k, {}), sums["C"].get(k, {})
            bt = lambda d: (d.get("ttft_ms", {}).get("p50", 0), (d.get("wave_tok_s", [0]) or [0])[-1])
            # cells = [B_dt, B_dw, C_dt, C_dw]
            print(f"| {k} | {row[1]:.0f} | {bt(b)[0]:.0f} ({cells[0]}) | {bt(c)[0]:.0f} ({cells[2]}) | "
                  f"{row[2]:.0f} | {bt(b)[1]:.0f} ({cells[1]}) | {bt(c)[1]:.0f} ({cells[3]}) |")
        else:
            print(f"{k:>14} {row[1]:>7.0f} {cells[0]:>7} {cells[2]:>7} "
                  f"{row[2]:>7.0f} {cells[1]:>7} {cells[3]:>7}")
    print("perplexity:", {n: ppls[n] for n in ("A", "B", "C")},
          "| config cost USD:", dict(zip(("A", "B", "C"), costs)))
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(keys)))
        w = 0.25
        for i, n in enumerate(("A", "B", "C")):
            ys = []
            for k in keys:
                d = sums[n].get(k, {})
                ws = d.get("wave_tok_s", [0]) if isinstance(d, dict) else [0]
                ys.append(ws[-1] if ws else 0)
            ax = plt.gca()
            ax.bar([x + (i - 1) * w for x in xs], ys, width=w, label=n)
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylabel("wave tok/s")
        ax.legend()
        plt.gcf().tight_layout()
        plt.gcf().savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: teardown.py A_dir B_dir C_dir [--plot o.png] [--report] [--cost-per-config a,b,c]")
    main()

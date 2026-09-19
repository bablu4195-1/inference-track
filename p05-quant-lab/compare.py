#!/usr/bin/env python3
"""P5 comparison: join bench + perplexity + VRAM per config -> tradeoff matrix.

Usage: python3 p05-quant-lab/compare.py results/ [--plot tradeoff.png] [--report]
Reads results/<label>-*/{bench/<stamp>/summary.json, ppl/*/ppl.json, vram_mib.txt}.
Table: label | VRAM MiB | TTFT p50 | ITL95 p50 | wave tok/s | ppl | ppl delta%
Prints --report markdown for README.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def load(results: Path):
    rows = []
    for d in sorted(results.iterdir()):
        if not d.is_dir() or "-" not in d.name:
            continue
        try:
            sums = glob.glob(str(d / "bench" / "*" / "summary.json"))
            ppls = glob.glob(str(d / "ppl" / "*" / "ppl.json"))
            if not (sums and ppls):
                continue
            s = json.loads(open(sums[0]).read())
            p = json.loads(open(ppls[0]).read())
            tt, it, tp = [], [], []
            for key, v in s.items():
                if not isinstance(v, dict) or "ttft_ms" not in v:
                    continue
                tt.append(v["ttft_ms"]["p50"])
                it.append(v["itl_p95_ms"]["p50"])
                tp += v.get("wave_tok_s", [])
            vram = Path(d, "vram_mib.txt").read_text().strip() if (d / "vram_mib.txt").exists() else "?"
            rows.append({"label": d.name.rsplit("-", 1)[0],
                         "vram_mib": vram,
                         "ttft": sum(tt) / len(tt), "itl": sum(it) / len(it),
                         "tps": sum(tp) / len(tp) if tp else 0.0,
                         "ppl": p["perplexity"]})
        except Exception as e:
            print(f"skip {d.name}: {e}", file=sys.stderr)
    order = {"fp16": 0, "fp8": 1, "awq": 2}
    return sorted(rows, key=lambda r: order.get(r["label"], 9))


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    plot = sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None
    md = "--report" in sys.argv
    rows = load(results)
    base = next((r["ppl"] for r in rows if r["label"] == "fp16"), None)
    if md:
        print("| config | VRAM MiB | TTFT p50 | ITL95 p50 | tok/s | ppl | ppl Δ% |")
        print("|---|---|---|---|---|---|---|")
    else:
        print(f"{'config':>6} {'VRAM':>7} {'TTFT':>7} {'ITL95':>7} {'tok/s':>7} {'ppl':>7} {'d_ppl':>6}")
    for r in rows:
        dp = ((r["ppl"] - base) / base * 100) if base else 0.0
        if md:
            print(f"| {r['label']} | {r['vram_mib']} | {r['ttft']:.0f} | {r['itl']:.1f} | "
                  f"{r['tps']:.0f} | {r['ppl']:.2f} | {dp:+.1f}% |")
        else:
            print(f"{r['label']:>6} {str(r['vram_mib']):>7} {r['ttft']:>7.0f} "
                  f"{r['itl']:>7.1f} {r['tps']:>7.0f} {r['ppl']:>7.2f} {dp:>+5.1f}%")
    if plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar([r["label"] for r in rows], [r["ppl"] for r in rows])
        ax1.set_ylabel("perplexity (lower = better)")
        ax1.set_title("Quality by quantization")
        ax2.bar([r["label"] for r in rows], [r["tps"] for r in rows])
        ax2.set_ylabel("wave tok/s")
        ax2.set_title("Throughput by quantization")
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    main()

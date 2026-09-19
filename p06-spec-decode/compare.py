#!/usr/bin/env python3
"""P6 comparison: decode speedup at matched quality + acceptance per draft size.

Usage: python3 p06-spec-decode/compare.py results/ [--plot speedup.png] [--report]
Reads results/<label>-*/{bench/*/summary.json, acceptance.json}.
Table: label | tok/s | speedup vs baseline | ITL95 | acceptance
Quality is matched by construction (greedy, temp 0, same prompts) — spec
decoding is exact, so any tok/s gain is the free lunch. acceptance.py output
is the explanation, bench output is the proof.
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
            if not sums:
                continue
            s = json.loads(open(sums[0]).read())
            tp, it = [], []
            for v in s.values():
                if not isinstance(v, dict) or "toks_per_s" not in v:
                    continue
                tp += v.get("wave_tok_s", [])
                it.append(v["itl_p95_ms"]["p50"])
            acc = glob.glob(str(d / "acceptance.json"))
            ader = {}
            if acc:
                ader = json.loads(open(acc[0]).read()).get("derived", {})
            rate = (ader.get("acceptance_rate_gauge")
                    or ader.get("acceptance_derived"))
            rows.append({"label": d.name.rsplit("-", 1)[0],
                         "tps": sum(tp) / len(tp) if tp else 0.0,
                         "itl": sum(it) / len(it) if it else 0.0,
                         "acceptance": rate})
        except Exception as e:
            print(f"skip {d.name}: {e}", file=sys.stderr)
    order = {"baseline": 0, "spec-3": 1, "spec-5": 2, "spec-8": 3}
    return sorted(rows, key=lambda r: order.get(r["label"], 9))


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    plot = sys.argv[sys.argv.index("--plot") + 1] if "--plot" in sys.argv else None
    md = "--report" in sys.argv
    rows = load(results)
    base = next((r["tps"] for r in rows if r["label"] == "baseline"), 0.0)
    if md:
        print("| config | tok/s | speedup | ITL95 | acceptance |")
        print("|---|---|---|---|---|")
    else:
        print(f"{'config':>8} {'tok/s':>7} {'speedup':>8} {'ITL95':>7} {'accept':>7}")
    for r in rows:
        su = (r["tps"] / base) if base else 0.0
        ac = "-" if r["acceptance"] is None else f"{r['acceptance']:.0%}"
        if md:
            print(f"| {r['label']} | {r['tps']:.0f} | {su:.2f}x | {r['itl']:.1f} | {ac} |")
        else:
            print(f"{r['label']:>8} {r['tps']:>7.0f} {su:>7.2f}x {r['itl']:>7.1f} {ac:>7}")
    if plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar([r["label"] for r in rows],
               [(r["tps"] / base) if base else 0 for r in rows])
        ax.axhline(1.0, linestyle="--")
        ax.set_ylabel("decode speedup vs baseline")
        ax.set_title("Speculative decoding speedup by draft size")
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("wrote", plot)


if __name__ == "__main__":
    main()

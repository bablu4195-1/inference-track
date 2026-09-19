"""Shared latency-metrics helpers for the inference track.

Client-agnostic: callers collect (text, recv_timestamp) events from a
streaming response and pass them here. Used by P1 smoke, P2 harness,
and every later project (P4/P5/P6/P8/P10/P14).

Results schema (CSV header order):
  run_id, config, concurrency, prompt_len, gen_len,
  ttft_ms, itl_p50_ms, itl_p95_ms, e2e_ms, toks_per_s, ok, err
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass

CSV_HEADER = [
    "run_id", "config", "concurrency", "prompt_len", "gen_len",
    "ttft_ms", "itl_p50_ms", "itl_p95_ms", "e2e_ms", "toks_per_s",
    "ok", "err",
]


@dataclass
class RequestStats:
    ttft_ms: float          # send -> first token
    itl_ms: list[float]     # inter-token gaps after the first token
    e2e_ms: float           # send -> last token
    gen_tokens: int

    @property
    def itl_p50_ms(self) -> float:
        return _pct(self.itl_ms, 50)

    @property
    def itl_p95_ms(self) -> float:
        return _pct(self.itl_ms, 95)

    @property
    def toks_per_s(self) -> float:
        return self.gen_tokens / (self.e2e_ms / 1000) if self.e2e_ms > 0 else 0.0


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.quantiles(xs, n=100, method="inclusive")[int(p) - 1]


def compute_stats(send_t: float, token_times: list[float], gen_tokens: int) -> RequestStats:
    """Build RequestStats from a send timestamp and per-token arrival times."""
    if not token_times:
        return RequestStats(ttft_ms=0.0, itl_ms=[], e2e_ms=0.0, gen_tokens=0)
    ttft = (token_times[0] - send_t) * 1000
    itls = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    e2e = (token_times[-1] - send_t) * 1000
    return RequestStats(ttft_ms=ttft, itl_ms=itls, e2e_ms=e2e, gen_tokens=gen_tokens)


def summarize(values: list[float]) -> dict[str, float]:
    """p50/p95/mean/min/max summary for a list of per-request scalars."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "p50": _pct(values, 50),
        "p95": _pct(values, 95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def append_csv(path: str, rows: list[dict]) -> None:
    """Append result rows, writing the header if the file is new/empty."""
    import os
    need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if need_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})


def row(run_id: str, config: str, concurrency: int, prompt_len: int,
        stats: RequestStats, ok: bool = True, err: str = "") -> dict:
    return {
        "run_id": run_id, "config": config, "concurrency": concurrency,
        "prompt_len": prompt_len, "gen_len": stats.gen_tokens,
        "ttft_ms": round(stats.ttft_ms, 1),
        "itl_p50_ms": round(stats.itl_p50_ms, 2),
        "itl_p95_ms": round(stats.itl_p95_ms, 2),
        "e2e_ms": round(stats.e2e_ms, 1),
        "toks_per_s": round(stats.toks_per_s, 2),
        "ok": ok, "err": err,
    }


# Make dataclass usable without importing `dataclasses` elsewhere.
_ = asdict

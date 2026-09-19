#!/usr/bin/env python3
"""P3 live KV-cache monitor: polls vLLM /metrics, logs, alerts.

Tracks the gauges that precede production OOMs:
  vllm:kv_cache_usage_perc   fraction of KV blocks in use (primary signal)
  vllm:gpu_cache_usage_perc  fraction of GPU KV cache used
  vllm:num_requests_running  actively decoding
  vllm:num_requests_waiting  queued = scheduler pressure (leading indicator)
  vllm:num_requests_swapped  preempted by eviction (lagging indicator)

Usage:
  python3 p03-kvcalc/monitor.py --base-url http://<GPU>:8000 --duration-s 600
  python3 p03-kvcalc/monitor.py --base-url http://<GPU>:8000   # until Ctrl-C

Output: monitor-<utc>.csv + one-line status per poll, ALERT when KV > --alert.
Run it in a second terminal alongside pressure.py or any bench run.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
import urllib.request

KEYS = [
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
]


def fetch_metrics(base_url: str, timeout_s: float = 10.0) -> dict[str, float]:
    """Parse Prometheus text exposition from /metrics into {name: value}."""
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout_s) as r:
        text = r.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.partition(" ")
        name = name.split("{")[0]  # strip label sets
        if name.startswith("vllm:"):
            try:
                out[name] = float(val)
            except ValueError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="P3 KV-cache live monitor")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--interval-s", type=float, default=2.0)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="0 = run until Ctrl-C")
    ap.add_argument("--alert", type=float, default=0.90,
                    help="KV usage fraction that triggers ALERT")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = a.out or f"p03-kvcalc/monitor-{stamp}.csv"
    t0 = time.time()
    alerted = False
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s"] + KEYS)
            print(f"logging -> {path}  (Ctrl-C to stop)")
            while True:
                el = time.time() - t0
                if a.duration_s and el >= a.duration_s:
                    break
                try:
                    m = fetch_metrics(a.base_url)
                    w.writerow([f"{el:.1f}"] + [m.get(k, "") for k in KEYS])
                    f.flush()
                    kv = m.get("vllm:kv_cache_usage_perc", 0.0)
                    print(f"t={el:6.1f}s kv={kv:5.1%} "
                          f"run={m.get('vllm:num_requests_running', 0):.0f} "
                          f"wait={m.get('vllm:num_requests_waiting', 0):.0f} "
                          f"swapped={m.get('vllm:num_requests_swapped', 0):.0f}")
                    if kv >= a.alert and not alerted:
                        print(f"*** ALERT: KV usage {kv:.1%} >= {a.alert:.0%} "
                              f"— expect queueing, then eviction/OOM ***")
                        alerted = True
                    elif kv < a.alert - 0.05:
                        alerted = False
                except Exception as e:
                    print(f"t={el:6.1f}s metrics fetch failed: {e}")
                time.sleep(a.interval_s)
    except KeyboardInterrupt:
        pass
    print(f"done -> {path}")


if __name__ == "__main__":
    main()

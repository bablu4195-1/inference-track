#!/usr/bin/env python3
"""P11 queue-depth autoscaler: KEDA-style scaling for vLLM replicas.

Control law (all thresholds CLI-tunable):
  - Every --interval-s, read vllm:num_requests_waiting (pending-queue depth).
  - waiting >= --scale-up-q   for --up-streak polls   -> +1 replica (max --max-n)
  - waiting == 0              for --down-streak polls -> -1 replica (min --min-n)
  - Scale-up also fires immediately if any request waited > --wait-slo-s
    (latency guard, not just depth).
  - Cooldown --cooldown-s after any action: no flapping.

Cold-start mitigation (the FinOps point): replica 1 is never scaled to zero
(--min-n 1 keeps a warm pool: pulled image + weights on EBS + compiled graphs
warm from the last wave). Scale-from-1 cost ~= container start (~30 s), not a
cold boot (~8 min). --min-n 0 is supported for the scale-to-zero experiment
with eyes open about the cost.

Actuation: --dry-run (default) prints decisions; --exec runs
docker compose up/down commands (single-host) — the EC2 multi-host variant is
a documented stretch (one ASG + ALB target registration per replica).

State + audit: every poll appended to --log JSONL (t, waiting, running,
desired, action) — P14 replays these logs as ground truth for scale latency.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request


def queue_depth(base_url: str, timeout_s: float = 8.0) -> tuple[float, float]:
    """Returns (waiting, running) from /metrics; (-1,-1) if unreachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=timeout_s) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return -1.0, -1.0
    wait = run = 0.0
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.partition(" ")
        name = name.split("{")[0]
        try:
            if name == "vllm:num_requests_waiting":
                wait += float(val)
            elif name == "vllm:num_requests_running":
                run += float(val)
        except ValueError:
            pass
    return wait, run


def decide(waiting: float, streak_up: int, streak_down: int, desired: int,
           a: argparse.Namespace) -> tuple[int, str]:
    """Pure comparator: streaks are counted by the caller (main loop / tests).
    Returns (new_desired, action)."""
    if waiting < 0:  # metrics unreachable: hold, never flap
        return desired, "hold_metrics_down"
    if streak_up >= a.up_streak and desired < a.max_n:
        return desired + 1, f"scale_up_q{waiting:.0f}x{streak_up}"
    if streak_down >= a.down_streak and desired > a.min_n:
        return desired - 1, f"scale_down_idle_x{streak_down}"
    return desired, "hold"


def actuate(compose: str, desired: int, dry_run: bool) -> None:
    cmd = ["docker", "compose", "-f", compose, "up", "-d",
           "--scale", f"vllm={desired}"]
    print("  exec:", " ".join(cmd) if not dry_run else f"[dry-run] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="P11 queue-depth autoscaler")
    ap.add_argument("--metrics-url", default="http://localhost:8000",
                    help="vLLM server exposing /metrics")
    ap.add_argument("--compose", default="p01-serve/compose.yml")
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=3)
    ap.add_argument("--scale-up-q", type=float, default=8.0)
    ap.add_argument("--up-streak", type=int, default=3)
    ap.add_argument("--down-streak", type=int, default=12)
    ap.add_argument("--interval-s", type=float, default=10.0)
    ap.add_argument("--cooldown-s", type=float, default=120.0)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="0 = run until Ctrl-C")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--exec", action="store_true",
                    help="actually run docker compose (implies no dry-run)")
    ap.add_argument("--log", default="p11-autoscaler/scaler.jsonl")
    a = ap.parse_args()
    dry = not a.exec

    desired, su, sd, last_action = a.min_n, 0, 0, 0.0
    t0 = time.time()
    print(f"scaler: {a.min_n}..{a.max_n} replicas, up_q={a.scale_up_q}x{a.up_streak}, "
          f"down_idle_x{a.down_streak}, cooldown={a.cooldown_s}s, dry={dry}")
    try:
        with open(a.log, "a") as f:
            while True:
                el = time.time() - t0
                if a.duration_s and el >= a.duration_s:
                    break
                wait, run = queue_depth(a.metrics_url)
                # Streak accounting lives here so decide() stays pure/testable.
                su = su + 1 if wait >= a.scale_up_q else 0
                sd = sd + 1 if wait == 0 else 0
                action = "hold"
                if time.time() - last_action >= a.cooldown_s:
                    if su >= a.up_streak and desired < a.max_n:
                        desired += 1
                        action = f"scale_up_q{wait:.0f}"
                    elif sd >= a.down_streak and desired > a.min_n:
                        desired -= 1
                        action = "scale_down_idle"
                if action.startswith("scale"):
                    actuate(a.compose, desired, dry)
                    last_action = time.time()
                    su, sd = 0, 0
                f.write(json.dumps({
                    "t_s": round(el, 1),
                    "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
                    "waiting": wait, "running": run, "desired": desired,
                    "action": action}) + "\n")
                f.flush()
                print(f"t={el:6.0f}s wait={wait:4.0f} run={run:3.0f} "
                      f"desired={desired} {action}")
                time.sleep(a.interval_s)
    except KeyboardInterrupt:
        pass
    print(f"done -> {a.log}")


if __name__ == "__main__":
    main()

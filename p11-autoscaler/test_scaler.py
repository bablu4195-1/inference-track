#!/usr/bin/env python3
"""P11 scaler unit tests: control law + streak/cooldown behavior. No GPU.

Usage: python3 p11-autoscaler/test_scaler.py
Covers: scale-up after streak, no flap on single spike, scale-down after idle,
min/max clamps, metrics-down hold, TokenBucket-free pure decide() paths.
Fakes /metrics via stdlib server for queue_depth() parsing incl. labels.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaler import decide, queue_depth  # noqa: E402

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    PASS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(f"FAILED: {name} {detail}")


def ns(**kw) -> argparse.Namespace:
    d = dict(scale_up_q=8.0, up_streak=3, down_streak=4, min_n=1, max_n=3)
    d.update(kw)
    return argparse.Namespace(**d)


A = ns()

# decide() pure-function tests (streaks passed in explicitly)
check("hold below threshold", decide(7, 0, 0, 1, A) == (1, "hold"))
check("no flap on single spike", decide(30, 1, 0, 1, A) == (1, "hold"))
check("scale up on streak", decide(9, 3, 0, 1, A)[0] == 2)
check("scale-up action string", decide(9, 3, 0, 1, A)[1].startswith("scale_up"))
check("max clamp", decide(99, 9, 0, 3, A) == (3, "hold"))
check("scale down on idle", decide(0, 0, 4, 3, A)[0] == 2)
check("min clamp", decide(0, 0, 9, 1, A) == (1, "hold"))
check("metrics-down holds", decide(-1, 9, 0, 2, A) == (2, "hold_metrics_down"))
check("idle resets up-streak effect", decide(0, 0, 0, 2, A) == (2, "hold"))


# queue_depth() parsing incl. vLLM label sets
class H(BaseHTTPRequestHandler):
    BODY = ('vllm:num_requests_waiting{engine="0",model_name="m"} 5.0\n'
            'vllm:num_requests_running 3.0\n'
            'http_requests_total 9.0\n')

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = self.BODY.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
check("metrics parse w/ labels", queue_depth(
    f"http://127.0.0.1:{srv.server_port}") == (5.0, 3.0))
check("metrics down -> (-1,-1)", queue_depth("http://127.0.0.1:1") == (-1.0, -1.0))


# streak integration: feed decide() a spike-then-sustain sequence
def run_seq(seq):
    desired, su, sd = 1, 0, 0
    acts = []
    for w in seq:
        su = su + 1 if w >= A.scale_up_q else 0
        sd = sd + 1 if w == 0 else 0
        desired, act = decide(w, su, sd, desired, A)
        acts.append(act)
        if act.startswith("scale"):
            su, sd = 0, 0
    return desired, acts


d, acts = run_seq([0, 0, 30, 0, 0])  # lone spike amid idle
check("lone spike never scales", d == 1, str(acts))
d, acts = run_seq([9, 9, 9, 9])      # sustained pressure
check("sustained pressure scales", d == 2, str(acts))
d, acts = run_seq([9, 9, 9, 0, 0, 0, 0])  # pressure then idle
check("idle after pressure scales down", d == 1, str(acts))

print(f"\nALL {len(PASS)} SCALER TESTS PASS")

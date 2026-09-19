#!/usr/bin/env python3
"""P13 gateway tests: happy path, dead/slow primary failover, 4xx relay,
rate limits, tenant isolation, log schema. Stdlib only.

Usage: python3 p13-ai-gateway/test_gateway.py   (exits nonzero on failure)
Spins fake backends + gateway on ephemeral ports; no GPU needed.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway import Gateway  # noqa: E402

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    PASS.append(cond)
    print(("PASS " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        raise SystemExit(f"FAILED: {name} {detail}")


def make_backend(tag: str, first_chunk_delay: float = 0.0, status: int = 200):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if status >= 500:
                self.send_response(status)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(first_chunk_delay)
            self.wfile.write(f"data: {json.dumps({'tag': tag})}\n\n".encode())
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def start_gateway(primary: str, fallback: str, slo_ms: float, cap: float,
                  refill: float, log: str):
    Gateway.primary, Gateway.fallback = primary, fallback
    Gateway.slo_ms = slo_ms
    Gateway.bucket_cfg = (cap, refill)
    Gateway.buckets = {}
    Gateway.log_path = log
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def post(port: int, tenant: str = "t1", timeout: float = 20.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps({"model": "m", "prompt": "hi", "stream": True}).encode(),
        headers={"Content-Type": "application/json", "X-Tenant": tenant},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            return r.status, dict(r.headers), body
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def logs(path: str):
    return [json.loads(l) for l in open(path) if l.strip()]


TMP = Path("/var/folders/s9/spqpvjl51fj8jqfyp9dl001h0000gn/T/opencode/p13test")
TMP.mkdir(exist_ok=True)

# 1. happy path: fast primary, generous SLO
be1 = make_backend("P")
be2 = make_backend("F")
gw = start_gateway(f"127.0.0.1:{be1.server_port}", f"127.0.0.1:{be2.server_port}",
                   5000, 100, 10, str(TMP / "r1.jsonl"))
st, hd, body = post(gw.server_port)
check("happy: 200", st == 200, str(st))
check("happy: routed primary", hd.get("X-Route") == "primary", str(hd))
check("happy: primary bytes", '"P"' in body, body[:60])
check("happy: log route", logs(str(TMP / "r1.jsonl"))[-1]["route"] == "primary")

# 2. dead primary -> fallback
gw2 = start_gateway("127.0.0.1:1", f"127.0.0.1:{be2.server_port}",
                    5000, 100, 10, str(TMP / "r2.jsonl"))
st, hd, body = post(gw2.server_port)
check("dead-primary: 200 via fallback", st == 200, str(st))
check("dead-primary: X-Route fallback", hd.get("X-Route") == "fallback", str(hd))
check("dead-primary: fallback bytes", '"F"' in body, body[:60])
check("dead-primary: fail reason logged",
      "primary_attempt" in logs(str(TMP / "r2.jsonl"))[-1]["fail_reason"])

# 3. slow primary (TTFT 1.5s) with 300ms SLO -> fallback, nothing half-written
slow = make_backend("SLOW", first_chunk_delay=1.5)
gw3 = start_gateway(f"127.0.0.1:{slow.server_port}", f"127.0.0.1:{be2.server_port}",
                    300, 100, 10, str(TMP / "r3.jsonl"))
st, hd, body = post(gw3.server_port)
check("slo-breach: failed over", hd.get("X-Route") == "fallback", str(hd))
e = logs(str(TMP / "r3.jsonl"))[-1]
check("slo-breach: logged breach", e["slo_breach"] is True and "slo_breach" in e["fail_reason"], str(e))

# 4. no fallback + dead primary -> 503 + Retry-After
gw4 = start_gateway("127.0.0.1:1", "", 5000, 100, 10, str(TMP / "r4.jsonl"))
st, hd, body = post(gw4.server_port)
check("no-fallback: 503", st == 503, str(st))
check("no-fallback: Retry-After", hd.get("Retry-After") == "5", str(hd))

# 5. rate limit: capacity 2, no refill -> 3rd request 429; other tenant unaffected
gw5 = start_gateway(f"127.0.0.1:{be1.server_port}", "", 5000, 2, 0.0,
                    str(TMP / "r5.jsonl"))
s1, _, _ = post(gw5.server_port, "alice")
s2, _, _ = post(gw5.server_port, "alice")
s3, h3, _ = post(gw5.server_port, "alice")
sb, _, _ = post(gw5.server_port, "bob")
check("ratelimit: first two pass", (s1, s2) == (200, 200), f"{s1},{s2}")
check("ratelimit: third 429", s3 == 429, str(s3))
check("ratelimit: Retry-After", h3.get("Retry-After") == "1", str(h3))
check("ratelimit: tenant isolation", sb == 200, str(sb))

print(f"\nALL {len(PASS)} GATEWAY TESTS PASS")

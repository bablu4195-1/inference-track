#!/usr/bin/env python3
"""P13 AI gateway: SLO-aware routing, degradation chain, per-tenant rate limits.

Chain: primary (self-hosted vLLM) -> fallback (second model / API provider /
stub) -> 503 + Retry-After. Provider outages are guaranteed; user-facing
errors are optional.

Policy:
  - Failover triggers: connection error, HTTP 5xx, or first-chunk time
    exceeding --slo-ms (TTFT SLO enforced as a deadline, not a dashboard).
    HTTP 4xx is a client bug -> relayed, never failed over.
  - One immediate retry on primary for connection-level errors, then failover.
  - Token bucket per tenant (X-Tenant header, else "default"): 429 with
    Retry-After when empty. Buckets are per-tenant isolated.
  - Every request appended to --log JSONL: tenant, route, fail_reason,
    ttft_ms, slo_breach — the tenant key feeds P12 accounting directly.

Stdlib only. Streaming relay preserves TTFT on the happy path.

Usage:
  python3 p13-ai-gateway/gateway.py --primary 127.0.0.1:8000 \\
      --fallback 127.0.0.1:8001 --slo-ms 2000 --rate-capacity 60 --rate-refill 1
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class TokenBucket:
    def __init__(self, capacity: float, refill_per_s: float):
        self.cap = capacity
        self.refill = refill_per_s
        self.tokens = capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> tuple[bool, float]:
        """Consume 1 token. Returns (ok, remaining)."""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.cap, self.tokens + (now - self.ts) * self.refill)
            self.ts = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, self.tokens
            return False, self.tokens


class Gateway(BaseHTTPRequestHandler):
    primary: str = ""
    fallback: str = ""
    slo_ms: float = 2000.0
    buckets: dict[str, TokenBucket] = {}
    bucket_cfg: tuple[float, float] = (60.0, 1.0)
    tenant_header: str = "X-Tenant"
    log_path: str = ""
    log_lock = threading.Lock()

    def log_message(self, *a):
        pass

    # -- helpers -----------------------------------------------------------
    def _tenant(self) -> str:
        return (self.headers.get(self.tenant_header) or "default")[:64]

    def _bucket(self, tenant: str) -> TokenBucket:
        b = self.buckets.get(tenant)
        if b is None:
            cap, refill = self.bucket_cfg
            b = TokenBucket(cap, refill)
            self.buckets[tenant] = b
        return b

    def _log(self, **kw) -> None:
        if not self.log_path:
            return
        with self.log_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(kw) + "\n")

    def _send_json(self, code: int, obj: dict, extra: dict | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def relay(self, backend: str, body: bytes, first_timeout_s: float,
              total_timeout_s: float = 600.0) -> tuple[int, str, float, bytes]:
        """Forward to backend, streaming. Returns (status, ctype, ttft_ms, head).

        Reads response HEAD (up to first 64KB incl. first chunk) with the SLO
        deadline, then streams the REST. Raises on connection/5xx/timeout.
        """
        host, _, port = backend.partition(":")
        conn = http.client.HTTPConnection(host, int(port or 80),
                                          timeout=first_timeout_s)
        try:
            conn.request("POST", self.path, body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status >= 500:
                raise RuntimeError(f"backend {resp.status}")
            ctype = resp.headers.get("Content-Type", "application/json")
            t0 = time.perf_counter()
            head = resp.read(65536)
            ttft = (time.perf_counter() - t0) * 1000 + 0.0
            # NOTE: ttft here excludes connection setup (measured post-status);
            # proxy-side TTFT incl. connect is logged separately as connect_ms.
            return resp.status, ctype, ttft, head, resp, conn
        except Exception:
            conn.close()
            raise

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            return self._send_json(200, {"gateway": "up", "primary": self.primary,
                                         "fallback": self.fallback or None})
        return self._send_json(404, {"error": "POST to /v1/* or GET /health"})

    def do_POST(self):
        t_start = time.perf_counter()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        tenant = self._tenant()
        ok, remaining = self._bucket(tenant).take()
        if not ok:
            self._log(ts=_ts(), tenant=tenant, path=self.path, route="rate_limited",
                      fail_reason="bucket_empty", ttft_ms=-1, slo_breach=False,
                      prompt_chars=len(body), status=429)
            return self._send_json(429, {"error": "tenant rate limit exceeded"},
                                    {"Retry-After": "1",
                                     "X-Tenant-Remaining": f"{remaining:.1f}"})
        route, reason, ttft, breach, status = self._serve(body)
        breach = bool(breach or "slo_breach" in (reason or ""))  # chain-level flag
        connect_ms = (time.perf_counter() - t_start) * 1000
        self._log(ts=_ts(), tenant=tenant, path=self.path, route=route,
                  fail_reason=reason, ttft_ms=round(ttft, 1),
                  slo_breach=breach, prompt_chars=len(body), status=status,
                  overhead_ms=round(connect_ms - max(ttft, 0), 1))
        # Response already relayed (or error sent) inside _serve.

    def _serve(self, body: bytes):
        """Attempt primary (+1 retry), else fallback, else 503. Returns
        (route, reason, ttft_ms, slo_breach, status). Relays bytes as it goes."""
        slo_s = self.slo_ms / 1000.0
        last_err = "unattempted"
        for attempt in (1, 2):  # primary + one retry on connection errors
            try:
                return self._relay_from(self.primary, body, slo_s, "primary", "")
            except (ConnectionRefusedError, socket.timeout, TimeoutError,
                    http.client.HTTPException, OSError) as e:
                last_err = f"primary_attempt{attempt}:{type(e).__name__}"
                continue
            except RuntimeError as e:  # 5xx / TTFT-breach style failures
                last_err = f"primary:{e}"
                break
        if self.fallback:
            try:
                return self._relay_from(self.fallback, body, slo_s * 2,
                                        "fallback", last_err)
            except Exception as e:
                last_err = f"{last_err} fallback:{type(e).__name__}"
        self._send_json(503, {"error": "all backends unavailable", "detail": last_err},
                        {"Retry-After": "5"})
        return "error", last_err, -1.0, True, 503

    def _relay_from(self, backend: str, body: bytes, first_timeout_s: float,
                    route: str, reason: str):
        t0 = time.perf_counter()
        try:
            status, ctype, stream_ms, head, resp, conn = self.relay(
                backend, body, first_timeout_s)
        except (socket.timeout, TimeoutError):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms >= self.slo_ms * 0.9:
                # Read deadline == SLO: a timeout here IS a TTFT breach.
                # No retry (primary already proved slow) -> immediate failover.
                raise RuntimeError(
                    f"ttft_slo_breach {elapsed_ms:.0f}ms >= {self.slo_ms:.0f}ms")
            raise
        ttft = (time.perf_counter() - t0) * 1000
        breach = ttft > self.slo_ms and route == "primary"
        if breach:
            # SLO missed on primary: abandon and fail over instead of serving
            # a late first token. Logged; client sees fallback latency.
            try:
                conn.close()
            finally:
                raise RuntimeError(f"ttft_slo_breach {ttft:.0f}ms > {self.slo_ms:.0f}ms")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("X-Route", route)
        self.send_header("X-TTFT-ms", f"{ttft:.1f}")
        self.end_headers()
        if head:
            self.wfile.write(head)
            self.wfile.flush()
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        conn.close()
        return route, reason, ttft, breach, status


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser(description="P13 AI gateway")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--primary", required=True, help="host:port of vLLM server")
    ap.add_argument("--fallback", default="",
                    help="host:port of fallback (empty = 503 when primary fails)")
    ap.add_argument("--slo-ms", type=float, default=2000.0)
    ap.add_argument("--rate-capacity", type=float, default=60.0)
    ap.add_argument("--rate-refill", type=float, default=1.0,
                    help="tokens/sec per tenant")
    ap.add_argument("--tenant-header", default="X-Tenant")
    ap.add_argument("--log", default="p13-ai-gateway/requests.jsonl")
    a = ap.parse_args()
    Gateway.primary = a.primary
    Gateway.fallback = a.fallback
    Gateway.slo_ms = a.slo_ms
    Gateway.bucket_cfg = (a.rate_capacity, a.rate_refill)
    Gateway.tenant_header = a.tenant_header
    Gateway.log_path = a.log
    print(f"gateway :{a.port} primary={a.primary} "
          f"fallback={a.fallback or '(none)'} slo={a.slo_ms:.0f}ms "
          f"rate={a.rate_capacity:g}/{a.rate_refill:g}s log={a.log}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Gateway).serve_forever()


if __name__ == "__main__":
    main()

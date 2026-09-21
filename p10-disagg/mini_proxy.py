#!/usr/bin/env python3
"""P10 minimal pull-mode disagg proxy (stdlib only).

Why this exists: the vLLM example proxy overwrites kv_transfer_params
remote_host with its own connection hostname (localhost), which the decode
worker — in another net namespace — then dials and fails. This proxy passes
the prefill's OWN advertisement (routable bridge IP) through untouched.

Flow per request:
  1. POST prompt to prefill with max_tokens=1 + kv_transfer_params
     {do_remote_decode: True} -> captures returned kv_transfer_params.
  2. POST full request to decode with those params verbatim.
  3. Stream decode's response back to the client.

Usage: python3 p10-disagg/mini_proxy.py --port 8000 \\
           --prefill http://127.0.0.1:8001 --decode http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Mini(BaseHTTPRequestHandler):
    prefill: str = ""
    decode: str = ""
    model: str = ""

    def log_message(self, *a):
        pass

    def _post(self, url: str, payload: dict, timeout_s: float = 600):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, r.read()

    def _send_json(self, code: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/status"):
            alive = {}
            for name, base in (("prefill", self.prefill), ("decode", self.decode)):
                try:
                    urllib.request.urlopen(base + "/health", timeout=5)
                    alive[name] = "up"
                except Exception as e:
                    alive[name] = f"down: {e}"[:80]
            code = 200 if all(v == "up" for v in alive.values()) else 503
            return self._send_json(code, {"proxy": "up", "workers": alive})
        return self._send_json(404, {"error": "POST /v1/* or GET /health"})

    def do_POST(self):
        t0 = time.perf_counter()
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send_json(400, {"error": "invalid JSON"})
        try:
            # 1. Prefill leg: 1 token, keep KV for remote decode.
            # Force non-streaming: callers (bench) send stream:true, but this
            # leg parses JSON (P10 lesson: SSE body -> json crash -> 502).
            p = dict(req)
            p["max_tokens"] = 1
            p["stream"] = False
            p.pop("stream_options", None)
            p["kv_transfer_params"] = {"do_remote_decode": True,
                                       "do_remote_prefill": False}
            st, body = self._post(self.prefill + self.path, p)
            p_resp = json.loads(body.decode())
            kvp = (p_resp.get("kv_transfer_params")
                   or p_resp["choices"][0].get("kv_transfer_params"))
            if not kvp:
                return self._send_json(502, {"error": "prefill gave no kv_transfer_params"})
            # 2. Decode leg: params verbatim (prefill's own routable remote_host).
            # Non-streaming: this proxy returns one JSON body (streaming
            # fan-out is a P13-shaped problem, not P10's).
            d = dict(req)
            d["stream"] = False
            d.pop("stream_options", None)
            d["kv_transfer_params"] = kvp
            st, body = self._post(self.decode + self.path, d)
            out = json.loads(body.decode())
            dt = (time.perf_counter() - t0) * 1000
            print(f"disagg ok: prefill+decode in {dt:.0f}ms "
                  f"remote={kvp.get('remote_host')}:{kvp.get('remote_port')}", flush=True)
            return self._send_json(200, out)
        except Exception as e:
            return self._send_json(502, {"error": f"disagg failed: {e}"[:300]})


def main() -> None:
    ap = argparse.ArgumentParser(description="P10 minimal pull-mode proxy")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    a = ap.parse_args()
    Mini.prefill, Mini.decode = a.prefill, a.decode
    print(f"mini-proxy :{a.port} prefill={a.prefill} decode={a.decode}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), Mini).serve_forever()


if __name__ == "__main__":
    main()

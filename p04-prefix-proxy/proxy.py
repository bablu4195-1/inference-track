#!/usr/bin/env python3
"""P4 prefix-caching proxy: consistent-hash on shared prompt prefix -> replica.

Why: vLLM prefix caching reuses KV blocks only when the SAME replica sees the
same prefix. Naive round-robin sprays shared system prompts across replicas
and destroys hit rate. This proxy hashes the cacheable prefix (system prompt /
prompt head) so same-prefix requests land on the same backend.

Zero dependencies (stdlib only) — runs on the Mac, the GPU host, anywhere.

Routing key extraction:
  /v1/chat/completions -> messages[0].content if role == 'system', else ''
  /v1/completions      -> first --prefix-chars of prompt (default 2000)
Backend = sorted(backends)[md5(key) % n]. Empty key -> least-recent backend
(fallback spread; logged with key_hash=null).

Streaming: body forwarded verbatim, response relayed chunk-by-chunk so TTFT
is preserved. Each request appended to --log as JSONL:
  {ts, backend, key_hash, route, prompt_chars, ttft_ms_proxy}

Usage:
  python3 p04-prefix-proxy/proxy.py --backends 127.0.0.1:8001,127.0.0.1:8002
  # then point bench/workload at http://localhost:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Proxy(BaseHTTPRequestHandler):
    backends: list[str] = []
    prefix_chars: int = 2000
    log_path: str = ""
    rr_counter: int = 0

    def log_message(self, *a):
        pass

    def _pick(self, key: str) -> tuple[str, str | None]:
        if key:
            h = hashlib.md5(key.encode()).hexdigest()
            return sorted(self.backends)[int(h, 16) % len(self.backends)], h[:12]
        # No shared prefix: round-robin so stateless traffic still balances.
        Proxy.rr_counter += 1
        return sorted(self.backends)[Proxy.rr_counter % len(self.backends)], None

    def _route_key(self, path: str, body: bytes) -> str:
        try:
            data = json.loads(body or b"{}")
        except Exception:
            return ""
        if "chat/completions" in path:
            msgs = data.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                return str(msgs[0].get("content", ""))
            return ""
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        return str(prompt)[: self.prefix_chars]

    def _send_json(self, code: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            status = {}
            for b in self.backends:
                try:
                    urllib.request.urlopen(f"http://{b}/health", timeout=3)
                    status[b] = "up"
                except Exception as e:
                    status[b] = f"down: {e}"[:100]
            return self._send_json(200, {"proxy": "up", "backends": status})
        return self._send_json(404, {"error": "use POST /v1/completions or /v1/chat/completions"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        backend, khash = self._pick(self._route_key(self.path, body))
        t0 = time.perf_counter()
        first_ms = -1.0
        try:
            req = urllib.request.Request(
                f"http://{backend}{self.path}", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                ctype = resp.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ctype)
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    if first_ms < 0:
                        first_ms = (time.perf_counter() - t0) * 1000
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            if first_ms < 0:
                return self._send_json(502, {"error": f"backend {backend}: {e}"[:200]})
        finally:
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "backend": backend, "key_hash": khash,
                        "prompt_chars": len(body),
                        "ttft_ms_proxy": round(first_ms, 1)}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 prefix-caching proxy")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True,
                    help="comma-separated host:port list")
    ap.add_argument("--prefix-chars", type=int, default=2000)
    ap.add_argument("--log", default="p04-prefix-proxy/routes.jsonl")
    a = ap.parse_args()
    Proxy.backends = [b.strip() for b in a.backends.split(",") if b.strip()]
    Proxy.prefix_chars = a.prefix_chars
    Proxy.log_path = a.log
    assert len(Proxy.backends) >= 1, "need at least one backend"
    print(f"proxy :{a.port} -> {sorted(Proxy.backends)}  log={a.log}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Proxy).serve_forever()


if __name__ == "__main__":
    main()

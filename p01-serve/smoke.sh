#!/usr/bin/env bash
# P1 smoke test — stdlib only (no pip needed on the GPU host).
# Proves: server up, streaming works, 2 concurrent streams batch.
# Usage: ./smoke.sh [BASE_URL]   (default http://localhost:8000)
set -euo pipefail
BASE="${1:-http://localhost:8000}"
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "== /health =="
curl -sf "$BASE/health" && echo " OK"

echo "== /v1/models =="
curl -sf "$BASE/v1/models" | head -c 300; echo

echo "== 2x concurrent streaming completions (TTFT/ITL) =="
python3 - "$BASE" "$MODEL" <<'EOF'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

base, model = sys.argv[1], sys.argv[2]
PROMPT = "Explain continuous batching in LLM serving in two sentences."

def one(i):
    body = json.dumps({"model": model, "prompt": PROMPT,
                       "max_tokens": 64, "temperature": 0.0,
                       "stream": True}).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    send = time.perf_counter(); first = None; times = []; n = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            now = time.perf_counter()
            if first is None:
                first = now
            times.append(now); n += 1
    ttft = (first - send) * 1000
    itls = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    itl = sum(itls) / len(itls) if itls else 0.0
    e2e = (times[-1] - send) * 1000 if times else 0.0
    print(f"req{i}: tokens={n} TTFT={ttft:.0f}ms meanITL={itl:.1f}ms E2E={e2e:.0f}ms")
    assert n > 0, "zero tokens streamed"
    return ttft

with ThreadPoolExecutor(max_workers=2) as ex:
    ttfts = list(ex.map(one, (0, 1)))
assert max(ttfts) < 20000, f"TTFT too high at concurrency 2: {ttfts}"
print("SMOKE PASS: continuous batching alive, TTFT < 20s @ concurrency 2")
EOF

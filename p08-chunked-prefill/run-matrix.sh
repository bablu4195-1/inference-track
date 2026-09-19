#!/usr/bin/env bash
# P8 config matrix runner — ONE GPU boot, sequential server restarts (~1h, ~$0.55).
# Scheduler flags are startup-time, so each config = compose restart + experiment.
# Per-config flags ship via a generated override file (compose `command:` is
# replaced wholesale on merge) — no edits to p01-serve/compose.yml needed.
# Usage (on the GPU host, p01-serve/.env in place):
#   ./p08-chunked-prefill/run-matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="http://localhost:8000"
OVERRIDE="$(pwd)/p08-chunked-prefill/.override.yml"

wait_healthy() {
  for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  echo "server never became healthy"; docker logs --tail 50 vllm-qwen25-7b; exit 1
}

run_config() {
  local label="$1"; shift
  echo "===== CONFIG $label : $* ====="
  {
    echo "services:"
    echo "  vllm:"
    echo "    command:"
    echo "      - \"--model=Qwen/Qwen2.5-7B-Instruct\""
    echo "      - \"--host=0.0.0.0\""
    echo "      - \"--port=8000\""
    echo "      - \"--dtype=auto\""
    echo "      - \"--gpu-memory-utilization=0.9\""
    echo "      - \"--max-model-len=8192\""
    echo "      - \"--max-num-seqs=128\""
    echo "      - \"--enable-prefix-caching\""
    echo "      - \"--trust-remote-code\""
    for flag in "$@"; do echo "      - \"$flag\""; done
  } > "$OVERRIDE"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" up -d --force-recreate
  wait_healthy
  python3 p08-chunked-prefill/experiment.py --base-url "$BASE" --label "$label"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" down
  rm -f "$OVERRIDE"
}

# Chunked ON, budget sweep (small budget = kinder decodes, slower hog TTFT).
run_config on-2048  --enable-chunked-prefill --max-num-batched-tokens 2048
run_config on-4096  --enable-chunked-prefill --max-num-batched-tokens 4096
run_config on-8192  --enable-chunked-prefill --max-num-batched-tokens 8192
run_config on-16384 --enable-chunked-prefill --max-num-batched-tokens 16384
# Chunked OFF (needs budget > max-model-len 8192): expect decode starvation.
run_config off-16384 --no-enable-chunked-prefill --max-num-batched-tokens 16384

echo "===== MATRIX DONE — comparing ====="
python3 p08-chunked-prefill/compare.py p08-chunked-prefill/results \
  --plot p08-chunked-prefill/results/starvation.png

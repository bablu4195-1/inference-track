#!/usr/bin/env bash
# P9 block-size matrix — ONE GPU boot, 3 sequential restarts (~45 min, ~$0.40).
# Same override-file pattern as P8 (compose `command:` replaced wholesale).
# Usage (on the GPU host, p01-serve/.env in place):
#   ./p09-paged-attn/run-matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="http://localhost:8000"
OVERRIDE="$(pwd)/p09-paged-attn/.override.yml"

wait_healthy() {
  for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  echo "server never became healthy"; docker logs --tail 50 vllm-qwen25-7b; exit 1
}

run_config() {
  local label="$1" bs="$2"
  echo "===== CONFIG $label (block-size $bs) ====="
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
    echo "      - \"--max-num-seqs=64\""
    echo "      - \"--enable-prefix-caching\""
    echo "      - \"--enable-chunked-prefill\""
    echo "      - \"--max-num-batched-tokens=8192\""
    echo "      - \"--block-size=$bs\""
    echo "      - \"--trust-remote-code\""
  } > "$OVERRIDE"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" up -d --force-recreate
  wait_healthy
  python3 p03-kvcalc/monitor.py --base-url "$BASE" --duration-s 1 --out /dev/null >/dev/null 2>&1 || true
  python3 p09-paged-attn/pressure.py --base-url "$BASE" --label "$label"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" down
  rm -f "$OVERRIDE"
}

run_config bs8  8
run_config bs16 16
run_config bs32 32

echo "===== MATRIX DONE — analyzing ====="
python3 p09-paged-attn/analyze.py p09-paged-attn/results \
  --plot p09-paged-attn/results/evictions.png --report

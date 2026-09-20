#!/usr/bin/env bash
# P5 serve matrix — ONE GPU boot, 3 sequential configs (~1.5h spot, ~$0.80).
# Per config: serve -> VRAM-at-rest -> P2 quick bench -> perplexity probe -> down.
# Weights note: FP16/FP8 pull Qwen2.5-7B-Instruct once (shared); AWQ pulls the
# -AWQ repo once (~4GB). All persist on /mnt/hf across configs.
# Usage (on the GPU host, p01-serve/.env in place):
#   ./p05-quant-lab/serve-matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="http://localhost:8000"
OVERRIDE="$(pwd)/p05-quant-lab/.override.yml"

wait_healthy() {
  for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  echo "server never became healthy"; docker logs --tail 50 vllm-qwen25-7b; exit 1
}

serve_and_measure() {
  local label="$1" model="$2"; shift 2
  echo "===== CONFIG $label ($model) ====="
  {
    echo "services:"
    echo "  vllm:"
    echo "    command:"
    echo "      - \"--model=$model\""
    echo "      - \"--host=0.0.0.0\""
    echo "      - \"--port=8000\""
    echo "      - \"--dtype=auto\""
    echo "      - \"--gpu-memory-utilization=0.9\""
    echo "      - \"--max-model-len=8192\""
    echo "      - \"--max-num-seqs=64\""
    echo "      - \"--enable-prefix-caching\""
    echo "      - \"--trust-remote-code\""
    for flag in "$@"; do echo "      - \"$flag\""; done
  } > "$OVERRIDE"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" up -d --force-recreate
  wait_healthy
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  D="p05-quant-lab/results/$label-$STAMP"
  mkdir -p "$D"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$D/vram_mib.txt"
  echo "VRAM at rest: $(cat "$D/vram_mib.txt") MiB"
  python3 p02-bench/bench.py --base-url "$BASE" --model "$model" --quick \
    --out-dir "$D/bench" --run-id "$STAMP"
  python3 p05-quant-lab/quality.py --base-url "$BASE" --model "$model" \
    --label "$label" --out-dir "$D/ppl"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" down
  rm -f "$OVERRIDE"
}

serve_and_measure fp16 Qwen/Qwen2.5-7B-Instruct
# NOTE (2026-09-20, A10G, vLLM 0.29.0): runtime --quantization fp8 fails
# (inductor assert, then Cutlass sm80 epilogue hard-fail). fp8-kv = KV-only
# FP8: viable path, uncalibrated scales cost quality (see README).
serve_and_measure fp8k  Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8
serve_and_measure awq  Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq

echo "===== MATRIX DONE — comparing ====="
python3 p05-quant-lab/compare.py p05-quant-lab/results \
  --plot p05-quant-lab/results/tradeoff.png --report

#!/usr/bin/env bash
# P6 spec-decode matrix — ONE GPU boot, 4 sequential configs (~1h, ~$0.55).
# Draft Qwen2.5-0.5B (~1 GB) + target Qwen2.5-7B. Per config: serve -> P2 quick
# bench -> acceptance snapshot -> down. Draft weights pull once (~1 GB).
# Usage (on the GPU host, p01-serve/.env in place):
#   ./p06-spec-decode/serve-matrix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="http://localhost:8000"
MODEL="Qwen/Qwen2.5-7B-Instruct"
DRAFT="Qwen/Qwen2.5-0.5B-Instruct"
OVERRIDE="$(pwd)/p06-spec-decode/.override.yml"

wait_healthy() {
  for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  echo "server never became healthy"; docker logs --tail 50 vllm-qwen25-7b; exit 1
}

serve_and_measure() {
  local label="$1" spec_json="$2"
  echo "===== CONFIG $label ====="
  {
    echo "services:"
    echo "  vllm:"
    echo "    command:"
    echo "      - \"--model=$MODEL\""
    echo "      - \"--host=0.0.0.0\""
    echo "      - \"--port=8000\""
    echo "      - \"--dtype=auto\""
    echo "      - \"--gpu-memory-utilization=0.9\""
    echo "      - \"--max-model-len=8192\""
    echo "      - \"--max-num-seqs=64\""
    echo "      - \"--enable-prefix-caching\""
    echo "      - \"--trust-remote-code\""
    if [ -n "$spec_json" ]; then echo "      - '--speculative-config=$spec_json'"; fi
  } > "$OVERRIDE"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" up -d --force-recreate
  wait_healthy
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  D="p06-spec-decode/results/$label-$STAMP"
  mkdir -p "$D"
  python3 p02-bench/bench.py --base-url "$BASE" --model "$MODEL" --quick \
    --out-dir "$D/bench" --run-id "$STAMP"
  python3 p06-spec-decode/acceptance.py --base-url "$BASE" --out "$D/acceptance.json"
  docker compose -f p01-serve/compose.yml -f "$OVERRIDE" down
  rm -f "$OVERRIDE"
}

SPEC() {
  printf '{"method":"draft_model","model":"%s","num_speculative_tokens":%s}' "$DRAFT" "$1"
}
serve_and_measure baseline ""
serve_and_measure spec-3 "$(SPEC 3)"
serve_and_measure spec-5 "$(SPEC 5)"
serve_and_measure spec-8 "$(SPEC 8)"

echo "===== MATRIX DONE — comparing ====="
python3 p06-spec-decode/compare.py p06-spec-decode/results \
  --plot p06-spec-decode/results/speedup.png --report

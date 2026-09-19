#!/usr/bin/env bash
# P14 fault injectors — three faults, each reversible, each with a built-in
# safety (auto-recover timers so a forgotten chaos session can't burn the
# budget or the GPU). --dry-run (default) prints what WOULD run.
# Usage on the GPU host:
#   ./p14-chaos/faults.sh kill-replica [--dry-run|--exec]
#   ./p14-chaos/faults.sh throttle-gpu [--dry-run|--exec]
#   ./p14-chaos/faults.sh spike --base-url URL [--dry-run|--exec]
set -euo pipefail
MODE="${2:---dry-run}"
[ "$MODE" = "--exec" ] && DRY="" || DRY="echo [dry-run]"

case "${1:-}" in
  kill-replica)
    # Replica kill: container restart. Recovery = docker restarts it
    # (~60-90 s with warm weights). P13 gateway should mask this entirely.
    $DRY docker restart vllm-qwen25-7b
    echo "inject: replica restart; watch P13 X-Route flip to fallback, then back"
    ;;
  throttle-gpu)
    # GPU throttle: halve clocks for 120 s, then auto-restore. Proves SLO burn
    # under degraded (not dead) hardware — the realistic failure mode.
    $DRY sudo nvidia-smi -lgc 500,1005
    echo "inject: clocks pinned low for 120 s"
    if [ -z "$DRY" ]; then
      sleep 120
      sudo nvidia-smi -rgc
      echo "recovered: clocks restored"
    else
      echo "[dry-run] would: sleep 120; sudo nvidia-smi -rgc"
    fi
    ;;
  spike)
    # Traffic spike: 4x knee concurrency via the P2 harness against the gateway.
    # Expects P11 to scale and P13 to shed via 429s, not 503s.
    BASE="--base-url http://localhost:8080"
    for a in "$@"; do case "$a" in --base-url) BASE="--base-url $3";; esac; done
    $DRY python3 p02-bench/bench.py "$BASE" --prompt-lens 512 --gen-lens 64 \
      --concs 64 --repeats 1 --cooldown-s 1 --run-id "spike-$(date -u +%H%M%SZ)"
    echo "inject: 64-concurrency spike; watch queue depth, scaler, gateway 429s"
    ;;
  *)
    echo "usage: faults.sh {kill-replica|throttle-gpu|spike} [--dry-run|--exec]"
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Cron every 5 min on GPU host: shutdown after 20 min of true idle.
# Idle = vLLM reports no running/waiting requests AND GPU util < 5%.
set -uo pipefail
IDLE_FILE=/tmp/vllm-idle-since
GRACE_S=1200  # 20 min

busy() {
  # Any queued/running requests?
  local q
  q=$(curl -sf --max-time 5 http://localhost:8000/metrics 2>/dev/null \
    | grep -E '^vllm:num_requests_(running|waiting)' | awk '{s+=$2} END {print s+0}')
  [ "${q:-0}" != "0" ] && return 0
  # GPU actually working?
  if command -v nvidia-smi >/dev/null; then
    local u
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-0}" -ge 5 ] && return 0
  fi
  return 1
}

if busy; then rm -f "$IDLE_FILE"; exit 0; fi
[ -f "$IDLE_FILE" ] || { date +%s > "$IDLE_FILE"; exit 0; }
SINCE=$(cat "$IDLE_FILE"); NOW=$(date +%s)
if [ $((NOW - SINCE)) -ge $GRACE_S ]; then
  logger -t idle-autostop "20min idle — shutting down"
  rm -f "$IDLE_FILE"
  sudo shutdown -h now
fi

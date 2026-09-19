#!/usr/bin/env bash
# P2 runner. Paid GPU time starts when this runs — have the server up first.
# Usage: ./run.sh [BASE_URL] [--quick]
#   ./run.sh http://<GPU-IP>:8000 --quick   # smoke (~2 min, ~$0.05)
#   ./run.sh http://<GPU-IP>:8000           # full sweep (~45-60 min, ~$0.50)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${1:-http://localhost:8000}"
MODE="${2:-}"
if [ "$MODE" = "--quick" ]; then
  python3 p02-bench/bench.py --base-url "$BASE" --quick
else
  python3 p02-bench/bench.py --base-url "$BASE"
fi

#!/usr/bin/env bash
# Record GPU spend. Usage: ./log-cost.sh <hours> [spot|ondemand] [note...]
# Rates (us-east-1 g5.xlarge, Sep 2026 — re-check if AWS reprices):
#   spot ~0.54 | on-demand 1.01
set -euo pipefail
cd "$(dirname "$0")"
H="${1:?hours, e.g. 2.5}"; MODE="${2:-spot}"; NOTE="${3:-}"
RATE=0.54; [ "$MODE" = "ondemand" ] && RATE=1.01
COST=$(python3 -c "print(round(float('$H')*float('$RATE'),2))")
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$MODE,g5.xlarge,$H,$RATE,$COST,$NOTE" >> cost-log.csv
TOT=$(python3 -c "import csv;print(round(sum(float(r[5]) for r in csv.reader(open('cost-log.csv')) if r and r[0][0].isdigit()),2))")
echo "logged: ${H}h ${MODE} = \$$COST | TOTAL SPEND: \$$TOT / \$128 budget"

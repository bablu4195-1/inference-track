#!/usr/bin/env bash
# TERMINATE (not stop) the tracked instance. Usage: ./stop.sh [reason]
# One-time spot instances CANNOT be stopped (AWS limitation) — terminate ends
# billing immediately. Weights re-pull next boot (~10 min, ~$0.09); cheaper
# than any retention scheme at this scale.
set -euo pipefail
cd "$(dirname "$0")"
REGION="${REGION:-us-east-1}"
IID="$(cat .instance-id)"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
echo "terminated: $IID (${1:-session end})"
date -u +"%Y-%m-%dT%H:%M:%SZ TERM $IID ${1:-}" >> boot-log.txt
echo "REMINDER: ./log-cost.sh <hours> spot   # record spend in cost-log.csv"

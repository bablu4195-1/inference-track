#!/usr/bin/env bash
# STOP (not terminate) the tracked instance. Usage: ./stop.sh [reason]
# Always stop — never leave running. EBS volumes persist; you pay ~$8/mo
# for 160GB retained, vs $0.54/hr for a running GPU.
set -euo pipefail
cd "$(dirname "$0")"
REGION="${REGION:-us-east-1}"
IID="$(cat .instance-id)"
aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null
aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$IID"
echo "stopped: $IID (${1:-session end})"
date -u +"%Y-%m-%dT%H:%M:%SZ STOP $IID ${1:-}" >> boot-log.txt
echo "REMINDER: ./log-cost.sh <hours> spot   # record spend in cost-log.csv"

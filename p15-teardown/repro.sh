#!/usr/bin/env bash
# P15 repro: from a bare account to published curves. Idempotent-ish: safe to
# re-run; skips weights re-pull when /mnt/hf is populated.
# Usage (Mac, AWS_PROFILE configured, quota approved):
#   ./p15-teardown/repro.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export AWS_PROFILE="${AWS_PROFILE:-default}"

echo "== 0. preflight (free) =="
aws sts get-caller-identity --query Account --output text
source infra/aws/launch.env
echo "spot g5.xlarge $AZ cap \$$SPOT_PRICE"

echo "== 1. launch =="
./infra/aws/spot-launch.sh
IP=$(aws ec2 describe-instances --region "$REGION" \
  --instance-ids "$(cat infra/aws/.instance-id)" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH="ssh -i infra/aws/vllm-track-key.pem -o StrictHostKeyChecking=accept-new $IP"
$SSH 'true' 2>/dev/null || { sleep 60; $SSH 'true'; }

echo "== 2. deploy ($MODEL via runbook) =="
$SSH 'sudo mkdir -p /mnt/hf; [ -d ~/track ] || git clone https://github.com/bablu4195-1/inference-track.git ~/track; cd ~/track && git pull'

echo "== 3. three configs, same P2 workload (see p15-teardown/REPORT.md §3) =="
echo "A: baseline compose up -> bench full -> down"
echo "B: awq override + prefix cache -> bench full + quality.py -> down"
echo "C: fp8 + spec-5 override -> bench full + acceptance.py -> down"
echo "Then: teardown.py A B C --report; log-cost.sh; stop.sh (terminate)"
echo "REPRO SCRIPT SCAFFOLDED — config loops intentionally manual until P5/P6 GPU runs land"

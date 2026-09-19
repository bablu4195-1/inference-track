#!/usr/bin/env bash
# Start the tracked instance. Usage: ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
REGION="${REGION:-us-east-1}"
IID="$(cat .instance-id)"
aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "up: $IID @ $IP  |  ssh -i <key>.pem ubuntu@$IP"
date -u +"%Y-%m-%dT%H:%M:%SZ BOOT $IID" >> boot-log.txt

#!/usr/bin/env bash
# Refresh vllm-sg ingress to THIS machine's current public IP.
# Home IPs rotate (observed 3x in one boot session) — run whenever SSH breaks.
# Usage: ./infra/aws/sg-refresh.sh
set -euo pipefail
cd "$(dirname "$0")"
REGION="${REGION:-us-east-1}"
SG_ID="${SG_ID:-sg-0f33101d46dbcb97c}"
IP="$(curl -sf --max-time 10 https://checkip.amazonaws.com/)"
echo "current IP: $IP"
for PORT in 22 8000; do
  OLD=$(aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG_ID" \
    --query "SecurityGroups[0].IpPermissions[?FromPort==\`$PORT\`].IpRanges[0].CidrIp" \
    --output text)
  if [ "$OLD" != "$IP/32" ]; then
    [ -n "$OLD" ] && [ "$OLD" != "None" ] && aws ec2 revoke-security-group-ingress \
      --region "$REGION" --group-id "$SG_ID" --protocol tcp --port "$PORT" --cidr "$OLD" >/dev/null
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
      --protocol tcp --port "$PORT" --cidr "$IP/32" >/dev/null
    echo "port $PORT: $OLD -> $IP/32"
  else
    echo "port $PORT: already $IP/32"
  fi
done

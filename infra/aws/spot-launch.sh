#!/usr/bin/env bash
# One-shot spot-fleet launcher: g5.xlarge in us-east-1d (cheapest AZ).
# Usage: ./spot-launch.sh [KEY_NAME]
# Writes instance id to infra/aws/.instance-id (gitignored).
# Prereqs: aws cli v2, SG + key (see docs/runbook.md §1), quota approved.
set -euo pipefail
cd "$(dirname "$0")"

REGION="${REGION:-us-east-1}"
AZ="${AZ:-us-east-1c}"   # cheapest g5.xlarge spot (Sep 2026 check: ~$0.51; 1d ~$0.53)
TYPE="${TYPE:-g5.xlarge}"
KEY="${1:-${KEY_NAME:?pass key name or set KEY_NAME}}"
SG_ID="${SG_ID:?set SG_ID to the vllm-sg security group id}"
AMI_ID="${AMI_ID:?set AMI_ID to a Deep Learning AMI (Ubuntu) id, e.g. resolve via console}"
PRICE="${SPOT_PRICE:-0.70}"   # cap; recent spot ~$0.54 — never bid near on-demand

echo "== requesting spot $TYPE in $AZ (cap \$$PRICE/hr) =="
REQ=$(aws ec2 request-spot-instances --region "$REGION" \
  --spot-price "$PRICE" \
  --launch-specification "{
    \"ImageId\":\"$AMI_ID\", \"InstanceType\":\"$TYPE\",
    \"KeyName\":\"$KEY\", \"SecurityGroupIds\":[\"$SG_ID\"],
    \"Placement\":{\"AvailabilityZone\":\"$AZ\"},
    \"BlockDeviceMappings\":[
      {\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":80,\"VolumeType\":\"gp3\"}},
      {\"DeviceName\":\"/dev/sdf\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}
    ],
    \"UserData\":\"$(openssl base64 -A < user-data.sh)\"
  }" --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text)
echo "request: $REQ — waiting for fulfilment (spot can take minutes)..."
aws ec2 wait spot-instance-request-fulfilled --region "$REGION" \
  --spot-instance-request-ids "$REQ"
IID=$(aws ec2 describe-spot-instance-requests --region "$REGION" \
  --spot-instance-request-ids "$REQ" --query 'SpotInstanceRequests[0].InstanceId' --output text)
echo "$IID" > .instance-id
aws ec2 create-tags --region "$REGION" --resources "$IID" \
  --tags Key=Name,Value=vllm-track Key=Project,Value=inference-track || true
echo "instance: $IID (saved to .instance-id)"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "ssh -i <key>.pem ubuntu@$IP"

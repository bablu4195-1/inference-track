#!/usr/bin/env bash
# EC2 user-data: Docker + NVIDIA toolkit + EBS weights volume + idle autostop.
# Runs once at first boot as root. Logs: /var/log/user-data.log
set -xeuo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends docker.io curl nvme-cli cron
systemctl enable --now docker
usermod -aG docker ubuntu || true

# NVIDIA container toolkit (skip if the AMI already ships it — reinstall hangs
# on gpg/tty in cloud-init on some DLAMIs; P5-boot lesson 2026-09-20).
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker || true
systemctl restart docker
fi

# Attach+mount 2nd EBS volume (weights) at /mnt/hf if present and unformatted.
if [ -b /dev/nvme1n1 ] && ! blkid /dev/nvme1n1 >/dev/null 2>&1; then
  mkfs -t ext4 /dev/nvme1n1
fi
mkdir -p /mnt/hf
if [ -b /dev/nvme1n1 ]; then
  mount /dev/nvme1n1 /mnt/hf || true
  grep -q /mnt/hf /etc/fstab || echo "/dev/nvme1n1 /mnt/hf ext4 defaults,nofail 0 2" >> /etc/fstab
fi
mkdir -p /mnt/hf /home/ubuntu/track
chown ubuntu:ubuntu /mnt/hf /home/ubuntu/track

# Idle autostop cron (script ships with repo; also copied to /usr/local/bin).
cp /home/ubuntu/track/infra/aws/idle-autostop.sh /usr/local/bin/idle-autostop.sh 2>/dev/null || true
chmod +x /usr/local/bin/idle-autostop.sh 2>/dev/null || true
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/idle-autostop.sh >> /var/log/idle-autostop.log 2>&1") | crontab -

nvidia-smi -L
docker --version
echo "USER-DATA DONE"

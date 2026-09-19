# Runbook — Phase A (and every GPU boot after)

Model: **Qwen/Qwen2.5-7B-Instruct** (ungated, no HF license click-through).
Instance: **g5.xlarge spot, us-east-1c**. Budget: **$128 hard cap**.

## 0. Local prerequisites (Mac, $0)

```bash
pip install -r requirements.txt   # harness deps (a scehdule: only needed from P2)
chmod +x infra/aws/*.sh p01-serve/smoke.sh
```

> **AWS CLI broken?** If `aws` fails with `bad CPU type in executable`, it's an
> x86_64 binary without Rosetta. Fix with either:
> `softwareupdate --install-rosetta` **or** `pip install awscli` (pure Python,
> ARM-native). Then `aws configure` (or SSO) and verify: `aws sts get-caller-identity`.

## 1. One-time AWS setup (console, $0)

1. **SSH key:** EC2 → Key Pairs → Create `vllm-track-key` (`.pem`) → `chmod 400`.
2. **Security group** `vllm-sg` (VPC default is fine):
   - Inbound: `22` from **your IP only**; `8000` from your IP only
     (never `0.0.0.0/0` — open inference endpoints get mined within hours).
   - Outbound: all.
3. **Quota:** EC2 → Limits → `Running On-Demand G... / Spot G instances` —
   need ≥1 vCPU of `g5` family in us-east-1 (you confirmed approved; if a boot
   fails with `VcpuLimitExceeded`, request increase and wait).
4. **Budget alarms:** Billing → Budgets → monthly cost budget **$128**,
   alerts at $40 / $70 / $100 to your email. This is non-optional.
5. **AMI:** pick current **Deep Learning AMI (Ubuntu)** in us-east-1, note its
   AMI ID → export as `AMI_ID` for `spot-launch.sh`.
6. **HF token:** `huggingface.co/settings/tokens` → fine-grained, read-only →
   never commit; goes into `p01-serve/.env` on the host only.

## 2. Boot sequence (paid time starts here — be ready)

```bash
export SG_ID=sg-xxxx AMI_ID=ami-xxxx KEY_NAME=vllm-track-key REGION=us-east-1
./infra/aws/spot-launch.sh            # ~5-10 min incl. fulfilment
ssh -i ~/.ssh/vllm-track-key.pem ubuntu@<IP>
```

On the host:
```bash
git clone <this-repo> ~/track && cd ~/track
cp p01-serve/vllm.env.example p01-serve/.env   # then insert real HF_TOKEN
sudo mkdir -p /mnt/hf && sudo mount /dev/nvme1n1 /mnt/hf  # if user-data missed it
export HF_TOKEN=$(grep HF_TOKEN p01-serve/.env | cut -d= -f2)
docker compose -f p01-serve/compose.yml up -d
docker logs -f vllm-qwen25-7b        # wait for "Uvicorn running" (~5-8 min incl. weight pull)
./p01-serve/smoke.sh                 # MUST PASS before any experiment
```

Record for P15 repro: `docker inspect vllm/vllm-openai:latest | grep -i digest`,
`curl localhost:8000/version`, model revision from logs. Paste into
`p01-serve/README.md` run table.

## 3. Shutdown sequence (every session, no exceptions)

```bash
./infra/aws/log-cost.sh 2.0 spot "P1 bring-up + smoke"   # from Mac AFTER stop
./infra/aws/stop.sh "P1 smoke pass"
```

- Prefer `stop` (keeps EBS) over `terminate` until the track ends (~$8/mo
  retained vs re-downloading 15 GB weights every boot).
- Snapshot `/dev/sdf` (weights volume) after first successful pull.
- **Kill rule: $40 remaining → no new GPU boots.** Finish write-ups from CSVs.

## 4. Cost model (re-check if AWS reprices)

| Mode | $/hr | $128 buys |
|---|---|---|
| g5.xlarge spot (default) | ~0.54 | ~237 h |
| g5.xlarge on-demand (P15 only if rich) | 1.01 | 126 h |

Phase-A allowance: **6 h / ~$3.20**.

## 5. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `VcpuLimitExceeded` | quota not applied in this AZ → request / try `us-east-1a` |
| Spot request never fills | AZ out of capacity → change `AZ=` to b/c/f |
| `401/403` pulling weights | Qwen is ungated — check `HF_TOKEN` typo/expired |
| `CUDA OOM` at startup | lower `--gpu-memory-utilization 0.85`, check `nvidia-smi` |
| `curl: connection refused :8000` | still loading weights (logs show download %) — wait |
| TTFT > 20 s @ conc 2 | cold compile cache; second run should drop — else check EBS IOPS |
| SSH hangs | SG ingress lost your IP (home IP changed) → re-add |
| `docker: runtime nvidia unknown` | user-data toolkit step failed → rerun `nvidia-ctk runtime configure` + restart docker |

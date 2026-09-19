# Inference Infra Track — 15 projects, one monorepo

Self-hosted LLM inference on AWS spot GPUs (vLLM, Qwen2.5-7B default),
built under a **$128 hard cap**. See `docs/runbook.md` before spending a cent.

## Status — code

- [x] Phase A scaffold, P1 serve (smoked on A10G 2026-09-19)
- [x] P2 bench harness · P3 KV monitor+validation · P4 prefix proxy+A/B
- [x] P5 quant lab · P6 spec decode · P7 Triton RMSNorm · P8 chunked prefill
- [x] P9 paged-attn report rig · P10 disagg · P11 autoscaler · P12 cost dash
- [x] P13 gateway · P14 chaos · P15 teardown scaffold

## Status — GPU measurements banked (`runs/`)

- [x] 20260919-p2 full curves (246/246) · p4 72–75% cuts · p3 PASS +10%
- [x] 20260919-p8 matrix (honest null) · p5/fp16 anchor
- [ ] P5 fp8/awq · P6 matrix · P8 redesign · P9 bs-matrix · P10 dual-GPU
- [ ] P11 scale events · P14 faults · P15 final numbers

Spend: `infra/aws/cost-log.csv` ($1.65 / $128 as of 2026-09-19).

## Layout

```text
infra/aws/    spot launch, start/stop, idle autostop, cost log
libs/         metrics.py (TTFT/ITL), vllm_client.py, kv_math.py
p01-serve/    vLLM compose + smoke test
docs/         runbook.md
```

## Conventions

- Every GPU session: `start.sh` → work → `log-cost.sh <h> spot "<note>"` → `stop.sh`.
- Every result row follows `libs/metrics.py::CSV_HEADER`.
- Every boot records AMI / image digest / model rev in the project README.

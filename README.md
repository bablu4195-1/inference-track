# Inference Infra Track — 15 projects, one monorepo

Self-hosted LLM inference on AWS spot GPUs (vLLM, Qwen2.5-7B default),
built under a **$128 hard cap**. See `docs/runbook.md` before spending a cent.

## Status

- [x] Phase A scaffold (P1 configs, shared libs, AWS scripts, runbook)
- [ ] P1 first boot + smoke (needs AWS: `spot-launch.sh`)
- [ ] P2 benchmark harness
- [ ] P3 KV calculator validation

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

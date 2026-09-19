# P01 — Self-hosted vLLM serve (Qwen2.5-7B-Instruct)

Continuous batching is on by default in vLLM (no flag needed); this project
proves it with concurrent streaming load. Prefix caching + chunked prefill
are enabled in `compose.yml` as the track-wide baseline.

## Files

| File | Purpose |
|---|---|
| `compose.yml` | vLLM OpenAI server, pinned flags |
| `vllm.env.example` | template for host `.env` (never commit `.env`) |
| `smoke.sh` | health + 2× concurrent streaming check (stdlib only) |

## Serve flags (why)

- `--gpu-memory-utilization 0.9` — leave headroom for CUDA graphs/activations
- `--max-model-len 8192` — A10G can't hold 32k × batch; P8/P9 raise deliberately
- `--max-num-batched-tokens 8192` — chunked-prefill budget; P8 sweeps this
- `--max-num-seqs 128` — cap concurrency; P9 pushes past it on purpose
- `--enable-prefix-caching` — P4 measures its win; baseline on
- `--enable-chunked-prefill` — P8 A/B's it; baseline on

## Run table (fill per boot — P15 needs this)

| Date (UTC) | AMI | Image digest | Model rev | Smoke | Notes |
|---|---|---|---|---|---|
| _pending first boot_ | | | | | |

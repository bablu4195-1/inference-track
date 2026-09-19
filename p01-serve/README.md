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
| 2026-09-19 | ami-028c26b58ab69e4ae (DL Base GPU, Ubuntu 26.04) | sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 (vLLM 0.29.0) | Qwen2.5-7B-Instruct (HF HEAD 2026-09-19) | PASS TTFT~480ms ITL~33ms @c2 | VRAM@rest 19483 MiB; first boot i-0f513f3e488b06352 us-east-1c spot |

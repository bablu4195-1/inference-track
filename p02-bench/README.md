# P02 — TTFT/ITL benchmark suite

Load harness every later project reuses (P4/P5/P6/P8/P10/P14). Metrics math
lives in `libs/metrics.py`; this folder owns scenarios, orchestration, plots.

## What it measures

Per request: **TTFT**, **ITL p50/p95**, E2E, tok/s. Per wave: wall-clock
wave throughput (server-side view). Failures are recorded as `ok=False` rows
with the error string — waves never abort.

## Matrix ($128-trimmed)

- prompts: 512 / 2048 / 8192 tokens (~4 chars/token synthetic prose)
- gen: 128 tokens · concs: 1 / 8 / 32 · repeats: 2 · cooldown 5 s
- warmup: 1 unrecorded request per combo (CUDA graphs / compile cache)

Est. full sweep: ~45–60 min spot ≈ **$0.50**. Quick smoke: ~2 min.

## Outputs (`results/<run-id>/`)

`results.csv` (shared schema) · `summary.json` · `manifest.json`
(git sha, server `/version`, scenario grid) · `ttft.png`, `itl.png`,
`throughput.png` via `plot.py`.

## Reading the curves

- **Knee** in TTFT-vs-concurrency = scheduler saturation; all later
  comparisons anchor to concs around the knee.
- ITL rising with prompt_len at fixed conc = prefill starving decode
  (motivates P8 chunked-prefill).

## Run table

| Date (UTC) | run-id | Config | Knee (conc) | Notes |
|---|---|---|---|---|
| _pending first GPU run_ | | | | |

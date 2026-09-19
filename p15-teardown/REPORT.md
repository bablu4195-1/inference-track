# P15 — Public benchmark teardown: 3 serving configs, full methodology

Published latency/throughput/cost curves with everything needed to reproduce
them. Public, reproducible benchmarks are the strongest hiring signal in infra.

## The three configs (same model Qwen2.5-7B, same P2 workload, same g5.xlarge)

| Config | Stack | Represents |
|---|---|---|
| A fp16-colocated | FP16, single replica | the baseline everyone understands |
| B awq-prefix | AWQ INT4 + prefix caching | max efficiency per dollar |
| C fp8-spec | FP8 + speculative decode (draft 0.5B) | max speed at near-zero quality loss |

## Methodology (fill alongside runs)

- AMI: ___ · image digest: ___ · model revs: ___ · git sha: ___
- Workload: P2 full matrix (512/2048/8192 × 128 gen × 1/8/32 × 2 reps)
- Quality: P5 logprob perplexity (B), exactness-by-construction (C)
- Cost: `cost-log.csv` notes per config → P12 $/M
- Repro: `./p15-teardown/repro.sh` (scaffolded; manual config loops until
  P5/P6 GPU runs land, then wired end-to-end)

## Results (teardown.py --report output goes here)

| scenario | A TTFT | B TTFT (Δ) | C TTFT (Δ) | A tps | B tps (Δ) | C tps (Δ) |
|---|---|---|---|---|---|---|
| _(pending P5/P6 GPU runs)_ | | | | | | |

## Honest appendix (already earned, keep adding)

- P2 burst-RST fix (retry-if-zero-tokens + stagger): 93 errs → 0
- P3 detector blindness (post-wave scrapes) → mid-wave sampler
- P8 failed hypothesis: budget moves ITL 46→97 ms, chunking flag doesn't (at tested scale)
- Spot reclaim killed boot 2 mid-run; 1f/1d/1a capacity lessons
- Turn0 prefix-cache subtlety (cross-conversation reuse)

A teardown that documents what broke is worth more than one that hides it.

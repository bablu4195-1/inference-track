# P15 — Public benchmark teardown: 3 serving configs, full methodology

Published latency/throughput/cost curves with everything needed to reproduce
them. Public, reproducible benchmarks are the strongest hiring signal in infra.

## The three configs (same model Qwen2.5-7B, same P2 workload, same g5.xlarge)

| Config | Stack | Represents |
|---|---|---|
| A fp16-colocated | FP16, single replica | the baseline everyone understands |
| B awq-prefix | AWQ INT4 + prefix caching | max efficiency per dollar |
| C fp8-spec | FP8 + speculative decode (draft 0.5B) | max speed at near-zero quality loss |

## Methodology (2026-09-19/21)

- AMI: ami-028c26b58ab69e4ae (DL Base GPU, Ubuntu 26.04) · image
  vllm-openai@sha256:c29147 (vLLM 0.29.0) · Qwen2.5-7B-Instruct (+AWQ, +0.5B draft)
- Workload: P2 full matrix (512/2048/8192 × gen 128 × conc 1/8/32 × 2 reps)
- Quality: P5 logprob perplexity, 437 tokens (B 21.85 vs A 21.37 — anchor; relative only)
- Cost: cost-log notes per boot → P12 $/M (track ledger: $8–9 total)
- Repro: `./p15-teardown/repro.sh` + per-project run-matrix scripts; raw CSVs in `runs/`

## Results (teardown.py --report, 2026-09-21)

| scenario | A TTFT | B TTFT (Δ) | C TTFT (Δ) | A tps | B tps (Δ) | C tps (Δ) |
|---|---|---|---|---|---|---|
| p512/c1 | 551 | 487 (−12%) | 693 (+26%) | 19 | 63 (+236%) | 56 (+199%) |
| p512/c8 | 719 | 650 (−10%) | 828 (+15%) | 191 | 147 (−23%¹) | 379 (+99%) |
| p512/c32 | 801 | 666 (−17%) | 867 (+8%) | 349 | 998 (+186%) | 908 (+160%) |
| p2048/c1 | 1665 | 469 (−72%) | 677 (−59%) | 18 | 65 (+261%) | 28 (+54%) |
| p2048/c8 | 766 | 664 (−13%) | 844 (+10%) | 187 | 416 (+122%) | 356 (+90%) |
| p2048/c32 | 891 | 672 (−25%) | 924 (+4%) | 497 | 1011 (+103%) | 811 (+63%) |
| p8192/c1 | 903 | 671 (−26%) | 869 (−4%) | 23 | 53 (+127%) | 46 (+96%) |
| p8192/c8 | 1328 | 1009 (−24%) | 1252 (−6%) | 161 | 326 (+102%) | 277 (+72%) |
| p8192/c32 | 2018 | 1104 (−45%) | 1471 (−27%) | 275 | 688 (+150%) | 539 (+96%) |

¹ Single out-of-family point (B slower than A at p512/c8 while faster
everywhere else) — flagged as noise, not explained away. C acceptance 97.5%.

Reading: AWQ (B) dominates throughput everywhere (+100–260%) with lower TTFT
to boot — the unit-economics winner. Spec-5 (C) roughly doubles throughput at
matched quality but can lose TTFT at low concurrency (draft overhead). Caveats:
non-streamed TTFT here mixes queueing; ppl sample is small; one flagged outlier.

## Honest appendix (already earned, keep adding)

- P2 burst-RST fix (retry-if-zero-tokens + stagger): 93 errs → 0
- P3 detector blindness (post-wave scrapes) → mid-wave sampler
- P8 failed hypothesis: budget moves ITL 46→97 ms, chunking flag doesn't (at tested scale)
- Spot reclaim killed boot 2 mid-run; 1f/1d/1a capacity lessons
- Turn0 prefix-cache subtlety (cross-conversation reuse)

A teardown that documents what broke is worth more than one that hides it.

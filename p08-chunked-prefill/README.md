# P08 — Chunked-prefill scheduler experiment

Long-context prefills are compute-bound; decodes are memory-bound. Without
scheduling control, one 8k hog stalls every decode stream. vLLM's answer:
chunk the hog's prefill into the `max-num-batched-tokens` budget alongside
decodes (on by default in V1, tunable via budget size).

## Parts

| File | Purpose |
|---|---|
| `experiment.py` | baseline-decode phase + mixed hog+decode phase → `starvation_x` |
| `run-matrix.sh` | 5 server configs, sequential restarts, one boot (~1h, ~$0.55) |
| `compare.py` | tradeoff table + starvation/TTFT plots |

## Matrix

| label | chunked | budget | Expectation |
|---|---|---|---|
| on-2048 | on | 2048 | kindest decodes, slowest hog |
| on-4096 | on | 4096 | |
| on-8192 | on | 8192 | balanced |
| on-16384 | on | 16384 | hog fast, decodes pressured |
| off-16384 | **off** | 16384 | decode starvation (control) |

Tuning rule of thumb (vLLM docs): smaller budget → better ITL, worse TTFT;
throughput wants budget > 8192 on big GPUs. The experiment finds the crossing
for *this* model/GPU/workload instead of quoting docs.

## Result table — first GPU run 2026-09-19 (Qwen2.5-7B, A10G, 4 decodes + 1×8k hog)

| label | ITL_base | ITL_mixed | starvation_x | hog_TTFT |
|---|---|---|---|---|
| on-2048 | 46.6 | 42.2 | 0.91 | 1766 |
| on-4096 | 45.5 | 43.8 | 0.96 | 2049 |
| on-8192 | 55.0 | 57.8 | 1.05 | 1778 |
| on-16384 | 96.2 | 97.7 | 1.02 | 2908 |
| off-16384 | 96.8 | 97.8 | 1.01 | 1754 |

Honest reading — the headline hypothesis FAILED at this workload scale:
no config starves decodes, chunking on/off included. Two real findings instead:
1. **Budget size dominates absolute ITL** (46 → 97 ms from 2048 to 16384):
   bigger batches cost every decode, chunking or not. The tuning knob that
   matters here is budget, not the chunked flag.
2. **V1 protects decodes regardless** (prioritizes decode batching even with
   chunking off) — starvation would show in hog TTFT, which is too noisy at
   n=2 reps to resolve (1754–2908 ms spread).
Redesign for next run: conc 16 decodes + hog 16k, and measure decode ITL
*inside the hog-prefill window only*. Raw: `runs/20260919-p8/`.

Update 2026-09-20 (conc12 retest): on-8192-c12 starvation 0.97x, off-16384-c12
1.01x — still no starvation at 3× the decode pressure. V1's decode-first
scheduling holds; the chunking flag is unresolvable below extreme pressure on
this stack. Standing answer: **budget size is the tuning knob** (46→97 ms).

# P09 — PagedAttention under memory pressure: deep-dive report

## 1. How vLLM paging works (the 2-minute primer)

The KV cache is split into fixed-size **blocks** (`--block-size` tokens each,
default 16). A request's blocks need not be contiguous in GPU memory — a
**block table** maps logical token positions to physical blocks, exactly like
OS virtual memory pages. Consequences:

- **No external fragmentation by design.** A 6k-token request never needs a
  contiguous 6k-token slab; any free blocks do. Allocation fails only when the
  pool is truly exhausted, never because free memory is "in the wrong shape".
- **Internal fragmentation remains.** The last block of each sequence is
  partially empty: average waste ≈ `block_size / 2` tokens per sequence.
  Waste fraction ≈ `block_size / (2 × seq_len)`:

  | seq_len | bs=8 | bs=16 | bs=32 |
  |---|---|---|---|
  | 128 | 3.1% | 6.3% | 12.5% |
  | 2048 | 0.2% | 0.4% | 0.8% |
  | 6000 | 0.07% | 0.13% | 0.27% |

  Small block sizes waste less per sequence but inflate the block table and
  kernel launch overhead; large blocks waste more on short requests. The
  matrix below measures which effect dominates on real hardware.

## 2. Methodology

- Model Qwen2.5-7B FP16, A10G 24 GB, `--gpu-memory-utilization 0.9`.
- Pressure workload: 6 waves × conc 32 × 6k-token prompts (~10.7 GB KV demand
  vs ~6.4 GB headroom — over capacity by design).
- Distress sequence watched in `/metrics`: `kv_cache_usage_perc` → 100%,
  then `num_requests_waiting` > 0 (queueing), then `num_requests_swapped` > 0
  (preemption: lowest-priority sequence evicted, its KV dropped, recomputed
  on reschedule — the latency tax of paging).
- Configs differ **only** in `--block-size` (8/16/32); all else pinned
  (see `run-matrix.sh`). Predictor says all three OOM — the question is
  *how gracefully* and at what throughput cost.

## 3. Results — first GPU run 2026-09-20 (Qwen2.5-7B, A10G, 6× conc32×6k, rotated prompts)

| label | wait@wave | swap@wave | err% | tok/s | peak_kv |
|---|---|---|---|---|---|
| bs16 | w0 | — | 0.0% | 217 | 100% |
| bs32 | w0 | — | 0.0% | 213 | 100% |

(bs8 untestable: 0.29 rejects it on all backends. First bs16 attempt measured
nothing — identical prompts shared all blocks; fixed with prompt rotation.)

## 4. Findings (2026-09-20)

- **bs16 vs bs32: indistinguishable.** Both saturate KV instantly, queue ~22,
  hold ~215 tok/s, never evict. The 0.13%-vs-0.27% internal-fragmentation gap
  is unresolvable at this scale — second honest null of the track (see P8).
- **No eviction at 100% KV is the real story.** The scheduler queues and
  time-slices instead of swapping at conc32×6k. Forcing actual preemption
  needs worse pressure (higher conc, longer gen, or smaller pool) — queued
  as a stretch, with the driver ready.
- **External fragmentation: zero observed**, as designed — allocation never
  failed below capacity, only queued. Paging works; the report's headline is
  that the failure mode is *graceful queueing*, not fragmentation cliffs.

## 5. Repro

`./p09-paged-attn/run-matrix.sh` on any 24 GB GPU (~45 min spot, ~$0.40).
Raw timelines: `results/<label>-*/summary.json`.

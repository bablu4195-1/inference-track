# P03 — KV-cache calculator + live monitor

Most production OOMs are KV math errors, not model size. This project makes
the math explicit, then checks it against a live server.

## Parts

| File | Purpose |
|---|---|
| `../libs/kv_math.py` | predictor: KV bytes/token, max batch per seq-len (GQA-aware) |
| `monitor.py` | polls `/metrics`, logs KV/queue/swap gauges, ALERTs at 90% |
| `pressure.py` | ramps concurrency past the prediction until distress; PASS if within ±15% |

Qwen2.5-7B reference: 56 KiB/token; seq-8192 tops out at batch ~10 in FP16
on a 24 GB A10G at `--gpu-memory-utilization 0.9`.

## Distress definition (what counts as "prediction failed")

Stop the ramp when **either** fires: any `ok=False` request in the wave, or
`vllm:num_requests_waiting >= 4` (scheduler pressure precedes eviction).
`monitor.py` in a second terminal timestamps the same event independently.

## GPU procedure (~30–45 min, ~$0.40)

Terminal 1: `python3 p03-kvcalc/monitor.py --base-url http://<GPU>:8000`
Terminal 2: `python3 p03-kvcalc/pressure.py --base-url http://<GPU>:8000`

## Validation table

| seq_len | Predicted max_batch | Observed | err% | Verdict |
|---|---|---|---|---|
| 8192 (2026-09-19, cache OFF) | 10 | ≥14 no-error; mid-wave queueing from ~conc 4–6 per engine logs | — | **RECALIBRATE (method, not math)** |

First-run lesson (kept, not hidden): the original detector scraped `/metrics`
*after* each wave — queue always reads 0 post-wave. Engine logs proved
queueing happened mid-wave (Waiting 1–4, KV to 61%+). Per-token KV math
validated within ~7% (61% pool at ~75k tokens ≈ 52 KB/tok vs 56 predicted).
Fix committed: `sample_during()` scrapes every 2 s *during* the wave and
distress uses peak mid-wave queue depth. Re-run next boot. Raw:
`runs/20260919-p3/`.

If err% > 15%: first suspect `--gpu-memory-utilization` and block-size
overhead (`BLOCK_OVERHEAD` in `kv_math.py`), not the formula — then recalibrate.

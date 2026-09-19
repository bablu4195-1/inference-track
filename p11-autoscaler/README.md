# P11 — Queue-based GPU autoscaler

GPUs idling at 30% burn money; autoscaling is FinOps. KEDA-style control on
the one signal that matters: **pending-request queue depth**.

## Control law (`scaler.py`, stdlib-only)

- Poll `/metrics` every `--interval-s` (10 s): `vllm:num_requests_waiting`.
- Scale **+1** after `--up-streak` (3) consecutive polls ≥ `--scale-up-q` (8).
- Scale **−1** after `--down-streak` (12) consecutive zero-queue polls.
- `--cooldown-s` (120 s) after any action: no flapping, ever.
- Metrics unreachable → hold (never scale on blindness).
- `--min-n 1` default: replica 1 is the warm pool (pulled image, weights on
  EBS, compiled graphs) — scale-from-1 costs ~30 s, not a cold boot.

## Cold-start math (why min-n=1, not 0)

Cold boot: ~8 min (AMI → docker → 15 GB pull → compile). Warm scale: container
start ~30 s against cached everything. At $0.51/hr, one avoided cold boot per
day pays for ~16 idle-warm hours — the ledger for this lives in P12.

## GPU procedure (~30 min, ~$0.30)

1. Serve P1 baseline. 2. `scaler.py --exec` in background (or dry-run first
   to watch decisions). 3. Replay P2 c32 spike → expect scale 1→2 in
   ~30–60 s, scale-down ~2 min after drain. 4. Paste `scaler.jsonl` excerpt +
   scale latency below. Multi-host (ASG) is a documented stretch.

## Test report (`test_scaler.py` — 14 checks, Mac, $0)

Streak logic, clamps, metrics-down hold, label-set parsing, lone-spike
rejection, pressure-then-idle round trip — green.

## Scale log (fill on GPU)

| Event | t | waiting | Action | Latency |
|---|---|---|---|---|
| spike start | | | | |
| scale 1→2 | | | | |
| drain + scale 2→1 | | | | |

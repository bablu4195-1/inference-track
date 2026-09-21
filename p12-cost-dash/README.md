# P12 — Cost-per-token dashboard

Inference is a unit-economics game. This project turns every results CSV in
the repo into $/M tokens and MFU, and tracks cumulative spend against the
$128 envelope.

## Method (auditable, in `accounting.py`)

1. **Cost source:** `--cost-usd` for ad-hoc runs, or `--cost-log-note` to sum
   matching `infra/aws/cost-log.csv` entries (the same file `log-cost.sh`
   writes after every boot — one ledger, no double books).
2. **Attribution by `e2e_ms`, not tokens.** An 8k-prefill request burns more
   GPU-time than its token count suggests; splitting cost by wall-clock share
   is the honest allocator. Blended $/M is still reported for comparability.
3. **MFU** = `tok/s × 2 × params / peak_TFLOPS` (A10G: 125 BF16 TFLOPS).
   Decode is memory-bound — expect 1–5%. A high MFU on a decode-heavy row
   would be the suspicious result, not a low one.

## Multi-tenancy

`--tenant` tags rows at ingest (default `track`). Per-tenant rate limits and
the gateway's usage headers (P13) feed the same key `(tenant, model, config)`.

## GPU procedure: none ($0)

Runs on any results CSV, including all synthetic validation data. First real
run: point it at P2's first GPU sweep + `log-cost.sh` entry.

## Live ledger (from `runs/`, P12 code on real data 2026-09-21)

| Run | Tokens (in+out) | GPU $ | $/M blended | MFU p50 | Notes |
|---|---|---|---|---|---|
| P2 full sweep (boot1) | 913,152 | 0.54 | **0.59** | ~0.3% | long-context rows cheapest/M (token density beats e2e growth); above commercial $0.20–0.50 at this scale — utilization story holds |

Reference anchor: commercial 8B-class APIs sit near $0.20–0.50/M blended;
self-hosted wins when utilization is high and loses idle — which is exactly
what P11 (autoscaler) optimizes.

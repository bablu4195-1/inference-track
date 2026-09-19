# P10 — Disaggregated prefill/decode cluster

Split prefill (compute-bound) and decode (memory-bound) onto separate GPU
pools — the architecture behind every frontier serving stack. One pool can no
longer starve the other; the price is KV transfer between them.

## Topology (`disagg-compose.yml`, single host first)

`prefill` (:8001, big batch budget) → KV over connector → `decode` (:8002,
tiny budget, pure streaming). Client hits decode; prefill is internal.
Multi-host is the same file with IPs + one NIC that can take it.

**Version warning (honest):** the disagg connector flags changed across vLLM
releases. The compose pins `NixlConnector` for 0.29.0 (boot-1 digest) — verify
against the docs page for whatever image you pull, run the handshake, and pin
what worked. A P10 that papers over version churn is fiction.

## GPU procedure, one boot, 2× g5 (~1h dual, ~$1.10)

1. Colocated baseline: P2 full sweep (already have 2026-09-19).
2. `disagg-compose.yml` up (both workers, same weights on shared /mnt/hf).
3. P2 full sweep against :8002. 4. `compare.py coloc disagg --report`.

## Decision table (fill on GPU)

| Scenario class | TTFT Δ | ITL Δ | tok/s Δ | Winner |
|---|---|---|---|---|
| short prompt, low conc | | | | expect coloc (transfer overhead) |
| long context (8k) | | | | expect disagg (uncontended prefill) |
| mixed P8 workload | | | | expect disagg (isolation) |

Success: disagg wins long-context/mixed, loses-or-ties short — the crossover
documented with numbers is the result, not a trophy for either side.

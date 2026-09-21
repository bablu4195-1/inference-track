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

## Status 2026-09-20: partial (single-host AWQ, no handshake yet)

Both workers healthy (AWQ, 0.45 util each, one A10G), NixlConnector
configured (`kv_both`, NIXL available) — but no evidence of actual KV
transfer: each engine self-registers (`kv_parallel_size=1`, no rank split)
and decode serves standalone. True disagg needs producer/consumer roles +
the routing proxy (0.29 semantics unverified offline — flagged, not faked).
Also fixed en route: missing `healthcheck` blocks (decode never started).
Next: roles + proxy, then the dual-GPU run the quota now allows.

## Decision table — dual-GPU run 2026-09-21 (AWQ, 2× A10G, cross-host NIXL)

(Earlier same-day single-GPU run: coloc won 9–0, +200–350% E2E,
throughput tied +7.3%. Raw: `runs/20260921-p10/`.)

Zero transfer errors across 246 requests — true cross-host disagg works.
But: coloc wins 9/9 on E2E (+190–330%), throughput tied (+10.7% mean).

| Scenario class | TTFT Δ | tok/s Δ | Winner |
|---|---|---|---|
| short prompt | +220–310% | −2…+10% | coloc |
| long context | +188–330% | −3…+78% (noisy) | coloc |

Interpretation (not a failure of the architecture): the mini-proxy is
non-streaming, so its hop serializes into every E2E; and this workload never
saturates colocated (P9: needs 6k-context pressure). Disagg's win condition —
colocated queueing while split pools stay fluid — was not tested. Next: same
duet under P9-class pressure + streaming proxy. Raw: `runs/20260921-p10-dual/`.

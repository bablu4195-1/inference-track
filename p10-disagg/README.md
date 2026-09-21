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

## Decision table — first GPU run 2026-09-21 (AWQ, ONE A10G, mini-proxy)

| Scenario class | TTFT Δ | ITL Δ | tok/s Δ | Winner |
|---|---|---|---|---|
| short prompt (512) | +230–310% | n/a (proxy non-streaming) | −7…+7% | coloc |
| long context (8k) | +200–315% | n/a | −4…+77% (noisy) | coloc |
| **Overall: disagg wins 0/9** (mean tok/s +7.3%) | | | | **coloc** |

Verdict: single-GPU disagg is a **loss** — the proxy hop + split pools on
shared SMs adds 3–4× E2E latency with no isolation benefit. The isolation
*is* the product: this result scopes P10 correctly to **two GPUs**
(quota now allows 8 vCPU spot). Also: mini-proxy is non-streaming, so TTFT
here is really E2E — the compare is fair on throughput, not latency shape.
Handshake chain that finally worked: roles + explicit routable `kv_ip` +
proxy passing prefill's own advertisement (the example proxy overwrites
`remote_host` with `localhost` — the actual bug). Raw: `runs/20260921-p10/`.

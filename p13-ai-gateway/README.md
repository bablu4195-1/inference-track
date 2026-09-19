# P13 — AI gateway with fallbacks and rate limits

Router across self-hosted and API providers with TTFT SLOs, retries, and a
degradation chain. Provider outages are guaranteed; user-facing errors are
optional.

## Policy (implemented in `gateway.py`, stdlib-only)

- **Chain:** primary vLLM → fallback endpoint → `503 + Retry-After`.
- **Failover on:** connection error, HTTP 5xx, or first-chunk time over
  `--slo-ms`. One immediate retry on connection errors, then failover.
- **Never fail over on 4xx** — client bugs relay as-is.
- **SLO breach mid-stream is impossible by construction:** the first 64 KB
  stays buffered until the SLO check passes; a breach closes the backend
  connection before a single byte reaches the client.
- **Rate limits:** token bucket per `X-Tenant` (default tenant `default`);
  429 + `Retry-After` on exhaustion, buckets isolated per tenant.
- **Observability:** every request → JSONL (`tenant, route, fail_reason,
  ttft_ms, slo_breach, status`) — the tenant key drops straight into P12.

## GPU procedure ($0 until first boot, then ~15 min of an existing boot)

1. Primary = P1 server. Fallback = `gpt-oss-120b` on Bedrock (works on your
   creds today; Terra does not — see track notes) or a second vLLM replica.
2. `gateway.py --primary <gpu>:8000 --fallback <fb> --slo-ms 2000 ...`
3. Re-run P2 quick through the gateway; kill the primary mid-run (P14) and
   show uninterrupted `X-Route: fallback` responses in the log.

## Test report (offline, `test_gateway.py` — 16 checks)

Happy-path routing, dead-primary failover, TTFT-breach abandonment,
503 chain exhaustion, 429 + tenant isolation, log schema — all green on Mac.

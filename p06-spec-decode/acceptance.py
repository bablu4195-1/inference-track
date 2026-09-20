#!/usr/bin/env python3
"""P6 acceptance snapshot: scrape vLLM spec-decode gauges after a bench run.

vLLM exposes speculative-decoding counters under the `vllm:spec_decode*`
prefix (exact names vary by version, e.g. acceptance rate / accepted / draft
token counts). This script scrapes by PREFIX so it survives renames: whatever
exists gets recorded, and acceptance is derived when the counters allow.

Derived (when accepted+total counters both present):
  acceptance = accepted_tokens / (accepted_tokens + rejected_tokens)

Falls back to reporting raw gauges + the bench-measured decode speedup, which
is the metric that actually matters (acceptance is the explanation).

Usage: python3 p06-spec-decode/acceptance.py --base-url URL --out acceptance.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request


def fetch(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=15) as r:
        text = r.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.partition(" ")
        name = name.split("{")[0]
        if "spec_decode" in name or "specdecode" in name:
            try:
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                pass
    return out


def derive(m: dict[str, float]) -> dict:
    """Best-effort acceptance from whatever counters exist.

    Prometheus exposes counters twice (`_total` value + `_created` timestamp):
    only `_total` keys participate, else timestamps inflate the ratio (P6
    lesson 2026-09-20: naive sum gave 600%).
    """
    tot = {k: v for k, v in m.items() if k.endswith("_total")}
    out: dict = {}
    acc = [v for k, v in m.items() if "acceptance" in k and "rate" in k]
    if acc:
        out["acceptance_rate_gauge"] = sum(acc) / len(acc)
    a = tot.get("vllm:spec_decode_num_accepted_tokens_total")
    t = tot.get("vllm:spec_decode_num_draft_tokens_total")
    if a is not None and t:
        out["acceptance_derived"] = a / t
    else:  # fuzzy fallback over _total keys only
        acc_keys = [k for k in tot if "accept" in k and "draft" not in k
                    and "per_pos" not in k]
        tot_keys = [k for k in tot if "draft" in k and "token" in k]
        if acc_keys and tot_keys:
            a2 = sum(tot[k] for k in acc_keys)
            t2 = sum(tot[k] for k in tot_keys)
            if t2 > 0:
                out["acceptance_derived"] = a2 / t2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="P6 spec-decode acceptance snapshot")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        m = fetch(a.base_url)
    except Exception as e:
        sys.exit(f"metrics fetch failed: {e}")
    snap = {"taken_utc": datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "spec_decode_gauges": m, "derived": derive(m)}
    with open(a.out, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"spec gauges: {len(m)} keys", json.dumps(snap["derived"]))
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()

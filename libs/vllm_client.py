"""Minimal async client for a vLLM OpenAI-compatible server.

Only dependency: aiohttp (see requirements.txt).
All timing uses time.perf_counter() at the caller for TTFT/ITL math.
"""

from __future__ import annotations

import time

import aiohttp

try:  # flat layout (GPU host runs with cwd=libs/)
    from metrics import RequestStats, compute_stats
except ImportError:  # package layout (repo root)
    from libs.metrics import RequestStats, compute_stats


async def health(base_url: str, timeout_s: float = 10.0) -> bool:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{base_url}/health", timeout=timeout_s) as r:
                return r.status == 200
    except Exception:
        return False


async def stream_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> tuple[RequestStats, str]:
    """POST /v1/completions with stream=true; return (stats, text)."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }
    send_t = time.perf_counter()
    token_times: list[float] = []
    text_parts: list[str] = []
    usage_tokens: int | None = None
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if '"usage"' in data:  # authoritative count (spec batches tokens/chunk)
                try:
                    import json as _json
                    usage_tokens = _json.loads(data)["usage"].get(
                        "completion_tokens", usage_tokens)
                except Exception:
                    pass
                continue
            token_times.append(time.perf_counter())
            # Best-effort text extraction; timing is what matters.
            if '"text"' in data:
                try:
                    import json as _json
                    text_parts.append(_json.loads(data)["choices"][0].get("text", ""))
                except Exception:
                    pass
    stats = compute_stats(send_t, token_times,
                            usage_tokens if usage_tokens else len(token_times))
    return stats, "".join(text_parts)


async def run_concurrent(
    base_url: str, model: str, prompt: str, max_tokens: int,
    concurrency: int,
) -> list[tuple[RequestStats, str]]:
    import asyncio
    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_completion(session, base_url, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Map an oMLX final ``usage`` block (durations in seconds) onto the llama.cpp ``timings`` object the chat UI reads its tok/s badge from. DwarfStar's launcher already emits ``timings`` itself, so a line that carries one passes through untouched."""

from __future__ import annotations

import json
from typing import Any, Optional

ATTACHED_PROVIDER_TYPES = frozenset({"omlx", "dwarfstar"})


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def timings_from_usage(usage: Any) -> Optional[dict[str, Any]]:
    """Build llama.cpp style ``timings`` from an oMLX ``usage`` block, or None when it holds nothing to rate."""
    if not isinstance(usage, dict):
        return None
    completion = _num(usage.get("completion_tokens"))
    generation_s = _num(usage.get("generation_duration"))
    if not completion or completion <= 0 or not generation_s or generation_s <= 0:
        return None
    prompt = _num(usage.get("prompt_tokens")) or 0.0
    details = usage.get("prompt_tokens_details")
    cached = _num(details.get("cached_tokens")) if isinstance(details, dict) else None
    cached = min(cached or 0.0, prompt)
    prompt_n = max(0.0, prompt - cached)
    prefill_s = _num(usage.get("prompt_eval_duration"))
    if not prefill_s or prefill_s <= 0:
        prefill_s = _num(usage.get("time_to_first_token")) or 0.0
    predicted_ps = _num(usage.get("generation_tokens_per_second"))
    if not predicted_ps or predicted_ps <= 0:
        predicted_ps = completion / generation_s
    prompt_ps = _num(usage.get("prompt_tokens_per_second"))
    if not prompt_ps or prompt_ps <= 0:
        prompt_ps = prompt_n / prefill_s if prefill_s > 0 and prompt_n > 0 else 0.0
    return {
        "prompt_n": int(prompt_n),
        "prompt_ms": prefill_s * 1000.0,
        "prompt_per_second": prompt_ps,
        "predicted_n": int(completion),
        "predicted_ms": generation_s * 1000.0,
        "predicted_per_second": predicted_ps,
        "cache_n": int(cached),
    }


def attach_timings(line: Any, provider_type: Optional[str]) -> Any:
    """Return ``line`` with ``timings`` added when it is a chat chunk (SSE ``data:`` line or a bare JSON body) from an attached provider carrying oMLX ``usage`` and no ``timings``. Anything else comes back as the same object."""
    if provider_type not in ATTACHED_PROVIDER_TYPES or not isinstance(line, str):
        return line
    if '"usage"' not in line:
        return line
    prefix = ""
    body = line
    if line.startswith("data:"):
        prefix = "data: "
        body = line[5:].strip()
    try:
        payload = json.loads(body)
    except ValueError:
        return line
    if not isinstance(payload, dict) or payload.get("timings"):
        return line
    timings = timings_from_usage(payload.get("usage"))
    if timings is None:
        return line
    payload["timings"] = timings
    return prefix + json.dumps(payload, separators = (",", ":"), ensure_ascii = False)

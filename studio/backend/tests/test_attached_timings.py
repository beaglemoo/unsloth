# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.attached.timings import attach_timings, timings_from_usage  # noqa: E402

USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 300,
    "total_tokens": 1500,
    "prompt_tokens_details": {"cached_tokens": 1000},
    "time_to_first_token": 0.9,
    "prompt_eval_duration": 0.8,
    "generation_duration": 6.0,
    "total_time": 6.9,
    "prompt_tokens_per_second": 250.0,
    "generation_tokens_per_second": 50.0,
}


def _chunk(**extra):
    return {"id": "x", "choices": [], "usage": dict(USAGE), **extra}


def test_timings_from_usage_maps_fields():
    t = timings_from_usage(USAGE)
    assert t == {
        "prompt_n": 200,
        "prompt_ms": 800.0,
        "prompt_per_second": 250.0,
        "predicted_n": 300,
        "predicted_ms": 6000.0,
        "predicted_per_second": 50.0,
        "cache_n": 1000,
    }


def test_prefill_falls_back_to_ttft_and_rates_are_derived():
    usage = {
        k: v
        for k, v in USAGE.items()
        if k not in ("prompt_eval_duration", "prompt_tokens_per_second", "generation_tokens_per_second")
    }
    t = timings_from_usage(usage)
    assert t["prompt_ms"] == pytest.approx(900.0)
    assert t["prompt_per_second"] == pytest.approx(200 / 0.9)
    assert t["predicted_per_second"] == pytest.approx(50.0)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 5, "completion_tokens": 0, "generation_duration": 1.0},
        {"prompt_tokens": 5, "completion_tokens": 9},
        {"completion_tokens": 9, "generation_duration": 0},
    ],
)
def test_nothing_to_rate(usage):
    assert timings_from_usage(usage) is None


def test_attach_to_sse_line():
    line = "data: " + json.dumps(_chunk())
    out = attach_timings(line, "omlx")
    assert out.startswith("data: ")
    body = json.loads(out[6:])
    assert body["timings"]["predicted_per_second"] == 50.0
    assert body["usage"] == USAGE


def test_attach_to_bare_json_non_stream():
    out = attach_timings(json.dumps(_chunk()), "dwarfstar")
    assert json.loads(out)["timings"]["cache_n"] == 1000


def test_existing_timings_pass_through_untouched():
    line = "data: " + json.dumps(_chunk(timings = {"predicted_per_second": 7.0}))
    assert attach_timings(line, "dwarfstar") is line


@pytest.mark.parametrize("provider", ["openai", "custom", None])
def test_other_providers_untouched(provider):
    line = "data: " + json.dumps(_chunk())
    assert attach_timings(line, provider) is line


@pytest.mark.parametrize(
    "line",
    [
        "data: [DONE]",
        "",
        ": keepalive",
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        'data: {"usage": not json',
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
    ],
)
def test_non_rateable_lines_unchanged(line):
    assert attach_timings(line, "omlx") is line

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
import asyncio
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.attached.timings import attach_timings, timings_from_usage  # noqa: E402
from routes.inference import _ToolLoopTimingTransport  # noqa: E402

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
    out = attach_timings(json.dumps(_chunk()), "omlx")
    assert json.loads(out)["timings"]["cache_n"] == 1000


def test_existing_timings_pass_through_untouched():
    line = "data: " + json.dumps(_chunk(timings = {"predicted_per_second": 7.0}))
    assert attach_timings(line, "legacy") is line


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


def test_tool_loop_final_usage_keeps_summed_omlx_durations():
    class Transport:
        heals_text_tool_calls = True
        sanitizes_provider_frames = True
        preserves_reasoning = False

        def stream(self, **_kwargs):
            async def lines():
                for usage in (
                    {"generation_duration": 2.0, "prompt_eval_duration": 0.2},
                    {"generation_duration": 3.0, "prompt_eval_duration": 0.4},
                ):
                    yield "data: " + json.dumps({"choices": [], "usage": usage})

            return lines()

    async def collect():
        wrapper = _ToolLoopTimingTransport(Transport())
        async for _ in wrapper.stream():
            pass
        final = "data: " + json.dumps(
            {
                "id": "chatcmpl-external-tools",
                "choices": [],
                "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            }
        )
        return attach_timings(wrapper.enrich_final_usage(final), "omlx")

    body = json.loads(asyncio.run(collect())[6:])
    assert body["usage"]["generation_duration"] == 5.0
    assert body["usage"]["prompt_eval_duration"] == pytest.approx(0.6)
    assert body["timings"]["predicted_per_second"] == pytest.approx(1.0)
    assert body["timings"]["prompt_per_second"] == pytest.approx(50.0)


def test_tool_loop_final_usage_keeps_summed_engine_timings():
    class Transport:
        heals_text_tool_calls = True
        sanitizes_provider_frames = True
        preserves_reasoning = False

        def stream(self, **_kwargs):
            async def lines():
                yield "data: " + json.dumps(
                    {"choices": [], "timings": {"predicted_n": 4, "predicted_ms": 200}}
                )
                yield "data: " + json.dumps(
                    {"choices": [], "timings": {"predicted_n": 6, "predicted_ms": 300}}
                )

            return lines()

    async def collect():
        wrapper = _ToolLoopTimingTransport(Transport())
        async for _ in wrapper.stream():
            pass
        return wrapper.enrich_final_usage(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-external-tools",
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 10, "total_tokens": 11},
                }
            )
        )

    body = json.loads(asyncio.run(collect())[6:])
    assert body["timings"]["predicted_n"] == 10.0
    assert body["timings"]["predicted_ms"] == 500.0
    assert body["timings"]["predicted_per_second"] == pytest.approx(20.0)

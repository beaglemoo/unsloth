// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { approximateServerTimings } from "../src/features/chat/api/approx-timings.ts";

test("derives tok/s from usage and the stream clock", () => {
  const t = approximateServerTimings({
    usage: {
      prompt_tokens: 1200,
      completion_tokens: 300,
      prompt_tokens_details: { cached_tokens: 1000 },
    },
    firstTokenMs: 1000,
    totalMs: 7000,
  });
  assert.ok(t);
  assert.equal(t.approx, true);
  assert.equal(t.predicted_n, 300);
  assert.equal(t.predicted_ms, 6000);
  assert.equal(t.predicted_per_second, 50);
  assert.equal(t.prompt_n, 200);
  assert.equal(t.cache_n, 1000);
  assert.equal(t.prompt_ms, 1000);
  assert.equal(t.prompt_per_second, 200);
});

test("no completion tokens means no timings", () => {
  for (const usage of [undefined, {}, { completion_tokens: 0 }]) {
    assert.equal(
      approximateServerTimings({ usage, firstTokenMs: 10, totalMs: 100 }),
      undefined,
    );
  }
});

test("a missing first token counts the whole stream as generation", () => {
  const t = approximateServerTimings({
    usage: { completion_tokens: 100 },
    firstTokenMs: undefined,
    totalMs: 2000,
  });
  assert.equal(t?.predicted_per_second, 50);
  assert.equal(t?.prompt_ms, 0);
});

test("a zero or negative decode window is refused rather than reported as Infinity", () => {
  assert.equal(
    approximateServerTimings({ usage: { completion_tokens: 5 }, firstTokenMs: 500, totalMs: 500 }),
    undefined,
  );
  assert.equal(
    approximateServerTimings({ usage: { completion_tokens: 5 }, firstTokenMs: 900, totalMs: 500 }),
    undefined,
  );
});

test("cached tokens never exceed the prompt", () => {
  const t = approximateServerTimings({
    usage: { prompt_tokens: 10, completion_tokens: 5, prompt_tokens_details: { cached_tokens: 99 } },
    firstTokenMs: 100,
    totalMs: 600,
  });
  assert.equal(t?.cache_n, 10);
  assert.equal(t?.prompt_n, 0);
});

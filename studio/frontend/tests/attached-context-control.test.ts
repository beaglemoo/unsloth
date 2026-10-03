// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampContext,
  ds4AppliedMessage,
  ds4ContextRange,
  ds4PendingNote,
  omlxContextRange,
} from "../src/features/attached-engines/context-control-state.ts";
import type { Ds4ContextInfo, OmlxContextInfo } from "../src/features/attached-engines/types.ts";

const omlx = (patch: Partial<OmlxContextInfo>): OmlxContextInfo => ({
  modelId: "m",
  dir: "m",
  maxContextWindow: 131072,
  nativeMax: 262144,
  effective: 131072,
  ...patch,
});

const ds4 = (patch: Partial<Ds4ContextInfo>): Ds4ContextInfo => ({
  ctx: 100000,
  ctxActive: 100000,
  ctxMin: 4096,
  ctxMax: 262144,
  pendingRestart: false,
  applied: null,
  ...patch,
});

test("oMLX range tops out at the native length and starts on the saved cap", () => {
  assert.deepEqual(omlxContextRange(omlx({})), {
    min: 4096,
    max: 262144,
    current: 131072,
    hasOverride: true,
  });
});

test("oMLX with no saved cap starts on the value in force and offers no reset", () => {
  const range = omlxContextRange(omlx({ maxContextWindow: null, effective: 40960 }));
  assert.equal(range?.current, 40960);
  assert.equal(range?.hasOverride, false);
  assert.equal(omlxContextRange(omlx({ maxContextWindow: null, effective: null }))?.current, 262144);
});

test("oMLX without a native length has no range to offer", () => {
  assert.equal(omlxContextRange(omlx({ nativeMax: null })), null);
});

test("DwarfStar range follows the launcher's bounds", () => {
  assert.deepEqual(ds4ContextRange(ds4({})), {
    min: 4096,
    max: 262144,
    current: 100000,
    hasOverride: false,
  });
  assert.equal(ds4ContextRange(ds4({ ctx: 999999 }))?.current, 262144);
  assert.equal(ds4ContextRange(ds4({ ctxMax: 0 })), null);
});

test("typed values are clamped into range", () => {
  const range = { min: 4096, max: 262144 };
  assert.equal(clampContext(10, range), 4096);
  assert.equal(clampContext(9e9, range), 262144);
  assert.equal(clampContext(50000.4, range), 50000);
  assert.equal(clampContext(Number.NaN, range), 4096);
});

test("each launcher outcome has its own toast", () => {
  assert.match(ds4AppliedMessage("next_start", 8192), /next start/);
  assert.match(ds4AppliedMessage("restarted", 8192), /next message reloads the model \(about 1-2 min\)/);
  assert.match(ds4AppliedMessage("after_current_requests", 8192), /after the current reply/);
  assert.equal(ds4AppliedMessage("unchanged", 8192), "Already using 8,192 tokens.");
  assert.match(ds4AppliedMessage(null, 8192), /8,192/);
});

test("a pending restart is spelled out", () => {
  assert.equal(ds4PendingNote(ds4({})), null);
  assert.match(
    ds4PendingNote(ds4({ pendingRestart: true, ctx: 200000, ctxActive: 100000 })) ?? "",
    /running with 100,000.*200,000/,
  );
});

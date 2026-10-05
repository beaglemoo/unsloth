// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampContext,
  omlxContextRange,
} from "../src/features/attached-engines/context-control-state.ts";
import type { OmlxContextInfo } from "../src/features/attached-engines/types.ts";

const omlx = (patch: Partial<OmlxContextInfo>): OmlxContextInfo => ({
  modelId: "m",
  dir: "m",
  maxContextWindow: 131072,
  nativeMax: 262144,
  effective: 131072,
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

test("typed values are clamped into range", () => {
  const range = { min: 4096, max: 262144 };
  assert.equal(clampContext(10, range), 4096);
  assert.equal(clampContext(9e9, range), 262144);
  assert.equal(clampContext(50000.4, range), 50000);
  assert.equal(clampContext(Number.NaN, range), 4096);
});

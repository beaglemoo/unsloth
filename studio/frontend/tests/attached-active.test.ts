// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  activeContextWindow,
  describeUnload,
  formatCountdown,
  remainingNow,
  unloadStateFor,
} from "../src/features/attached-engines/attached-active.ts";
import { groupOmlxModels } from "../src/features/attached-engines/omlx-groups.ts";
import type {
  AttachedDs4Status,
  AttachedOmlxModel,
  AttachedStatus,
} from "../src/features/attached-engines/types.ts";

const DIR = "/m/Swift";

function model(patch: Partial<AttachedOmlxModel>): AttachedOmlxModel {
  return {
    id: "Swift",
    modelPath: DIR,
    loaded: false,
    isLoading: false,
    estimatedSize: 1,
    pinned: false,
    engineType: "vlm",
    isHelper: false,
    modelAlias: null,
    maxContextWindow: null,
    modelContextLength: null,
    maxTokens: null,
    ttlS: null,
    idleRemainingS: null,
    ...patch,
  };
}

function ds4(patch: Partial<AttachedDs4Status>): AttachedDs4Status {
  return {
    reachable: true,
    loaded: false,
    starting: false,
    pid: null,
    uptimeS: null,
    inFlight: 0,
    idleRemainingS: null,
    liveTps: null,
    lastGenTps: null,
    lastTtftMs: null,
    startTimeoutS: 120,
    ctx: null,
    ctxActive: null,
    pendingRestart: false,
    error: null,
    ...patch,
  };
}

function status(
  rows: AttachedOmlxModel[],
  engine: AttachedDs4Status | null = null,
  receivedAt = 1_000,
): AttachedStatus {
  return {
    omlx: {
      reachable: true,
      models: rows,
      memoryBytes: 0,
      ceilingBytes: 0,
      error: null,
      chatModelIds: [],
    },
    ds4: engine,
    modelsHash: "",
    notices: [],
    receivedAt,
  };
}

const omlx = (modelId: string) => ({ kind: "omlx" as const, modelId });
const star = { kind: "dwarfstar" as const, modelId: "qwen" };

test("context window prefers the resolved cap and falls back to the native length", () => {
  const capped = status([
    model({ id: "Swift", modelAlias: "swift", maxContextWindow: 131072, modelContextLength: 262144 }),
    model({ id: "swift:fast", maxContextWindow: 131072, modelContextLength: 262144 }),
  ]);
  assert.equal(activeContextWindow(capped, omlx("swift:fast")), 131072);
  assert.equal(activeContextWindow(capped, omlx("swift")), 131072);
  const native = status([model({ modelContextLength: 40960 })]);
  assert.equal(activeContextWindow(native, omlx("Swift")), 40960);
  assert.equal(activeContextWindow(native, omlx("missing")), null);
});

test("DwarfStar context is the running value, else the configured one", () => {
  assert.equal(activeContextWindow(status([], ds4({ ctx: 100000, ctxActive: 90000 })), star), 90000);
  assert.equal(activeContextWindow(status([], ds4({ ctx: 100000 })), star), 100000);
  assert.equal(activeContextWindow(status([], ds4({ reachable: false, ctx: 5 })), star), null);
  assert.equal(activeContextWindow(null, star), null);
});

test("oMLX unload states", () => {
  const at = (rows: AttachedOmlxModel[], id = "Swift") => unloadStateFor(status(rows), omlx(id));
  assert.deepEqual(at([model({})]), { kind: "not-loaded" });
  assert.deepEqual(at([model({ isLoading: true })]), { kind: "loading" });
  assert.deepEqual(at([model({ loaded: true, pinned: true })]), { kind: "pinned" });
  assert.deepEqual(at([model({ loaded: true })]), { kind: "untimed" });
  assert.deepEqual(at([model({ loaded: true, idleRemainingS: 252 })]), {
    kind: "countdown",
    remainingS: 252,
  });
  // a profile row reads the directory's loaded state
  assert.deepEqual(
    at(
      [model({ loaded: true, idleRemainingS: 90 }), model({ id: "Swift:fast" })],
      "Swift:fast",
    ),
    { kind: "countdown", remainingS: 90 },
  );
});

test("DwarfStar unload states", () => {
  const at = (patch: Partial<AttachedDs4Status>) => unloadStateFor(status([], ds4(patch)), star);
  assert.deepEqual(at({ reachable: false }), { kind: "offline" });
  assert.deepEqual(at({ starting: true }), { kind: "loading" });
  assert.deepEqual(at({}), { kind: "not-loaded" });
  assert.deepEqual(at({ loaded: true, inFlight: 1 }), { kind: "busy" });
  assert.deepEqual(at({ loaded: true }), { kind: "untimed" });
  assert.deepEqual(at({ loaded: true, idleRemainingS: 180 }), { kind: "countdown", remainingS: 180 });
});

test("the countdown runs down from the poll and never goes negative", () => {
  assert.equal(remainingNow(252, 1_000, 1_000), 252);
  assert.equal(remainingNow(252, 1_000, 61_000), 192);
  assert.equal(remainingNow(10, 1_000, 99_000), 0);
  assert.equal(remainingNow(10, undefined, 99_000), 10);
  assert.equal(remainingNow(10, 5_000, 1_000), 10);
});

test("formats m:ss, rounding up", () => {
  assert.equal(formatCountdown(252), "4:12");
  assert.equal(formatCountdown(59.2), "1:00");
  assert.equal(formatCountdown(0.1), "0:01");
  assert.equal(formatCountdown(0), "0:00");
  assert.equal(formatCountdown(600), "10:00");
});

test("describeUnload text", () => {
  const now = 11_000;
  assert.equal(describeUnload({ kind: "countdown", remainingS: 262 }, 1_000, now), "Unloads in 4:12");
  assert.equal(describeUnload({ kind: "countdown", remainingS: 5 }, 1_000, now), "Unloading");
  assert.equal(describeUnload({ kind: "loading" }, 1_000, now), "Loading");
  assert.equal(describeUnload({ kind: "not-loaded" }, 1_000, now), "Not loaded");
  assert.equal(describeUnload({ kind: "pinned" }, 1_000, now), "Pinned, stays loaded");
  assert.equal(describeUnload({ kind: "offline" }, 1_000, now), null);
  assert.equal(describeUnload(null, 1_000, now), null);
});

test("a group carries the smallest idle countdown of its loaded rows", () => {
  const groups = groupOmlxModels(
    status([
      model({ loaded: true, idleRemainingS: 120 }),
      model({ id: "Swift:fast", loaded: true, idleRemainingS: 100 }),
    ]).omlx!,
  );
  assert.equal(groups[0].idleRemainingS, 100);
});

test("output ceiling: oMLX's max_tokens, DwarfStar's context", async () => {
  const { attachedMaxOutputTokens } = await import(
    "../src/features/attached-engines/attached-active.ts"
  );
  const withTokens = status([model({ maxTokens: 8192 })]);
  assert.equal(attachedMaxOutputTokens(withTokens, omlx("Swift")), 8192);
  assert.equal(attachedMaxOutputTokens(status([model({})]), omlx("Swift")), null);
  assert.equal(
    attachedMaxOutputTokens(status([], ds4({ ctx: 100000, ctxActive: 65536 })), star),
    65536,
  );
  assert.equal(attachedMaxOutputTokens(status([], ds4({ reachable: false })), star), null);
  assert.equal(attachedMaxOutputTokens(null, star), null);
});

test("the output ceiling bounds the request, never the saved value", async () => {
  const { capMaxTokensForAttached } = await import(
    "../src/features/attached-engines/attached-active.ts"
  );
  const small = status([], ds4({ ctx: 100000, ctxActive: 16384 }));
  const saved = 65536;
  assert.equal(capMaxTokensForAttached(saved, small, star), 16384);
  // the caller's value is a plain number: nothing was written back
  assert.equal(saved, 65536);
  // unknown ceiling, offline engine, no attached selection: pass through
  assert.equal(capMaxTokensForAttached(saved, status([], ds4({ reachable: false })), star), saved);
  assert.equal(capMaxTokensForAttached(saved, null, star), saved);
  assert.equal(capMaxTokensForAttached(saved, small, null), saved);
  // already within the ceiling
  assert.equal(capMaxTokensForAttached(4096, small, star), 4096);
});

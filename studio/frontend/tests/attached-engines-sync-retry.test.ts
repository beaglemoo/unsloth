// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createAttachedPoller } from "../src/features/attached-engines/attached-engines-poller.ts";
import type {
  AttachedStatus,
  AttachedSyncResult,
} from "../src/features/attached-engines/types.ts";

const STATUS: AttachedStatus = {
  omlx: null,
  modelsHash: "h",
  notices: [],
};

async function settle() {
  for (let i = 0; i < 10; i += 1) await Promise.resolve();
}

function run(syncReplies: AttachedSyncResult[]) {
  const calls: string[] = [];
  const timers: Array<{ fn: () => void; live: boolean }> = [];
  let syncIndex = 0;
  const poller = createAttachedPoller({
    fetchStatus: async () => STATUS,
    postSync: async () => {
      calls.push("sync");
      const reply = syncReplies[Math.min(syncIndex, syncReplies.length - 1)];
      syncIndex += 1;
      return reply;
    },
    syncProviders: async () => {
      calls.push("providers");
    },
    onStatus: () => undefined,
    onNotice: () => undefined,
    setTimer: (fn) => {
      const timer = { fn, live: true };
      timers.push(timer);
      return timer;
    },
    clearTimer: () => undefined,
  });
  const tick = async () => {
    const timer = timers.filter((t) => t.live).at(-1);
    assert.ok(timer);
    timer.live = false;
    timer.fn();
    await settle();
  };
  return { poller, calls, tick };
}

test("a sync that kept the old models (incomplete catalog) is retried on the next tick", async () => {
  const h = run([
    { enabled: true, incomplete: true },
    { enabled: true, incomplete: false },
  ]);
  h.poller.start();
  await settle();
  assert.deepEqual(h.calls, ["sync", "providers"]);
  await h.tick();
  assert.deepEqual(h.calls, ["sync", "providers", "sync", "providers"]);
  h.poller.stop();
});

test("once the sync completes the hash is remembered and syncing stops", async () => {
  const h = run([
    { enabled: true, incomplete: true },
    { enabled: true, incomplete: false },
  ]);
  h.poller.start();
  await settle();
  await h.tick();
  await h.tick();
  await h.tick();
  assert.equal(h.calls.filter((c) => c === "sync").length, 2);
  h.poller.stop();
});

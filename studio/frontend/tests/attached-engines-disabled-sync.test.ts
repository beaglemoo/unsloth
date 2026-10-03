// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createAttachedPoller } from "../src/features/attached-engines/attached-engines-poller.ts";
import type { AttachedStatus } from "../src/features/attached-engines/types.ts";

async function settle() {
  for (let i = 0; i < 10; i += 1) await Promise.resolve();
}

function run(replies: Array<AttachedStatus | null>, failDisabled = false) {
  const calls: string[] = [];
  const timers: Array<{ fn: () => void; live: boolean }> = [];
  let index = 0;
  const poller = createAttachedPoller({
    fetchStatus: async () => {
      calls.push("status");
      const reply = replies[Math.min(index, replies.length - 1)];
      index += 1;
      return reply;
    },
    postSync: async () => {
      calls.push("sync");
    },
    syncProviders: async () => {
      calls.push("providers");
    },
    onStatus: (status) => calls.push(status === null ? "status:null" : "status:ok"),
    onNotice: () => undefined,
    onDisabled: async () => {
      calls.push("disabled");
      if (failDisabled) throw new Error("sync failed");
    },
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
  return { poller, calls, tick, timers };
}

const OK: AttachedStatus = { omlx: null, ds4: null, modelsHash: "h", notices: [] };

test("a 404 from /status runs the disabled-feature sync, then stops polling", async () => {
  const h = run([OK, null]);
  h.poller.start();
  await settle();
  await h.tick();
  assert.deepEqual(h.calls.slice(-2), ["status:null", "disabled"]);
  assert.equal(h.timers.filter((t) => t.live).length, 0);
});

test("a failing disabled-feature sync still stops polling", async () => {
  const h = run([null], true);
  h.poller.start();
  await settle();
  assert.deepEqual(h.calls, ["status", "status:null", "disabled"]);
  assert.equal(h.timers.filter((t) => t.live).length, 0);
});

test("the disabled-feature sync does not run while the feature is on", async () => {
  const h = run([OK]);
  h.poller.start();
  await settle();
  await h.tick();
  assert.equal(h.calls.includes("disabled"), false);
  h.poller.stop();
});

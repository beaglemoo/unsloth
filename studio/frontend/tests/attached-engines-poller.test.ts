// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  ATTACHED_POLL_INTERVAL_MS,
  createAttachedPoller,
} from "../src/features/attached-engines/attached-engines-poller.ts";
import {
  isAttachedEngineOffline,
  isDwarfStarWarming,
} from "../src/features/attached-engines/attached-state.ts";
import { noticeMessage } from "../src/features/attached-engines/notice-message.ts";
import {
  groupOmlxModels,
  loadableGroups,
  residentGroups,
} from "../src/features/attached-engines/omlx-groups.ts";
import type {
  AttachedNotice,
  AttachedOmlxModel,
  AttachedOmlxStatus,
  AttachedStatus,
} from "../src/features/attached-engines/types.ts";

function status(
  modelsHash: string,
  notices: AttachedNotice[] = [],
): AttachedStatus {
  return { omlx: null, ds4: null, modelsHash, notices };
}

function notice(ts: number): AttachedNotice {
  return { ts, reason: "training", actions: ["Stopped DwarfStar"], inFlightKilled: 0 };
}

// A scripted world: the poller's timers, requests and sinks, all recorded in order.
function harness(replies: Array<AttachedStatus | null | Error>) {
  const calls: string[] = [];
  const sinceArgs: number[] = [];
  const statuses: Array<AttachedStatus | null> = [];
  const notices: AttachedNotice[] = [];
  const timers: Array<{ fn: () => void; ms: number; live: boolean }> = [];
  let index = 0;
  const poller = createAttachedPoller({
    fetchStatus: async (since) => {
      sinceArgs.push(since);
      calls.push("status");
      const reply = replies[Math.min(index, replies.length - 1)];
      index += 1;
      if (reply instanceof Error) throw reply;
      return reply;
    },
    postSync: async () => {
      calls.push("sync");
    },
    syncProviders: async () => {
      calls.push("providers");
    },
    onStatus: (next) => statuses.push(next),
    onNotice: (n) => notices.push(n),
    setTimer: (fn, ms) => {
      const timer = { fn, ms, live: true };
      timers.push(timer);
      return timer;
    },
    clearTimer: (handle) => {
      (handle as { live: boolean }).live = false;
    },
  });
  const fire = async () => {
    const timer = timers.filter((t) => t.live).at(-1);
    assert.ok(timer, "a timer is pending");
    timer.live = false;
    timer.fn();
    // Let the tick and the re-schedule that follows it settle.
    for (let i = 0; i < 10; i += 1) await Promise.resolve();
  };
  const settle = async () => {
    for (let i = 0; i < 10; i += 1) await Promise.resolve();
  };
  return { poller, calls, sinceArgs, statuses, notices, timers, fire, settle };
}

test("the first status is a models_hash change: sync, then pull providers", async () => {
  const h = harness([status("a")]);
  h.poller.start();
  await h.settle();
  assert.deepEqual(h.calls, ["status", "sync", "providers"]);
  h.poller.stop();
});

test("an unchanged models_hash does not sync again", async () => {
  const h = harness([status("a")]);
  h.poller.start();
  await h.settle();
  await h.fire();
  await h.fire();
  assert.deepEqual(h.calls, ["status", "sync", "providers", "status", "status"]);
  h.poller.stop();
});

test("a changed models_hash syncs again", async () => {
  const h = harness([status("a"), status("b")]);
  h.poller.start();
  await h.settle();
  await h.fire();
  assert.deepEqual(h.calls, [
    "status", "sync", "providers", "status", "sync", "providers",
  ]);
  h.poller.stop();
});

test("polls every 5 s and stop cancels the pending tick", async () => {
  const h = harness([status("a")]);
  h.poller.start();
  await h.settle();
  const pending = h.timers.filter((t) => t.live);
  assert.equal(pending.length, 1);
  assert.equal(pending[0].ms, ATTACHED_POLL_INTERVAL_MS);
  assert.equal(ATTACHED_POLL_INTERVAL_MS, 5000);
  h.poller.stop();
  assert.equal(h.timers.filter((t) => t.live).length, 0);
});

test("a 404 (flag off) clears the status and stops polling", async () => {
  const h = harness([status("a"), null]);
  h.poller.start();
  await h.settle();
  await h.fire();
  assert.equal(h.statuses.at(-1), null);
  assert.equal(h.timers.filter((t) => t.live).length, 0);
});

test("a failed sync is retried on the next tick", async () => {
  let attempts = 0;
  const calls: string[] = [];
  const timers: Array<{ fn: () => void; live: boolean }> = [];
  const poller = createAttachedPoller({
    fetchStatus: async () => status("a"),
    postSync: async () => {
      attempts += 1;
      calls.push("sync");
      if (attempts === 1) throw new Error("boom");
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
  poller.start();
  for (let i = 0; i < 10; i += 1) await Promise.resolve();
  timers.filter((t) => t.live).at(-1)?.fn();
  for (let i = 0; i < 10; i += 1) await Promise.resolve();
  assert.deepEqual(calls, ["sync", "sync", "providers"]);
  poller.stop();
});

test("a network error keeps polling and does not clear the status", async () => {
  const h = harness([status("a"), new Error("offline"), status("a")]);
  h.poller.start();
  await h.settle();
  await h.fire();
  await h.fire();
  assert.equal(h.statuses.filter((s) => s === null).length, 0);
  assert.equal(h.sinceArgs.length, 3);
  h.poller.stop();
});

test("notices from before the page opened are not toasted; later ones are, once", async () => {
  const h = harness([
    status("a", [notice(10)]),
    status("a", [notice(10), notice(20)]),
    status("a", [notice(20)]),
    status("a", [notice(20), notice(30)]),
  ]);
  h.poller.start();
  await h.settle();
  assert.deepEqual(h.notices.map((n) => n.ts), []);
  await h.fire();
  assert.deepEqual(h.notices.map((n) => n.ts), [20]);
  await h.fire();
  assert.deepEqual(h.notices.map((n) => n.ts), [20]);
  await h.fire();
  assert.deepEqual(h.notices.map((n) => n.ts), [20, 30]);
  // The newest stamp seen is what the next request filters by.
  assert.deepEqual(h.sinceArgs, [0, 10, 20, 20]);
  h.poller.stop();
});

test("a manual tick polls now and does not overlap one already running", async () => {
  const h = harness([status("a")]);
  h.poller.start();
  await Promise.all([h.poller.tick(), h.poller.tick()]);
  assert.equal(h.calls.filter((c) => c === "status").length, 1);
  h.poller.stop();
});

test("offline is only claimed once the engines have been polled", () => {
  assert.equal(isAttachedEngineOffline("omlx", null), false);
  const reachable = (reachableFlag: boolean) =>
    ({
      omlx: { reachable: reachableFlag } as AttachedOmlxStatus,
      ds4: { reachable: true } as never,
      modelsHash: "",
      notices: [],
    }) as AttachedStatus;
  assert.equal(isAttachedEngineOffline("omlx", reachable(false)), true);
  assert.equal(isAttachedEngineOffline("omlx", reachable(true)), false);
  assert.equal(isAttachedEngineOffline("dwarfstar", reachable(false)), false);
});

test("DwarfStar warming reads ds4.starting", () => {
  assert.equal(isDwarfStarWarming(null), false);
  const warm = {
    omlx: null,
    ds4: { starting: true } as never,
    modelsHash: "",
    notices: [],
  } as AttachedStatus;
  assert.equal(isDwarfStarWarming(warm), true);
});

test("notice messages name the reason, actions and killed requests", () => {
  assert.equal(
    noticeMessage({
      ts: 1,
      reason: "omlx_use",
      actions: ["Stopped DwarfStar"],
      inFlightKilled: 2,
    }),
    "Switched to oMLX. Stopped DwarfStar. 2 in-flight requests stopped",
  );
});

function omlxModel(patch: Partial<AttachedOmlxModel>): AttachedOmlxModel {
  return {
    id: "m",
    modelPath: "/models/m",
    loaded: false,
    isLoading: false,
    estimatedSize: 0,
    pinned: false,
    engineType: null,
    isHelper: false,
    modelAlias: null,
    ...patch,
  };
}

test("oMLX aliases collapse onto their directory and embeddings are not loadable", () => {
  const dir = "/models/ukisai--Swift-27B";
  const models = [
    omlxModel({ id: "ukisai--Swift-27B", modelPath: dir, estimatedSize: 100, loaded: true }),
    omlxModel({ id: "swift-1.5-27b", modelPath: dir, estimatedSize: 100 }),
    omlxModel({ id: "swift-1.5-27b:fast", modelPath: dir, estimatedSize: 100 }),
    omlxModel({ id: "other", modelPath: "/models/other", estimatedSize: 50 }),
    omlxModel({
      id: "embed",
      modelPath: "/models/embed",
      engineType: "embedding",
      pinned: true,
      loaded: true,
      estimatedSize: 5,
    }),
  ];
  const groups = groupOmlxModels({
    reachable: true,
    models,
    memoryBytes: 105,
    ceilingBytes: 1000,
    error: null,
    chatModelIds: ["swift-1.5-27b", "swift-1.5-27b:fast", "other"],
  });
  assert.equal(groups.length, 3);
  const resident = residentGroups(groups);
  assert.deepEqual(resident.map((g) => g.id), ["ukisai--Swift-27B", "embed"]);
  assert.deepEqual(loadableGroups(groups).map((g) => g.id), ["other"]);
});

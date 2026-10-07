// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  comfyuiFromApi,
  comfyuiModelsFromApi,
  comfyuiQueueFromApi,
} from "../src/features/attached-engines/comfyui-wire.ts";
import { createAttachedPoller } from "../src/features/attached-engines/attached-engines-poller.ts";
import type { AttachedStatus } from "../src/features/attached-engines/types.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

const LIVE = {
  url: "http://127.0.0.1:8844",
  reachable: true,
  state: "busy",
  version: "0.39.0",
  queue_running: 1,
  queue_pending: 2,
  devices: [{ name: "Apple M5 Pro", type: "mps", vram_total: 1 }],
  ram_total: 68719476736,
  ram_free: 32000000000,
  failure: null,
  helper_wanted: true,
  peers: [{ url: "http://127.0.0.1:8188", reachable: true, busy: false, state: "idle" }],
};

test("the /status comfyui block parses to camel case", () => {
  assert.deepEqual(comfyuiFromApi(LIVE), {
    url: "http://127.0.0.1:8844",
    reachable: true,
    state: "busy",
    version: "0.39.0",
    queueRunning: 1,
    queuePending: 2,
    devices: [{ name: "Apple M5 Pro", type: "mps" }],
    ramTotal: 68719476736,
    ramFree: 32000000000,
    failure: null,
    helperWanted: true,
    peers: [{ url: "http://127.0.0.1:8188", reachable: true, busy: false, state: "idle" }],
  });
});

test("a down ComfyUI with a failure marker keeps the marker", () => {
  const parsed = comfyuiFromApi({
    ...LIVE,
    reachable: false,
    state: "down",
    version: null,
    ram_free: null,
    devices: [],
    queue_running: 0,
    queue_pending: 0,
    failure: { reason: "venv missing", ts: 5, count: 2 },
    helper_wanted: null,
    peers: [],
  });
  assert.equal(parsed?.state, "down");
  assert.deepEqual(parsed?.failure, { reason: "venv missing", ts: 5, count: 2 });
  assert.equal(parsed?.helperWanted, null);
  assert.equal(parsed?.ramFree, null);
});

test("junk is tolerated: an unknown state reads as unknown, missing fields as empty", () => {
  assert.equal(comfyuiFromApi(undefined), null);
  assert.equal(comfyuiFromApi(null), null);
  assert.equal(comfyuiFromApi("x"), null);
  const sparse = comfyuiFromApi({ state: "weird", devices: [null, 3], peers: [null] });
  assert.equal(sparse?.state, "unknown");
  assert.equal(sparse?.reachable, false);
  assert.equal(sparse?.queueRunning, 0);
  assert.deepEqual(sparse?.devices, [
    { name: "", type: "" },
    { name: "", type: "" },
  ]);
  assert.equal(sparse?.peers.length, 1);
});

test("the queue parses running before pending and flags Studio's own jobs", () => {
  assert.deepEqual(
    comfyuiQueueFromApi({
      running: [{ prompt_id: "a", number: 1, state: "running", studio: { template: "x" } }],
      pending: [
        { prompt_id: "b", number: 2, state: "pending", studio: null },
        { prompt_id: "", number: 3 },
        { number: 4 },
      ],
    }),
    [
      { promptId: "a", number: 1, state: "running", studio: true },
      { promptId: "b", number: 2, state: "pending", studio: false },
    ],
  );
  assert.deepEqual(comfyuiQueueFromApi(null), []);
  assert.deepEqual(comfyuiQueueFromApi({}), []);
});

test("the models route parses folders and drops non-strings", () => {
  assert.deepEqual(
    comfyuiModelsFromApi({ folders: { vae: ["v.safetensors", 3], loras: [] } }),
    { vae: ["v.safetensors"], loras: [] },
  );
  assert.deepEqual(comfyuiModelsFromApi({}), {});
  assert.deepEqual(comfyuiModelsFromApi(null), {});
});

test("fetchAttachedStatus parses comfyui, so an older backend without the block still works", () => {
  const api = src("features/attached-engines/api.ts");
  assert.match(api, /comfyui: comfyuiFromApi\(body\.comfyui\)/);
  // the block is optional on the type, so existing fixtures without it keep compiling
  assert.match(src("features/attached-engines/types.ts"), /comfyui\?: ComfyuiStatus \| null;/);
});

test("the poller is indifferent to the comfyui block", async () => {
  const seen: Array<AttachedStatus | null> = [];
  const withBlock: AttachedStatus = {
    omlx: null,
    modelsHash: "h",
    notices: [],
    comfyui: comfyuiFromApi(LIVE),
  };
  const without: AttachedStatus = { omlx: null, modelsHash: "h", notices: [] };
  const replies = [withBlock, without];
  const poller = createAttachedPoller({
    fetchStatus: async () => replies.shift() ?? null,
    postSync: async () => undefined,
    syncProviders: async () => undefined,
    onStatus: (status) => void seen.push(status),
    onNotice: () => undefined,
    setTimer: () => 0,
    clearTimer: () => undefined,
  });
  poller.start();
  await poller.tick();
  await poller.tick();
  poller.stop();
  assert.equal(seen.length >= 2, true);
  assert.equal(seen[0]?.comfyui?.state, "busy");
  assert.equal(seen[1]?.comfyui, undefined);
});

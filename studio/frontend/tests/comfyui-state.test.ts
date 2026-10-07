// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  STORYPRESS_PORT_HINT,
  cancelOutcome,
  comfyuiDeviceLine,
  comfyuiStatusLine,
  comfyuiUnloadRule,
  failureText,
  formatBytes,
  freeResultMessage,
  groupComfyuiModels,
  modelCountText,
  pendingPosition,
  peerNotes,
  queueRowDetail,
  queueRowTitle,
} from "../src/features/attached-engines/comfyui-state.ts";
import { EngineHttpError } from "../src/features/attached-engines/engine-errors.ts";
import {
  type EngineHelperRow,
  type EngineHelpersStatus,
  engineHelperRow,
} from "../src/features/attached-engines/engine-helpers-state.ts";
import type {
  ComfyuiQueueRow,
  ComfyuiStatus,
} from "../src/features/attached-engines/types.ts";

function comfy(patch: Partial<ComfyuiStatus> = {}): ComfyuiStatus {
  return {
    url: "http://127.0.0.1:8844",
    reachable: true,
    state: "idle",
    version: "0.39.0",
    queueRunning: 0,
    queuePending: 0,
    devices: [{ name: "Apple M5 Pro", type: "mps" }],
    ramTotal: 64 * 1024 ** 3,
    ramFree: 30 * 1024 ** 3,
    failure: null,
    helperWanted: null,
    peers: [],
    ...patch,
  };
}

const down = (patch: Partial<ComfyuiStatus> = {}) =>
  comfy({ reachable: false, state: "down", version: null, devices: [], ramFree: null, ...patch });

function helper(patch: Partial<EngineHelperRow> = {}): EngineHelperRow {
  return { checked: true, message: null, isError: false, state: "enabled", ...patch };
}

const line = (status: ComfyuiStatus | null | undefined, row: EngineHelperRow | null = null) =>
  comfyuiStatusLine({ status, helper: row });

test("idle, generating and queued", () => {
  assert.deepEqual(line(comfy()), { text: "Idle", tone: "ok" });
  assert.deepEqual(
    line(comfy({ state: "busy", queueRunning: 1, queuePending: 2 })),
    { text: "Generating (2 queued)", tone: "busy" },
  );
  assert.equal(
    line(comfy({ state: "busy", queueRunning: 1, queuePending: 0 })).text,
    "Generating (0 queued)",
  );
  assert.equal(
    line(comfy({ state: "busy", queueRunning: 0, queuePending: 1 })).text,
    "Queued (1)",
  );
});

test("an unreadable queue is unknown, not idle", () => {
  const unknown = line(comfy({ reachable: false, state: "unknown" }));
  assert.equal(unknown.tone, "warn");
  assert.match(unknown.text, /^Unknown/);
});

test("not running, starting and waiting for approval", () => {
  assert.deepEqual(line(down()), { text: "Not running", tone: "muted" });
  // the user wants it and the helper is registered: it is on its way up
  assert.equal(line(down(), helper()).text, "Starting");
  assert.equal(line(down({ helperWanted: true })).text, "Starting");
  // wanted but the master switch is off, so nothing is registered
  assert.equal(line(down(), helper({ state: "not_registered" })).text, "Not running");
  // not wanted
  assert.equal(line(down(), helper({ checked: false, state: "not_registered" })).text, "Not running");
  assert.equal(line(down({ helperWanted: false })).text, "Not running");
  assert.deepEqual(line(down(), helper({ state: "requires_approval" })), {
    text: "Waiting for approval in Login Items",
    tone: "warn",
  });
});

test("a launch failure shows its reason while ComfyUI is down, and never over a live one", () => {
  const failing = down({ failure: { reason: "venv missing", ts: 1, count: 1 } });
  assert.deepEqual(line(failing, helper()), {
    text: "ComfyUI failing: venv missing",
    tone: "error",
  });
  assert.match(
    line(down({ failure: { reason: "venv missing", ts: 1, count: 3 } })).text,
    /\(failed 3 times\)/,
  );
  // a stale marker next to a healthy ComfyUI is ignored
  assert.equal(line(comfy({ failure: { reason: "old", ts: 1, count: 1 } })).text, "Idle");
});

test("the StoryPress port case names the culprit and the way out", () => {
  const status = down({
    failure: {
      reason: "port 8844 (ComfyUI) is held by StoryPress's ComfyUI (pid 7)",
      ts: 1,
      count: 1,
    },
  });
  const text = failureText(status) ?? "";
  assert.match(text, /held by StoryPress's ComfyUI/);
  assert.ok(text.endsWith(STORYPRESS_PORT_HINT));
  assert.equal(failureText(down()), null);
  assert.equal(
    failureText(down({ failure: { reason: "model dir missing", ts: 1, count: 1 } })),
    "ComfyUI failing: model dir missing",
  );
});

test("no comfyui block (an older backend) reads as unavailable", () => {
  assert.deepEqual(line(null), { text: "Status unavailable", tone: "muted" });
  assert.deepEqual(line(undefined), { text: "Status unavailable", tone: "muted" });
});

test("unload is offered only on an idle ComfyUI and never interrupts", () => {
  assert.deepEqual(comfyuiUnloadRule(comfy()), { enabled: true, reason: null });
  assert.deepEqual(comfyuiUnloadRule(down()), { enabled: false, reason: null });
  assert.deepEqual(comfyuiUnloadRule(null), { enabled: false, reason: null });
  const busy = comfyuiUnloadRule(comfy({ state: "busy", queueRunning: 1 }));
  assert.equal(busy.enabled, false);
  assert.match(busy.reason ?? "", /nothing is interrupted/);
  const unknown = comfyuiUnloadRule(comfy({ reachable: false, state: "unknown" }));
  assert.equal(unknown.enabled, false);
  assert.match(unknown.reason ?? "", /queue/);
});

test("a deferred free is told apart from a done one", () => {
  assert.equal(
    freeResultMessage({ freed: true, deferred: true }),
    "Memory will be freed when the current job finishes.",
  );
  assert.match(freeResultMessage({ freed: true, deferred: false }), /unloaded/);
});

test("devices and free memory", () => {
  assert.equal(comfyuiDeviceLine(comfy()), "Apple M5 Pro (mps), 30.0 GB free");
  assert.equal(comfyuiDeviceLine(comfy({ devices: [], ramFree: null })), null);
  assert.equal(comfyuiDeviceLine(comfy({ devices: [] })), "30.0 GB free");
  assert.equal(formatBytes(512 * 1024 ** 2), "512 MB");
  assert.equal(formatBytes(null), "-");
});

test("a reachable StoryPress peer is called out, and a generating one explains the wait", () => {
  assert.deepEqual(peerNotes([]), []);
  assert.deepEqual(
    peerNotes([{ url: "http://127.0.0.1:8188", reachable: false, busy: false, state: "down" }]),
    [],
  );
  assert.deepEqual(
    peerNotes([{ url: "http://127.0.0.1:8188", reachable: true, busy: false, state: "idle" }]),
    ["StoryPress ComfyUI is also running on :8188 (shares memory)."],
  );
  const busy = peerNotes([
    { url: "http://127.0.0.1:8188", reachable: true, busy: true, state: "busy" },
  ]);
  assert.match(busy[0], /is generating/);
  assert.match(busy[0], /Studio waits/);
  assert.match(
    peerNotes([{ url: "http://127.0.0.1:9000", reachable: true, busy: false, state: "idle" }])[0],
    /^Another ComfyUI is also running on :9000/,
  );
});

test("queue rows: titles, details and pending positions", () => {
  const rows: ComfyuiQueueRow[] = [
    { promptId: "aaaaaaaa-1111", number: 1, state: "running", studio: true },
    { promptId: "bbbbbbbb-2222", number: 2, state: "pending", studio: false },
    { promptId: "cc", number: 3, state: "pending", studio: false },
  ];
  assert.equal(queueRowTitle(rows[0]), "Studio job aaaaaaaa");
  assert.equal(queueRowTitle(rows[1]), "Job bbbbbbbb");
  assert.equal(queueRowTitle(rows[2]), "Job cc");
  assert.equal(queueRowDetail(rows[0], 0), "Running");
  assert.equal(pendingPosition(rows, rows[1]), 1);
  assert.equal(pendingPosition(rows, rows[2]), 2);
  assert.equal(queueRowDetail(rows[2], pendingPosition(rows, rows[2])), "Queued, position 2");
});

test("cancelling a job that already finished is not an error", () => {
  assert.equal(cancelOutcome(new EngineHttpError("Job not found in the ComfyUI queue.", 404)), "gone");
  assert.equal(cancelOutcome(new EngineHttpError("ComfyUI is not reachable.", 502)), "error");
  assert.equal(cancelOutcome(new Error("boom")), "error");
});

test("models are grouped by type in a fixed order, empty folders left out", () => {
  const groups = groupComfyuiModels({
    loras: ["l.safetensors"],
    vae: ["v.safetensors"],
    diffusion_models: ["d1.safetensors", "d2.safetensors"],
    checkpoints: [],
    text_encoders: ["t.safetensors"],
    zeta: ["z"],
    alpha: ["a"],
  });
  assert.deepEqual(
    groups.map((g) => [g.label, g.files.length]),
    [
      ["Diffusion models", 2],
      ["Text encoders", 1],
      ["VAE", 1],
      ["LoRA", 1],
      ["alpha", 1],
      ["zeta", 1],
    ],
  );
  assert.equal(modelCountText(groups), "7 model files");
  assert.equal(modelCountText(groupComfyuiModels({})), "No models found");
  assert.equal(modelCountText(groupComfyuiModels({ vae: ["v"] })), "1 model file");
});

test("the Run ComfyUI switch follows the helper row", () => {
  const status = (extra: Partial<EngineHelpersStatus["helpers"][number]> = {}): EngineHelpersStatus => ({
    supported: true,
    state: "not_registered",
    cli: 2,
    engines_enabled: true,
    error: null,
    helpers: [
      { name: "omlx", plist: "o", state: "enabled", wanted: true },
      { name: "comfyui", plist: "c", state: "not_registered", wanted: false, ...extra },
    ],
  });
  const off = engineHelperRow(status(), "comfyui");
  assert.equal(off?.checked, false);
  // the first-time Login Items note is shown while it is off
  assert.match(off?.message ?? "", /Login Items/);
  assert.equal(engineHelperRow(status({ state: "enabled", wanted: true }), "comfyui")?.message, null);
  const approval = engineHelperRow(status({ state: "requires_approval", wanted: true }), "comfyui");
  assert.equal(approval?.checked, true);
  assert.match(approval?.message ?? "", /Approve Unsloth/);
});

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_RETRY_AFTER_S,
  EngineBusyError,
  EngineHttpError,
  busyMessage,
  describeEngineFailure,
  parseRetryAfter,
} from "../src/features/attached-engines/engine-errors.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("Retry-After parses seconds only", () => {
  assert.equal(parseRetryAfter("15"), 15);
  assert.equal(parseRetryAfter(" 30 "), 30);
  for (const bad of [null, undefined, "", "0", "-5", "soon", "Wed, 21 Oct 2026 07:28:00 GMT", "1.5"]) {
    assert.equal(parseRetryAfter(bad), null, String(bad));
  }
});

test("a busy engine reads as a retry hint with the server's own wording", () => {
  const busy = new EngineBusyError("oMLX is busy serving another client; retry shortly.", 15);
  assert.equal(
    busyMessage(busy),
    "oMLX is busy serving another client; retry shortly. Try again in about 15 s.",
  );
  assert.equal(new EngineBusyError("", null).retryAfterS, DEFAULT_RETRY_AFTER_S);
  assert.equal(busyMessage(new EngineBusyError("", null)), "Busy right now. Try again in about 15 s.");
  assert.equal(new EngineBusyError("x", 0).retryAfterS, DEFAULT_RETRY_AFTER_S);
});

test("describeEngineFailure keeps busy and error apart", () => {
  assert.deepEqual(describeEngineFailure(new EngineBusyError("ComfyUI is generating", 20), "x"), {
    kind: "busy",
    message: "ComfyUI is generating. Try again in about 20 s.",
  });
  assert.deepEqual(describeEngineFailure(new EngineHttpError("ComfyUI is not reachable.", 502), "x"), {
    kind: "error",
    message: "ComfyUI is not reachable.",
  });
  assert.deepEqual(describeEngineFailure("boom", "Could not do it"), {
    kind: "error",
    message: "Could not do it",
  });
  assert.deepEqual(describeEngineFailure(new Error(""), "Could not do it"), {
    kind: "error",
    message: "Could not do it",
  });
});

test("the request helper turns a 503 into EngineBusyError and everything else into EngineHttpError", () => {
  const api = src("features/attached-engines/api.ts");
  assert.match(api, /res\.status === 503/);
  assert.match(api, /new EngineBusyError\(detail, parseRetryAfter\(res\.headers\.get\("Retry-After"\)\)\)/);
  assert.match(api, /throw new EngineHttpError\(detail, res\.status\)/);
});

test("the oMLX panel and the ComfyUI section show a busy reply as information, not an error toast", () => {
  const panel = src("features/attached-engines/engines-panel.tsx");
  assert.match(panel, /shown\.kind === "busy"\) toast\.info\(shown\.message\)/);
  assert.match(panel, /else toast\.error\(shown\.message\)/);
  const section = src("features/attached-engines/comfyui-section.tsx");
  assert.match(section, /kind: shown\.kind === "busy" \? "info" : "error"/);
});

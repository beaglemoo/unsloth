// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  failureFromApi,
  failureMessage,
} from "../src/features/attached-engines/failure.ts";

test("a complete marker parses and trims the reason", () => {
  assert.deepEqual(
    failureFromApi({ reason: " port 8843 already in use ", ts: 1760000000.5, count: 3 }),
    { reason: "port 8843 already in use", ts: 1760000000.5, count: 3 },
  );
});

test("anything incomplete is no failure", () => {
  for (const raw of [
    null,
    undefined,
    "oops",
    [],
    {},
    { reason: "", ts: 1, count: 1 },
    { reason: "  ", ts: 1, count: 1 },
    { reason: 5, ts: 1, count: 1 },
    { reason: "x", ts: 1 },
    { reason: "x", ts: 1, count: 0 },
    { reason: "x", ts: 1, count: 1.5 },
    { reason: "x", ts: 1, count: "2" },
  ]) {
    assert.equal(failureFromApi(raw), null, JSON.stringify(raw));
  }
});

test("a missing timestamp does not hide the reason", () => {
  assert.deepEqual(failureFromApi({ reason: "venv missing", count: 1 }), {
    reason: "venv missing",
    ts: 0,
    count: 1,
  });
});

test("the message names the engine and the repeat count", () => {
  assert.equal(
    failureMessage("oMLX", { reason: "venv missing", ts: 1, count: 1 }),
    "oMLX failing: venv missing",
  );
});

test("the panel shows the failure instead of the generic not-reachable text", () => {
  const panel = readFileSync(
    new URL("../src/features/attached-engines/engines-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(panel, /failureMessage\("oMLX", status\.failure\)/);
  const api = readFileSync(
    new URL("../src/features/attached-engines/api.ts", import.meta.url),
    "utf8",
  );
  assert.equal(api.match(/failure: failureFromApi\(raw\.failure\)/g)?.length, 1);
});

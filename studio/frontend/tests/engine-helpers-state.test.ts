// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type EngineHelperState,
  type EngineHelpersStatus,
  engineHelpersToggle,
} from "../src/features/attached-engines/engine-helpers-state.ts";

function status(
  state: EngineHelperState,
  extra: Partial<EngineHelpersStatus> = {},
): EngineHelpersStatus {
  return { supported: true, state, helpers: [], error: null, ...extra };
}

test("the row stays hidden outside the desktop shell or without SMAppService", () => {
  assert.equal(engineHelpersToggle(null), null);
  assert.equal(
    engineHelpersToggle(status("unsupported", { supported: false })),
    null,
  );
});

test("enabled and not registered map to the switch position", () => {
  assert.deepEqual(engineHelpersToggle(status("enabled")), {
    checked: true,
    message: null,
    isError: false,
  });
  assert.deepEqual(engineHelpersToggle(status("not_registered")), {
    checked: false,
    message: null,
    isError: false,
  });
});

test("pending approval keeps the switch on and explains Login Items", () => {
  const toggle = engineHelpersToggle(status("requires_approval"));
  assert.equal(toggle?.checked, true);
  assert.equal(toggle?.isError, false);
  assert.match(toggle?.message ?? "", /Login Items/);
});

test("a registration error is reported as a failure", () => {
  const toggle = engineHelpersToggle(
    status("not_registered", { error: "omlx: Operation not permitted" }),
  );
  assert.equal(toggle?.checked, false);
  assert.equal(toggle?.isError, true);
  assert.match(toggle?.message ?? "", /Operation not permitted/);
});

test("a build without the bundled agents says so", () => {
  const toggle = engineHelpersToggle(status("not_found"));
  assert.equal(toggle?.checked, false);
  assert.match(toggle?.message ?? "", /does not include/);
});

test("a half-registered pair stays on with guidance", () => {
  const toggle = engineHelpersToggle(status("partial"));
  assert.equal(toggle?.checked, true);
  assert.match(toggle?.message ?? "", /some of the engine helpers/);
});

test("bundled but unregistered helpers offer the enable switch, not a missing-build notice", () => {
  const toggle = engineHelpersToggle(
    status("not_registered", {
      helpers: [
        { name: "omlx", plist: "ai.unsloth.studio.omlx.plist", state: "not_registered" },
      ],
    }),
  );
  assert.deepEqual(toggle, { checked: false, message: null, isError: false });
});

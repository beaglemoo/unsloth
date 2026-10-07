// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  LOGIN_ITEMS_NOTE,
  type EngineHelpersStatus,
  engineHelperRow,
} from "../src/features/attached-engines/engine-helpers-state.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

function status(patch: Partial<EngineHelpersStatus> = {}): EngineHelpersStatus {
  return {
    supported: true,
    state: "enabled",
    cli: 2,
    engines_enabled: true,
    error: null,
    helpers: [
      { name: "omlx", plist: "ai.unsloth.studio.omlx.plist", state: "enabled", wanted: true },
      { name: "comfyui", plist: "ai.unsloth.studio.comfyui.plist", state: "not_registered", wanted: false },
    ],
    ...patch,
  };
}

test("the per-engine switch is hidden in a browser, without SMAppService or without cli 2", () => {
  assert.equal(engineHelperRow(null, "comfyui"), null);
  assert.equal(engineHelperRow(status({ supported: false }), "comfyui"), null);
  assert.equal(engineHelperRow(status({ cli: undefined }), "comfyui"), null);
  assert.equal(engineHelperRow(status({ cli: 1 }), "comfyui"), null);
  // a build that lists no such helper
  assert.equal(engineHelperRow(status({ helpers: [] }), "comfyui"), null);
});

test("the switch shows the choice (wanted), not the registration", () => {
  assert.equal(engineHelperRow(status(), "omlx")?.checked, true);
  assert.equal(engineHelperRow(status(), "comfyui")?.checked, false);
  // wanted but not registered yet (starting): still on
  const starting = status({
    helpers: [
      { name: "comfyui", plist: "c", state: "not_registered", wanted: true },
    ],
  });
  assert.equal(engineHelperRow(starting, "comfyui")?.checked, true);
});

test("ComfyUI off tells the user about Login Items, oMLX off does not", () => {
  assert.equal(engineHelperRow(status(), "comfyui")?.message, LOGIN_ITEMS_NOTE);
  assert.match(LOGIN_ITEMS_NOTE, /first time/);
  assert.equal(engineHelperRow(status(), "omlx")?.message, null);
});

test("with the master switch off the row says why nothing runs", () => {
  const masterOff = status({
    engines_enabled: false,
    helpers: [
      { name: "omlx", plist: "o", state: "not_registered", wanted: false },
      { name: "comfyui", plist: "c", state: "not_registered", wanted: false },
    ],
  });
  assert.match(engineHelperRow(masterOff, "comfyui")?.message ?? "", /turned off above/);
  assert.match(engineHelperRow(masterOff, "omlx")?.message ?? "", /turned off above/);
});

test("a registration error and a missing helper are reported", () => {
  const failed = engineHelperRow(status({ error: "comfyui: Operation not permitted" }), "comfyui");
  assert.equal(failed?.isError, true);
  assert.match(failed?.message ?? "", /Operation not permitted/);
  const missing = engineHelperRow(
    status({ helpers: [{ name: "comfyui", plist: "c", state: "not_found", wanted: false }] }),
    "comfyui",
  );
  assert.match(missing?.message ?? "", /does not include/);
});

test("the toggle calls engine_helper_set_enabled with the engine name", () => {
  const api = src("features/attached-engines/engine-helpers-api.ts");
  assert.match(api, /call\("engine_helper_set_enabled", \{ name, enabled \}\)/);
  assert.match(api, /name: EngineName/);
  const section = src("features/attached-engines/comfyui-section.tsx");
  assert.match(section, /setEngineHelperEnabled\("comfyui", next\)/);
  assert.match(section, /engineHelperRow\(helpers, "comfyui"\)/);
  // the switch only renders when the row exists, so a browser never sees it
  assert.match(section, /\{helperRow \? \(\s+<SettingsRow\s+label="Run ComfyUI"/);
  assert.match(
    src("features/attached-engines/settings-section.tsx"),
    /setEngineHelperEnabled\("omlx", enabled\)/,
  );
});

test("the Rust command and the status fields the UI reads exist", () => {
  const rust = readFileSync(
    new URL("../../src-tauri/src/engine_helpers.rs", import.meta.url),
    "utf8",
  );
  assert.match(rust, /pub\(crate\) async fn engine_helper_set_enabled\(\s+name: String,\s+enabled: bool,/);
  assert.match(rust, /wanted: bool,/);
  assert.match(rust, /cli: u32,/);
});

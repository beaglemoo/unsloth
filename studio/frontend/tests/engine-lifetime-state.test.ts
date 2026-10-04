// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { EngineHelpersStatus } from "../src/features/attached-engines/engine-helpers-state.ts";
import {
  ENGINE_LIFETIME_OPTIONS,
  engineLifetimeNote,
  engineLifetimeOf,
  engineLifetimeRow,
  enginesEnabledDescription,
  isEngineLifetime,
} from "../src/features/attached-engines/engine-lifetime-state.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

function status(extra: Partial<EngineHelpersStatus> = {}): EngineHelpersStatus {
  return {
    supported: true,
    state: "enabled",
    helpers: [],
    error: null,
    ...extra,
  };
}

test("the two options carry the labels the settings spec asks for", () => {
  assert.deepEqual(
    ENGINE_LIFETIME_OPTIONS.map((o) => [o.value, o.label]),
    [
      ["with_app", "With Unsloth (stop when Unsloth quits)"],
      ["always", "Always running (keep serving after Unsloth quits)"],
    ],
  );
});

test("only the two known words are lifetimes", () => {
  assert.equal(isEngineLifetime("with_app"), true);
  assert.equal(isEngineLifetime("always"), true);
  for (const bad of ["", "withApp", "forever", null, undefined, 1]) {
    assert.equal(isEngineLifetime(bad), false);
  }
});

test("the control follows what the shell reports", () => {
  assert.equal(
    engineLifetimeOf(status({ engine_lifetime: "with_app" })),
    "with_app",
  );
  assert.equal(
    engineLifetimeOf(status({ engine_lifetime: "always" })),
    "always",
  );
  assert.equal(engineLifetimeOf(null), "with_app");
});

test("a shell from before the setting behaves as always running", () => {
  assert.equal(engineLifetimeOf(status()), "always");
});

test("the row is hidden outside the desktop shell and without SMAppService", () => {
  assert.equal(engineLifetimeRow(null), null);
  assert.equal(
    engineLifetimeRow(status({ supported: false, state: "unsupported" })),
    null,
  );
});

test("the row offers both options with the selected one and its note", () => {
  const row = engineLifetimeRow(status({ engine_lifetime: "with_app" }));
  assert.equal(row?.value, "with_app");
  assert.equal(row?.options.length, 2);
  assert.equal(row?.note, engineLifetimeNote("with_app"));
  const always = engineLifetimeRow(status({ engine_lifetime: "always" }));
  assert.equal(always?.value, "always");
  assert.equal(always?.note, engineLifetimeNote("always"));
});

test("the trade-off names the clients that lose local models while Unsloth is closed", () => {
  const note = engineLifetimeNote("with_app");
  for (const client of ["pi", "Claude Code", "OpenCode"]) {
    assert.match(note, new RegExp(client));
  }
  assert.match(note, /only get local models while Unsloth is open/);
  assert.match(
    engineLifetimeNote("always"),
    /keep their local models after Unsloth quits/,
  );
});

test("the Engines enabled description matches the mode", () => {
  assert.match(enginesEnabledDescription("with_app"), /stop when it quits/);
  assert.doesNotMatch(enginesEnabledDescription("with_app"), /keep serving/);
  assert.match(
    enginesEnabledDescription("always"),
    /keep serving after Unsloth quits/,
  );
});

test("the settings section mounts the control behind the flag, next to the Engines enabled switch", () => {
  const section = src("features/attached-engines/settings-section.tsx");
  const gate = section.indexOf("settings?.enabled ? (");
  const enabled = section.indexOf('label="Engines enabled"');
  const lifetime = section.indexOf('label="Engine lifetime"');
  assert.ok(gate > 0 && enabled > gate && lifetime > enabled);
  assert.ok(lifetime < section.indexOf('label="oMLX URL"'));
  assert.doesNotMatch(section, /label="Background engines"/);
  assert.match(section, /setEngineLifetime\(value\)/);
  assert.match(
    src("features/attached-engines/engine-helpers-api.ts"),
    /call\("engine_lifetime_set", \{ lifetime \}\)/,
  );
});

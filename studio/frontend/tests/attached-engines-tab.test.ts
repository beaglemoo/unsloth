// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { engineContextTargets } from "../src/features/attached-engines/engine-context-targets.ts";
import {
  resolveSettingsTab,
  settingsTabVisible,
} from "../src/features/settings/settings-tab-visibility.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";
import type {
  AttachedDs4Status,
  AttachedOmlxModel,
  AttachedStatus,
} from "../src/features/attached-engines/types.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

function row(patch: Partial<AttachedOmlxModel>): AttachedOmlxModel {
  return {
    id: "Swift",
    modelPath: "/m/Swift",
    loaded: false,
    isLoading: false,
    estimatedSize: 1,
    pinned: false,
    engineType: "vlm",
    isHelper: false,
    modelAlias: null,
    maxContextWindow: null,
    modelContextLength: null,
    maxTokens: null,
    ttlS: null,
    idleRemainingS: null,
    ...patch,
  };
}

function ds4(patch: Partial<AttachedDs4Status> = {}): AttachedDs4Status {
  return {
    reachable: true,
    loaded: false,
    starting: false,
    pid: null,
    uptimeS: null,
    inFlight: 0,
    idleRemainingS: null,
    liveTps: null,
    lastGenTps: null,
    lastTtftMs: null,
    startTimeoutS: 120,
    ctx: null,
    ctxActive: null,
    pendingRestart: false,
    error: null,
    failure: null,
    ...patch,
  };
}

function status(
  rows: AttachedOmlxModel[],
  engine: AttachedDs4Status | null,
  omlxReachable = true,
  chatModelIds: string[] = rows.map((r) => r.id),
): AttachedStatus {
  return {
    omlx: {
      reachable: omlxReachable,
      models: rows,
      memoryBytes: 0,
      ceilingBytes: 0,
      error: null,
      chatModelIds,
      failure: null,
    },
    ds4: engine,
    modelsHash: "",
    notices: [],
    receivedAt: 1,
  };
}

test("Attached engines is a registered settings tab, owner-only", () => {
  assert.ok((SETTINGS_TABS as readonly string[]).includes("attached-engines"));
  assert.equal(settingsTabVisible("attached-engines", true), true);
  assert.equal(settingsTabVisible("attached-engines", false), false);
  assert.equal(resolveSettingsTab("attached-engines", false), "general");
  assert.equal(resolveSettingsTab("attached-engines", true), "attached-engines");
});

test("the dialog loads, lists, labels and indexes the tab", () => {
  const dialog = src("features/settings/settings-dialog.tsx");
  assert.match(dialog, /"attached-engines": \(\) =>\s+import\("\.\/tabs\/attached-engines-tab"\)/);
  assert.match(dialog, /id: "attached-engines",\s+labelKey: "settings\.tabs\.attachedEngines"/);
  assert.match(dialog, /"attached-engines": null/);
  assert.match(src("components/command-palette.tsx"), /"attached-engines": "settings\.tabs\.attachedEngines"/);
  assert.match(src("features/settings/settings-search.ts"), /"attached-engines": \[/);
  assert.match(src("i18n/locales/en.ts"), /attachedEngines: "Attached engines"/);
});

test("the engine pieces moved off the API Keys and Resources tabs onto the new one", () => {
  const apiKeys = src("features/settings/tabs/api-keys-tab.tsx");
  const resources = src("features/settings/tabs/resources-tab.tsx");
  for (const text of [apiKeys, resources]) {
    assert.doesNotMatch(text, /attached-engines/);
    assert.doesNotMatch(text, /AttachedEngines/);
  }
  const tab = src("features/settings/tabs/attached-engines-tab.tsx");
  assert.match(tab, /<AttachedEnginesSettingsSection \/>/);
  assert.match(tab, /<AttachedEnginesPanel \/>/);
  assert.match(tab, /<AttachedEnginesContextSection \/>/);
});

test("with the flag off only the enable toggle is rendered", () => {
  const section = src("features/attached-engines/settings-section.tsx");
  const enable = section.indexOf('label="Enable attached engines"');
  const gate = section.indexOf("settings?.enabled ? (");
  assert.ok(enable > 0 && gate > enable);
  // everything else sits behind the flag, the Background engines switch included
  assert.ok(section.indexOf('label="Background engines"') > gate);
  assert.ok(section.indexOf('label="oMLX URL"') > gate);
  assert.match(src("features/attached-engines/engines-panel.tsx"), /if \(!enabled\) return null;/);
  assert.match(
    src("features/attached-engines/engine-context-section.tsx"),
    /if \(!enabled\) return null;/,
  );
});

test("the chat settings Provider section keeps its own context control", () => {
  const sheet = src("features/chat/chat-settings-sheet.tsx");
  assert.match(sheet, /AttachedContextControl/);
});

test("context targets: one per oMLX chat model, plus DwarfStar", () => {
  const targets = engineContextTargets(
    status(
      [
        row({ id: "Swift", modelPath: "/m/Swift" }),
        row({ id: "swift:fast", modelPath: "/m/Swift" }),
        row({ id: "Gemma", modelPath: "/m/Gemma" }),
        row({ id: "Embed", modelPath: "/m/Embed", engineType: "embedding" }),
      ],
      ds4(),
      true,
      ["Swift", "swift:fast", "Gemma"],
    ),
  );
  assert.deepEqual(
    targets.map((t) => [t.kind, t.label]),
    [
      ["omlx", "Swift"],
      ["omlx", "Gemma"],
      ["dwarfstar", "DwarfStar"],
    ],
  );
});

test("context targets skip an engine that cannot be read", () => {
  assert.deepEqual(engineContextTargets(null), []);
  assert.deepEqual(
    engineContextTargets(status([row({})], ds4({ reachable: false }), false)),
    [],
  );
  const onlyDs4 = engineContextTargets(status([row({})], ds4(), false));
  assert.deepEqual(onlyDs4.map((t) => t.kind), ["dwarfstar"]);
});

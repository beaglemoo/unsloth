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

function status(
  rows: AttachedOmlxModel[],
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
    modelsHash: "",
    notices: [],
    receivedAt: 1,
  };
}

test("the Engines tab (id attached-engines) is a registered settings tab, owner-only", () => {
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
  assert.match(src("i18n/locales/en.ts"), /attachedEngines: "Engines"/);
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
  assert.match(tab, /<OmlxDashboardSection \/>/);
});

test("the tab adds the ComfyUI section and its web UI after the oMLX sections, and keeps the helper status fresh", () => {
  const tab = src("features/settings/tabs/attached-engines-tab.tsx");
  const order = [
    "<AttachedEnginesSettingsSection />",
    "<AttachedEnginesPanel />",
    "<AttachedEnginesContextSection />",
    "<OmlxDashboardSection />",
    "<ComfyuiSection />",
    "<ComfyuiWebUiSection />",
  ].map((tag) => tab.indexOf(tag));
  assert.ok(order.every((at) => at > 0), String(order));
  assert.deepEqual([...order].sort((a, b) => a - b), order);
  assert.match(tab, /useEngineHelpers\(\);/);
  assert.match(tab, /oMLX and ComfyUI run beside Studio/);
  assert.match(src("features/attached-engines/engines-panel.tsx"), /title="oMLX"/);
});

test("the ComfyUI section follows the flag, offers Unload and shows the queue, models and settings", () => {
  const section = src("features/attached-engines/comfyui-section.tsx");
  assert.match(section, /if \(!enabled\) return null;/);
  // hooks are all called before the early return
  assert.ok(section.indexOf("useEffect(") < section.indexOf("if (!enabled) return null;"));
  assert.match(section, /label="Unload models"/);
  assert.match(section, /disabled=\{!rule\.enabled \|\| unloading\}/);
  assert.match(section, /freeResultMessage\(result\)/);
  assert.match(section, /<QueueRows rows=\{queue\} onChange=\{loadQueue\} \/>/);
  assert.match(section, /<ModelsList reachable=/);
  assert.match(section, /label="ComfyUI URL"/);
  assert.match(section, /label="Arbitrate ComfyUI memory"/);
  assert.match(section, /label="Free ComfyUI after idle \(s\)"/);
  // one poller: the section reads the shared status, it never starts its own timer
  assert.doesNotMatch(section, /setInterval|createAttachedPoller/);
  assert.match(section, /Interrupt/);
});

test("the ComfyUI settings reach the settings API", () => {
  const api = src("features/attached-engines/api.ts");
  for (const key of ["comfyui_url", "comfyui_peer_urls", "arbitrate_comfyui", "comfyui_idle_free_s"]) {
    assert.match(api, new RegExp(`"${key}"`));
  }
});

test("with the flag off only the enable toggle is rendered", () => {
  const section = src("features/attached-engines/settings-section.tsx");
  const enable = section.indexOf('label="Show engines in Studio"');
  const gate = section.indexOf("settings?.enabled ? (");
  assert.ok(enable > 0 && gate > enable);
  // everything else sits behind the flag, the Engines enabled switch included
  assert.ok(section.indexOf('label="Engines enabled"') > gate);
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

test("context targets: one per oMLX chat model", () => {
  const targets = engineContextTargets(status([row({ id: "Swift" }), row({ id: "Embed", engineType: "embedding" })]));
  assert.deepEqual(targets.map((t) => [t.kind, t.label]), [["omlx", "Swift"]]);
});

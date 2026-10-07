// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  OMLX_DASHBOARD_EXPANDED_KEY,
  omlxDashboardUrl,
  omlxDashboardView,
  readDashboardExpanded,
  writeDashboardExpanded,
} from "../src/features/attached-engines/omlx-dashboard-state.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
const rust = (path: string) =>
  readFileSync(new URL(`../../src-tauri/src/${path}`, import.meta.url), "utf8");

test("the settings tab is labelled Engines everywhere the user sees it, with its id unchanged", () => {
  assert.ok((SETTINGS_TABS as readonly string[]).includes("attached-engines"));
  assert.match(src("i18n/locales/en.ts"), /attachedEngines: "Engines"/);
  const dialog = src("features/settings/settings-dialog.tsx");
  assert.match(dialog, /id: "attached-engines",\s+labelKey: "settings\.tabs\.attachedEngines"/);
  for (const file of ["ar", "de", "en", "es", "fr", "he", "hi", "it", "ja", "ko", "pt-br", "ru", "sv", "zh-CN"]) {
    assert.doesNotMatch(src(`i18n/locales/${file}.ts`), /attachedEngines: "oMLX"/, file);
    assert.match(src(`i18n/locales/${file}.ts`), /attachedEngines: "[^"]+"/, file);
  }
  assert.match(
    rust("app_menu.rs"),
    /Row::Action\("settings-attached-engines", "Engines", ""\)/,
  );
  assert.doesNotMatch(rust("app_menu.rs"), /"Attached engines"/);
  assert.match(
    src("features/chat/attached-connections.ts"),
    /ATTACHED_CONNECTIONS_LINK = "Engines"/,
  );
});

test("the dashboard URL is built from the configured oMLX base URL", () => {
  assert.equal(
    omlxDashboardUrl("http://127.0.0.1:8843"),
    "http://127.0.0.1:8843/admin/dashboard",
  );
  assert.equal(
    omlxDashboardUrl(" http://localhost:9000/ "),
    "http://localhost:9000/admin/dashboard",
  );
  assert.equal(omlxDashboardUrl(""), null);
  assert.equal(omlxDashboardUrl(null), null);
  assert.equal(omlxDashboardUrl("javascript:alert(1)"), null);
  const section = src("features/attached-engines/omlx-dashboard-section.tsx");
  assert.match(section, /omlxUrl: settings\?\.omlxUrl/);
});

test("the dashboard renders only with the flag on and oMLX reachable", () => {
  const url = "http://127.0.0.1:8843";
  assert.deepEqual(omlxDashboardView({ enabled: false, reachable: true, omlxUrl: url }), {
    kind: "hidden",
  });
  assert.deepEqual(omlxDashboardView({ enabled: false, reachable: false, omlxUrl: url }), {
    kind: "hidden",
  });
  assert.deepEqual(omlxDashboardView({ enabled: true, reachable: false, omlxUrl: url }), {
    kind: "unreachable",
  });
  assert.deepEqual(omlxDashboardView({ enabled: true, reachable: true, omlxUrl: "" }), {
    kind: "unreachable",
  });
  assert.deepEqual(omlxDashboardView({ enabled: true, reachable: true, omlxUrl: url }), {
    kind: "ready",
    url: `${url}/admin/dashboard`,
  });
});

test("the section gates on the flag and reachability, and frames lazily", () => {
  const section = src("features/attached-engines/omlx-dashboard-section.tsx");
  assert.match(section, /enabled: settings\?\.enabled === true/);
  assert.match(section, /reachable: status\?\.omlx\?\.reachable === true/);
  assert.match(section, /view\.kind === "hidden"\) return null/);
  assert.match(section, /oMLX is not running/);
  // the iframe sits behind the expanded disclosure and the ready view
  const frame = section.indexOf("<iframe");
  assert.ok(frame > section.indexOf("expanded ? ("));
  assert.ok(frame > section.indexOf('view.kind === "unreachable"'));
  assert.match(section, /useState\(\(\) => readDashboardExpanded\(\)\)/);
  assert.match(section, /title="oMLX dashboard"/);
  assert.match(section, /referrerPolicy="no-referrer"/);
  assert.match(section, /h-\[70vh\] min-h-\[calc\(520px\*var\(--ui-space-scale,1\)\)\] w-full/);
  assert.doesNotMatch(section, /sandbox/);
  assert.match(section, /key=\{reloadKey\}/);
  assert.match(section, /Reload/);
  assert.match(section, /openLink\(view\.url\)/);
  assert.match(section, /Open in browser/);
  // the tab mounts it last
  const tab = src("features/settings/tabs/attached-engines-tab.tsx");
  assert.ok(tab.indexOf("<OmlxDashboardSection />") > tab.indexOf("<AttachedEnginesContextSection />"));
});

test("the expanded state is remembered and survives broken storage", () => {
  const store = new Map<string, string>();
  const storage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
  };
  assert.equal(readDashboardExpanded(storage), false);
  writeDashboardExpanded(true, storage);
  assert.equal(store.get(OMLX_DASHBOARD_EXPANDED_KEY), "1");
  assert.equal(readDashboardExpanded(storage), true);
  writeDashboardExpanded(false, storage);
  assert.equal(readDashboardExpanded(storage), false);
  const broken = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("denied");
    },
  };
  assert.equal(readDashboardExpanded(broken), false);
  assert.doesNotThrow(() => writeDashboardExpanded(true, broken));
  assert.equal(readDashboardExpanded(null), false);
});

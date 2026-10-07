// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  COMFYUI_WEBUI_EXPANDED_KEY,
  comfyuiWebUiFrameable,
  comfyuiWebUiUrl,
  comfyuiWebUiView,
  readWebUiExpanded,
  writeWebUiExpanded,
} from "../src/features/attached-engines/comfyui-webui-state.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("only loopback http URLs are ever framed", () => {
  assert.equal(comfyuiWebUiUrl("http://127.0.0.1:8844"), "http://127.0.0.1:8844/");
  assert.equal(comfyuiWebUiUrl(" http://127.0.0.1:8844/ "), "http://127.0.0.1:8844/");
  assert.equal(comfyuiWebUiUrl("http://localhost:9000"), "http://localhost:9000/");
  assert.equal(comfyuiWebUiUrl("http://127.0.0.1"), "http://127.0.0.1/");
  for (const bad of [
    "",
    null,
    undefined,
    "https://127.0.0.1:8844",
    "http://example.com:8844",
    "http://192.168.2.5:8844",
    "http://127.0.0.1.evil.com:8844",
    "http://127.0.0.1:8844/../x",
    "javascript:alert(1)",
    "http://127.0.0.1@evil.com:8844",
  ]) {
    assert.equal(comfyuiWebUiUrl(bad), null, String(bad));
  }
});

test("the IPv6 loopback is linked but not framed (the CSP does not allow it)", () => {
  assert.equal(comfyuiWebUiFrameable("http://[::1]:8844/"), false);
  assert.equal(comfyuiWebUiFrameable("http://127.0.0.1:8844/"), true);
  assert.deepEqual(
    comfyuiWebUiView({ enabled: true, reachable: true, comfyuiUrl: "http://[::1]:8844" }),
    { kind: "link-only", url: "http://[::1]:8844/" },
  );
});

test("the web UI renders only with the flag on and ComfyUI reachable", () => {
  const view = (enabled: boolean, reachable: boolean, comfyuiUrl: string | null) =>
    comfyuiWebUiView({ enabled, reachable, comfyuiUrl });
  assert.deepEqual(view(false, true, "http://127.0.0.1:8844"), { kind: "hidden" });
  assert.deepEqual(view(true, false, "http://127.0.0.1:8844"), { kind: "unreachable" });
  assert.deepEqual(view(true, true, "http://evil.example"), { kind: "unreachable" });
  assert.deepEqual(view(true, true, "http://127.0.0.1:8844"), {
    kind: "ready",
    url: "http://127.0.0.1:8844/",
  });
});

test("collapsed by default, remembered when expanded, storage failures read as collapsed", () => {
  const store = new Map<string, string>();
  const storage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
  };
  assert.equal(COMFYUI_WEBUI_EXPANDED_KEY, "unsloth.comfyuiWebUi.expanded");
  assert.equal(readWebUiExpanded(storage), false);
  writeWebUiExpanded(true, storage);
  assert.equal(store.get(COMFYUI_WEBUI_EXPANDED_KEY), "1");
  assert.equal(readWebUiExpanded(storage), true);
  writeWebUiExpanded(false, storage);
  assert.equal(readWebUiExpanded(storage), false);
  const broken = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("full");
    },
  };
  assert.equal(readWebUiExpanded(broken), false);
  assert.doesNotThrow(() => writeWebUiExpanded(true, broken));
  assert.equal(readWebUiExpanded(null), false);
});

test("the section frames nothing until expanded and offers reload and open in browser", () => {
  const section = src("features/attached-engines/comfyui-webui-section.tsx");
  assert.match(section, /useState\(\(\) => readWebUiExpanded\(\)\)/);
  assert.match(section, /view\.kind === "ready" && expanded \? \(\s+<iframe/);
  assert.match(section, /Reload/);
  assert.match(section, /openLink\(view\.url\)/);
  assert.match(section, /title="ComfyUI web UI"/);
  assert.match(section, /if \(view\.kind === "hidden"\) return null;/);
});

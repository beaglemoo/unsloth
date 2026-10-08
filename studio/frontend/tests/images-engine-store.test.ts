// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { comfyuiAvailable } from "../src/features/images/comfyui/comfyui-panel-state.ts";
import {
  IMAGE_ENGINE_KEY,
  effectiveEngine,
  persistEngine,
  readStoredEngine,
} from "../src/features/images/comfyui/engine-choice.ts";
import { readSrc } from "./helpers/kit.ts";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
  };
}

test("the engine defaults to Studio", () => {
  assert.equal(readStoredEngine(fakeStorage()), "studio");
  assert.equal(readStoredEngine(null), "studio");
  assert.equal(readStoredEngine(fakeStorage({ [IMAGE_ENGINE_KEY]: "junk" })), "studio");
});

test("the choice is remembered under unsloth.images.engine", () => {
  assert.equal(IMAGE_ENGINE_KEY, "unsloth.images.engine");
  const storage = fakeStorage();
  persistEngine("comfyui", storage);
  assert.equal(storage.data.get("unsloth.images.engine"), "comfyui");
  assert.equal(readStoredEngine(storage), "comfyui");
  persistEngine("studio", storage);
  assert.equal(readStoredEngine(storage), "studio");
});

test("a storage that throws never breaks the page", () => {
  const broken = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("denied");
    },
  };
  assert.equal(readStoredEngine(broken), "studio");
  assert.doesNotThrow(() => persistEngine("comfyui", broken));
});

test("ComfyUI mode needs ComfyUI to be available and the Create page mode", () => {
  assert.equal(effectiveEngine("comfyui", true, "create"), "comfyui");
  assert.equal(effectiveEngine("comfyui", false, "create"), "studio");
  assert.equal(effectiveEngine("comfyui", true, "train"), "studio");
  assert.equal(effectiveEngine("studio", true, "create"), "studio");
});

test("the engine switch is offered to the owner with the flag on and ComfyUI reachable or wanted", () => {
  const up = { reachable: true, helperWanted: null };
  const wanted = { reachable: false, helperWanted: true };
  const off = { reachable: false, helperWanted: false };
  const unknown = { reachable: false, helperWanted: null };
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: true, status: up }), true);
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: true, status: wanted }), true);
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: true, status: off }), false);
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: true, status: unknown }), false);
  // hidden for a non-owner, with the flag off, and before the first status
  assert.equal(comfyuiAvailable({ isOwner: false, enabled: true, status: up }), false);
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: false, status: up }), false);
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: true, status: null }), false);
  assert.equal(comfyuiAvailable({ isOwner: true, enabled: true, status: undefined }), false);
});

test("the workflow store holds the engine and persists it through the helper", () => {
  const store = readSrc("features/images/stores/image-workflow-store.ts");
  assert.match(store, /engine: readStoredEngine\(\)/);
  assert.match(store, /persistEngine\(engine\)/);
});

test("the Images page keeps ComfyUI behind one panel and one effective-engine conditional", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /effectiveEngine\(engine, comfyAvailable \|\| comfyRunning, pageMode\)/);
  assert.equal(page.match(/<ComfyuiCreatePanel/g)?.length, 1);
  // the engine switch only renders while ComfyUI is available
  assert.match(page, /comfyAvailable && pageMode === "create" && \(\s*<PillTabs/);
  // one writer for the sidebar's supported list: ComfyUI publishes Create only
  assert.match(page, /if \(comfyMode\) \{\s*setSupported\(\["create"\]\);/);
  assert.equal(page.match(/setSupported\(/g)?.length, 3);
  // Studio's footer (Generate, Stop, Reapply) is replaced, not shared, in ComfyUI mode
  assert.match(page, /\{!comfyMode && \(\s*<div className="relative z-10 flex shrink-0 flex-wrap justify-center gap-2/);
});

test("the panel never publishes the supported workflows itself", () => {
  const panel = readSrc("features/images/comfyui/comfyui-create-panel.tsx");
  assert.doesNotMatch(panel, /setSupported/);
  // its only poller is the progress poll that runs while a job does
  assert.equal(panel.match(/setInterval\(/g)?.length, 1);
  assert.match(panel, /startPolling\(\);\s*try \{/);
  assert.match(panel, /finally \{\s*stopPolling\(\);/);
});

test("the Engines tab names the show switch apart from Engines enabled", () => {
  const section = readSrc("features/attached-engines/settings-section.tsx");
  assert.match(section, /label="Show engines in Studio"/);
  assert.doesNotMatch(section, /label="Enable engines"/);
  assert.match(section, /label="Engines enabled"/);
});

test("the ComfyUI panel cannot shrink below its controls, so the sticky Generate bar never covers Add LoRA", () => {
  const panel = readSrc("features/images/comfyui/comfyui-create-panel.tsx");
  const wrapper = panel.match(/<div className="([^"]*)" data-comfyui-panel="">/);
  assert.ok(wrapper, "panel wrapper found");
  assert.ok(!wrapper[1].split(" ").includes("min-h-0"), "wrapper must not be shrinkable");
  assert.match(panel, /sticky bottom-0/);
  // Studio mode keeps its Generate footer outside the settings scroller.
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /\{!comfyMode && \(\s*<div className="relative z-10 flex shrink-0 flex-wrap/);
});

test("the ComfyUI panel offers the sampler lists ComfyUI reports, not the static ones", () => {
  const panel = readSrc("features/images/comfyui/comfyui-create-panel.tsx");
  assert.match(panel, /options=\{samplerOptions\.samplers\}/);
  assert.match(panel, /options=\{samplerOptions\.schedulers\}/);
  assert.doesNotMatch(panel, /options=\{SAMPLER_OPTIONS\}|options=\{SCHEDULER_OPTIONS\}/);
  assert.match(readSrc("features/images/comfyui/api.ts"), /\/samplers/);
});

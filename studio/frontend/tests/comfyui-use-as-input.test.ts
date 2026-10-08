// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

// use-as-input.ts imports the store extensionless, so the bundler resolver goes in first.
registerBundlerResolver();
const { useComfyPanelStore } = await import("../src/features/images/comfyui/comfyui-panel-store.ts");
const { requestComfyInput } = await import("../src/features/images/comfyui/use-as-input.ts");

test("requestComfyInput parks the gallery image for the panel", () => {
  useComfyPanelStore.getState().clearPendingInput();
  requestComfyInput({ id: "fox_1", url: "/api/inference/images/gallery/fox_1/file", width: 1024, height: 768 });
  assert.deepEqual(useComfyPanelStore.getState().pendingInput, {
    galleryId: "fox_1",
    url: "/api/inference/images/gallery/fox_1/file",
    width: 1024,
    height: 768,
  });
  useComfyPanelStore.getState().clearPendingInput();
});

test("the viewer shows Use as input only in ComfyUI mode, beside Recipe", () => {
  const page = readSrc("features/images/images-page.tsx");
  const at = page.indexOf('aria-label="Use as input"');
  assert.ok(at > 0);
  const block = page.slice(page.lastIndexOf("{comfyMode && (", at), page.indexOf("<RecipePopover image={selected}", at));
  assert.match(block, /requestComfyInput\(selected\)/);
  assert.match(block, /ImageUpload01Icon/);
  assert.equal(page.split("requestComfyInput(").length - 1, 1);
});

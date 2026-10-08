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

test("a stale gallery fetch never overwrites a newer input and its URL is revoked", () => {
  const revoked: string[] = [];
  const original = URL.revokeObjectURL;
  URL.revokeObjectURL = (url: string) => void revoked.push(url);
  try {
    const store = useComfyPanelStore.getState();
    store.clearInputs();
    const older = store.beginInput("image");
    const newer = store.beginInput("image");
    const gallery = (id: string) => ({ kind: "gallery" as const, galleryId: id, previewUrl: `blob:${id}`, width: 1, height: 1 });
    // The newer fetch finishes first, then the older one.
    assert.equal(store.setInputIfCurrent("image", newer, gallery("new")), true);
    assert.equal(store.setInputIfCurrent("image", older, gallery("old")), false);
    assert.deepEqual(revoked, ["blob:old"]);
    const held = useComfyPanelStore.getState().inputs.image;
    assert.equal(held?.kind === "gallery" ? held.galleryId : null, "new");
  } finally {
    URL.revokeObjectURL = original;
    useComfyPanelStore.getState().clearInputs();
  }
});

test("a user setInput or clearInputs invalidates pending fetches for that slot only", () => {
  const original = URL.revokeObjectURL;
  URL.revokeObjectURL = () => undefined;
  try {
    const store = useComfyPanelStore.getState();
    store.clearInputs();
    const gallery = (id: string) => ({ kind: "gallery" as const, galleryId: id, previewUrl: `blob:${id}`, width: 1, height: 1 });
    const a = store.beginInput("image");
    const b = store.beginInput("second");
    store.setInput("image", { kind: "data", dataUrl: "data:image/png;base64,AA", width: 1, height: 1 });
    assert.equal(store.isInputCurrent("image", a), false);
    assert.equal(store.setInputIfCurrent("image", a, gallery("late")), false);
    assert.equal(useComfyPanelStore.getState().inputs.image?.kind, "data");
    assert.equal(store.isInputCurrent("second", b), true);
    store.clearInputs();
    assert.equal(store.isInputCurrent("second", b), false);
    assert.equal(store.setInputIfCurrent("second", b, gallery("later")), false);
    assert.deepEqual(useComfyPanelStore.getState().inputs, {});
    const fresh = store.beginInput("image");
    assert.equal(store.setInputIfCurrent("image", fresh, gallery("ok")), true);
  } finally {
    URL.revokeObjectURL = original;
    useComfyPanelStore.getState().clearInputs();
  }
});

test("the panel fills gallery inputs through the token guard", () => {
  const panel = readSrc("features/images/comfyui/comfyui-create-panel.tsx");
  assert.match(panel, /beginInput\(slot\)/);
  assert.match(panel, /setInputIfCurrent\(slot, token/);
});

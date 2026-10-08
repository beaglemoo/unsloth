// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ComfyRecallSource,
  applyRecall,
  isComfyRecord,
  recallFromImage,
  templatesFromApi,
} from "../src/features/images/comfyui/comfyui-panel-state.ts";
import { readSrc } from "./helpers/kit.ts";

const TEMPLATE = {
  id: "qwen-image-2.1-t2i",
  name: "Qwen-Image 2.1",
  defaults: { steps: 25, cfg: 1, width: 1024, height: 1024, sampler: "euler", scheduler: "simple" },
  limits: { steps: [1, 100], cfg: [0, 20], side: [256, 2048], multiple: 16, batch_size: [1, 4] },
  slots: ["prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "width", "height", "batch_size"],
  supports_lora: true,
  required_models: {},
  missing_models: {},
};
const templates = templatesFromApi({ templates: [TEMPLATE, { ...TEMPLATE, id: "user:other", name: "Other", source: "user" }] }).templates;

function record(patch: Partial<ComfyRecallSource> = {}): ComfyRecallSource {
  return {
    engine: "comfyui",
    comfyui_template: "qwen-image-2.1-t2i",
    prompt: "a cabin in snow",
    negative_prompt: "blurry",
    width: 1344,
    height: 768,
    steps: 30,
    guidance: 2,
    seed: 12,
    batch_seed: 10,
    batch_size: 2,
    loras: ["style.safetensors:0.7"],
    sampler: "dpmpp_2m",
    scheduler: "karras",
    ...patch,
  };
}

test("only a ComfyUI record recalls into ComfyUI", () => {
  assert.equal(isComfyRecord({ engine: "comfyui" }), true);
  assert.equal(isComfyRecord({ engine: null }), false);
  assert.equal(isComfyRecord({}), false);
  assert.equal(recallFromImage(record({ engine: undefined })), null);
  // a ComfyUI record with no template cannot be recalled
  assert.equal(recallFromImage(record({ comfyui_template: null })), null);
});

test("a ComfyUI record carries its template and parameters", () => {
  const recall = recallFromImage(record());
  assert.ok(recall);
  assert.equal(recall.templateId, "qwen-image-2.1-t2i");
  assert.deepEqual(recall.params, {
    prompt: "a cabin in snow",
    negativePrompt: "blurry",
    width: 1344,
    height: 768,
    steps: 30,
    cfg: 2,
    // the batch's base seed, not the image's derived one
    seed: "10",
    batchSize: 2,
    loras: [{ name: "style.safetensors", strength: 0.7 }],
    sampler: "dpmpp_2m",
    scheduler: "karras",
  });
});

test("without a batch seed the image seed is used", () => {
  assert.equal(recallFromImage(record({ batch_seed: null }))?.params.seed, "12");
});

test("applying a recall selects its template and clamps the values to it", () => {
  const recall = recallFromImage(record({ width: 1000, steps: 500 }));
  assert.ok(recall);
  const applied = applyRecall(recall, templates);
  assert.ok(applied);
  assert.equal(applied.found, true);
  assert.equal(applied.template.id, "qwen-image-2.1-t2i");
  assert.equal(applied.params.width, 992);
  assert.equal(applied.params.steps, 100);
  assert.equal(applied.params.prompt, "a cabin in snow");
  assert.equal(applied.params.sampler, "dpmpp_2m");
});

test("a recall of a deleted template falls back to the first with the values kept", () => {
  const recall = recallFromImage(record({ comfyui_template: "user:gone" }));
  assert.ok(recall);
  const applied = applyRecall(recall, templates);
  assert.ok(applied);
  assert.equal(applied.found, false);
  assert.equal(applied.template.id, "qwen-image-2.1-t2i");
  assert.equal(applied.params.prompt, "a cabin in snow");
  assert.equal(applyRecall(recall, []), null);
});

test("an unset sampler in the record keeps the template default", () => {
  const recall = recallFromImage(record({ sampler: null, scheduler: null }));
  assert.ok(recall);
  const applied = applyRecall(recall, templates);
  assert.equal(applied?.params.sampler, "euler");
  assert.equal(applied?.params.scheduler, "simple");
});

test("the page routes a ComfyUI record to the panel and any other to Studio", () => {
  const page = readSrc("features/images/images-page.tsx");
  const start = page.indexOf("const restoreFromRecipe = useCallback");
  const wrapper = page.slice(start, page.indexOf("[comfyAvailable, restoreSettings, setEngine, setWorkflow]", start));
  assert.match(wrapper, /recallFromImage\(image\)/);
  // with ComfyUI unavailable the recall is parked and the owner is told, not silently dropped
  assert.match(wrapper, /comfyRecall && !comfyAvailable\) \{\s*\/\/[^\n]*\n\s*useComfyPanelStore\.getState\(\)\.requestRecall\(comfyRecall\);\s*toast\.info\(/);
  assert.match(page, /prev === "generating" \? null : prev/);
  assert.match(wrapper, /setEngine\("comfyui"\)/);
  assert.match(wrapper, /requestRecall\(comfyRecall\)/);
  assert.match(wrapper, /setEngine\("studio"\);\s*restoreSettings\(image\)/);
  // the Recipe popover restores through the wrapper, and Studio's own restore is untouched
  assert.match(page, /<RecipePopover image=\{selected\} onRestore=\{restoreFromRecipe\}/);
  const restore = page.slice(page.indexOf("const restoreSettings = useCallback"), start);
  assert.doesNotMatch(restore, /recallFromImage|setEngine|ComfyPanel/);
});

// ---------------------------------------------------------------- phase 7: image inputs

test("an img2img record restores denoise and its gallery input", () => {
  const recall = recallFromImage(
    record({ workflow: "img2img", strength: 0.55, comfyui_inputs: { image: "fox_1" } }),
  );
  assert.ok(recall);
  assert.equal(recall.params.denoise, 0.55);
  assert.equal(recall.params.referenceResolution, undefined);
  assert.deepEqual(recall.inputs, { image: "fox_1" });
});

test("an edit record restores the reference resolution", () => {
  const recall = recallFromImage(
    record({ workflow: "edit", reference_resolution: 768, comfyui_inputs: { image: "fox_1" } }),
  );
  assert.ok(recall);
  assert.equal(recall.params.referenceResolution, 768);
  assert.equal(recall.params.denoise, undefined);
});

test("an upload record has no gallery id to re-attach", () => {
  const recall = recallFromImage(record({ workflow: "img2img", strength: 0.6, comfyui_inputs: { image: null } }));
  assert.deepEqual(recall?.inputs, { image: null });
});

test("a text-to-image record carries no image state", () => {
  const recall = recallFromImage(record());
  assert.ok(recall);
  assert.deepEqual(recall.inputs, {});
  assert.equal("denoise" in recall.params, false);
  assert.equal("referenceResolution" in recall.params, false);
});

test("the panel store keeps inputs by slot and revokes a replaced gallery preview", async () => {
  const revoked: string[] = [];
  const original = URL.revokeObjectURL;
  URL.revokeObjectURL = (u: string) => void revoked.push(u);
  try {
    const { useComfyPanelStore } = await import("../src/features/images/comfyui/comfyui-panel-store.ts");
    const store = useComfyPanelStore.getState();
    const a = { kind: "gallery", galleryId: "a", previewUrl: "blob:a", width: 1, height: 1 } as const;
    const b = { kind: "data", dataUrl: "data:x", width: 1, height: 1 } as const;
    store.setInput("image", a);
    store.setInput("image", b);
    assert.deepEqual(revoked, ["blob:a"]);
    assert.equal(useComfyPanelStore.getState().inputs.image, b);
    store.setInput("image", { ...a, previewUrl: "blob:c" });
    store.clearInputs();
    assert.deepEqual(revoked, ["blob:a", "blob:c"]);
    assert.deepEqual(useComfyPanelStore.getState().inputs, {});
    store.requestInput({ galleryId: "g", url: "/u", width: 2, height: 3 });
    assert.equal(useComfyPanelStore.getState().pendingInput?.galleryId, "g");
    store.clearPendingInput();
    assert.equal(useComfyPanelStore.getState().pendingInput, null);
  } finally {
    URL.revokeObjectURL = original;
  }
});

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
  const restore = page.slice(page.indexOf("const restoreSettings = useCallback"));
  const head = restore.slice(0, restore.indexOf("setNegativePrompt(restoredNegative)"));
  assert.match(head, /recallFromImage\(image\)/);
  assert.match(head, /setEngine\("comfyui"\)/);
  assert.match(head, /requestRecall\(comfyRecall\)/);
  assert.match(head, /setEngine\("studio"\)/);
});

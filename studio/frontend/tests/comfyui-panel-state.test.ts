// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FALLBACK_SAMPLER_OPTIONS,
  cancelOutcomeFromApi,
  SAMPLER_OPTIONS,
  SCHEDULER_OPTIONS,
  type ComfyParams,
  type ComfyTemplate,
  MAX_LORAS,
  buildGenerateRequest,
  clampNumber,
  defaultParams,
  describeComfyFailure,
  fixedSamplingNote,
  hasSlot,
  loraRowsFromRecipe,
  missingModelList,
  parseGraphJson,
  parseSeed,
  progressFromApi,
  queueLabel,
  randomSeed,
  reconcileParams,
  samplerOptionsFromApi,
  snapSide,
  templatesFromApi,
} from "../src/features/images/comfyui/comfyui-panel-state.ts";

const QWEN = {
  id: "qwen-image-2.1-t2i",
  name: "Qwen-Image 2.1 (text to image)",
  kind: "t2i",
  source: "shipped",
  defaults: { steps: 25, cfg: 1.0, width: 1024, height: 1024, sampler: "euler", scheduler: "simple" },
  limits: { steps: [1, 100], cfg: [0, 20], side: [256, 2048], multiple: 16, batch_size: [1, 4] },
  slots: [
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "width",
    "height",
    "batch_size",
  ],
  supports_lora: true,
  required_models: { diffusion_models: ["qwen_image_2.1_bf16.safetensors"], vae: ["vae.safetensors"] },
  missing_models: {},
};

function qwen(patch: Partial<typeof QWEN> = {}): ComfyTemplate {
  const parsed = templatesFromApi({ comfyui_reachable: true, templates: [{ ...QWEN, ...patch }] });
  return parsed.templates[0];
}

function params(patch: Partial<ComfyParams> = {}): ComfyParams {
  return { ...defaultParams(qwen()), prompt: "a cabin", ...patch };
}

test("the templates response parses to camel case, with reachability", () => {
  const parsed = templatesFromApi({ comfyui_reachable: true, templates: [QWEN] });
  assert.equal(parsed.reachable, true);
  const [t] = parsed.templates;
  assert.equal(t.id, "qwen-image-2.1-t2i");
  assert.equal(t.supportsLora, true);
  assert.deepEqual(t.limits.side, [256, 2048]);
  assert.equal(t.limits.multiple, 16);
  assert.deepEqual(t.limits.batchSize, [1, 4]);
  assert.equal(t.defaults.sampler, "euler");
  assert.deepEqual(t.missingModels, {});
});

test("missing_models null (ComfyUI down) stays null and lists nothing", () => {
  const t = qwen({ missing_models: null as unknown as Record<string, never> });
  assert.equal(t.missingModels, null);
  assert.deepEqual(missingModelList(t), []);
});

test("missing models list as folder/name", () => {
  const t = qwen({ missing_models: { vae: ["vae.safetensors"] } as never });
  assert.deepEqual(missingModelList(t), ["vae/vae.safetensors"]);
});

test("junk in the templates response is tolerated", () => {
  assert.deepEqual(templatesFromApi(null), { reachable: false, templates: [] });
  const parsed = templatesFromApi({ templates: [null, { id: "" }, { id: "x" }, 4] });
  assert.deepEqual(parsed.templates.map((t) => t.id), ["x"]);
  // a template with no limits falls back to the shipped defaults
  assert.deepEqual(parsed.templates[0].limits.side, [256, 2048]);
  assert.equal(parsed.templates[0].supportsLora, false);
});

test("sides snap down to the multiple and stay inside the range", () => {
  const { limits } = qwen();
  assert.equal(snapSide(1030, limits), 1024);
  assert.equal(snapSide(1040, limits), 1040);
  assert.equal(snapSide(100, limits), 256);
  assert.equal(snapSide(5000, limits), 2048);
  assert.equal(snapSide(Number.NaN, limits), 256);
  // a range whose floor is not a multiple still lands on a multiple
  assert.equal(snapSide(260, { ...limits, side: [250, 2048] }), 256);
});

test("clampNumber keeps a value inside its range", () => {
  assert.equal(clampNumber(150, [1, 100]), 100);
  assert.equal(clampNumber(-1, [0, 20]), 0);
  assert.equal(clampNumber(Number.NaN, [1, 100]), 1);
});

test("defaults come from the template", () => {
  const p = defaultParams(qwen());
  assert.equal(p.steps, 25);
  assert.equal(p.cfg, 1);
  assert.equal(p.width, 1024);
  assert.equal(p.sampler, "euler");
  assert.equal(p.scheduler, "simple");
  assert.equal(p.seed, "");
  assert.deepEqual(p.loras, []);
});

test("reconcile clamps numbers, snaps sides and trims LoRAs", () => {
  const rows = Array.from({ length: 6 }, (_, i) => ({ name: `l${i}.safetensors`, strength: 1 }));
  const out = reconcileParams(
    params({ steps: 500, cfg: 99, width: 1001, height: 99999, batchSize: 9, loras: rows }),
    qwen(),
  );
  assert.equal(out.steps, 100);
  assert.equal(out.cfg, 20);
  assert.equal(out.width, 992);
  assert.equal(out.height, 2048);
  assert.equal(out.batchSize, 4);
  assert.equal(out.loras.length, MAX_LORAS);
  // a template without LoRA support drops them
  assert.deepEqual(
    reconcileParams(params({ loras: rows.slice(0, 1) }), qwen({ supports_lora: false })).loras,
    [],
  );
});

test("seed text: empty is random, digits are exact, anything else is invalid", () => {
  assert.equal(parseSeed(""), null);
  assert.equal(parseSeed("  "), null);
  assert.equal(parseSeed("42"), 42);
  assert.equal(parseSeed("-1"), undefined);
  assert.equal(parseSeed("1.5"), undefined);
  assert.equal(parseSeed("abc"), undefined);
  assert.equal(parseSeed(String(2 ** 53)), undefined);
});

test("randomize makes a numeric seed the field accepts", () => {
  assert.equal(randomSeed(() => 0), "0");
  const seed = randomSeed(() => 0.999999);
  assert.ok(parseSeed(seed) !== undefined && parseSeed(seed) !== null);
});

test("the request carries every slot the template binds", () => {
  const built = buildGenerateRequest(
    qwen(),
    params({
      prompt: "  a cabin  ",
      negativePrompt: "blurry",
      seed: "7",
      steps: 30,
      cfg: 2,
      width: 1344,
      height: 768,
      batchSize: 2,
      loras: [{ name: " style.safetensors ", strength: 0.8 }, { name: "", strength: 1 }],
    }),
  );
  assert.ok(built.ok);
  assert.deepEqual(built.body, {
    template_id: "qwen-image-2.1-t2i",
    prompt: "a cabin",
    negative_prompt: "blurry",
    width: 1344,
    height: 768,
    seed: 7,
    steps: 30,
    cfg: 2,
    sampler: "euler",
    scheduler: "simple",
    batch_size: 2,
    loras: [{ name: "style.safetensors", strength: 0.8 }],
  });
});

test("an empty seed is left out so the backend picks one", () => {
  const built = buildGenerateRequest(qwen(), params({ seed: "" }));
  assert.ok(built.ok);
  assert.equal("seed" in built.body, false);
});

test("slots the template does not bind are not sent", () => {
  const slim = qwen({ slots: ["prompt", "seed", "steps"] });
  const built = buildGenerateRequest(
    slim,
    params({ negativePrompt: "x", width: 512, cfg: 5, sampler: "heun" }),
  );
  assert.ok(built.ok);
  for (const key of ["negative_prompt", "width", "height", "cfg", "sampler", "scheduler"]) {
    assert.equal(key in built.body, false, key);
  }
  assert.equal(built.body.steps, 25);
});

test("an empty prompt or a bad seed is refused before any request", () => {
  assert.deepEqual(buildGenerateRequest(qwen(), params({ prompt: "   " })), {
    ok: false,
    error: "Prompt is empty",
  });
  const bad = buildGenerateRequest(qwen(), params({ seed: "abc" }));
  assert.equal(bad.ok, false);
});

test("the request is clamped to the template limits", () => {
  const built = buildGenerateRequest(qwen(), params({ steps: 999, width: 5000, cfg: -3 }));
  assert.ok(built.ok);
  assert.equal(built.body.steps, 100);
  assert.equal(built.body.width, 2048);
  assert.equal(built.body.cfg, 0);
});

test("LoRA recipe entries split on the last colon", () => {
  assert.deepEqual(loraRowsFromRecipe(["style.safetensors:0.8", "a:b.safetensors:1.5", "bad", ":2"]), [
    { name: "style.safetensors", strength: 0.8 },
    { name: "a:b.safetensors", strength: 1.5 },
  ]);
  assert.deepEqual(loraRowsFromRecipe(undefined), []);
});

test("503 off points at Settings > Engines", () => {
  const f = describeComfyFailure({
    status: 503,
    code: "off",
    detail: "ComfyUI is off, enable it in Settings > Engines.",
  });
  assert.equal(f.kind, "off");
  assert.equal(f.openSettings, true);
  assert.match(f.message, /Settings > Engines/);
});

test("503 admission is a calm retry with the Retry-After seconds and an unload offer", () => {
  const f = describeComfyFailure({
    status: 503,
    code: "admission",
    detail: "A Studio image model is loaded.",
    retryAfterS: 15,
  });
  assert.equal(f.kind, "busy");
  assert.equal(f.retryAfterS, 15);
  assert.equal(f.offerUnload, true);
  assert.match(f.message, /Try again in about 15 s\./);
  assert.doesNotMatch(f.message, /\.\./);
  // no header: the default applies
  assert.equal(
    describeComfyFailure({ status: 503, code: "admission", detail: "x", retryAfterS: null }).retryAfterS,
    15,
  );
  assert.equal(
    describeComfyFailure({ status: 503, code: "admission", detail: "x", retryAfterS: 30 }).retryAfterS,
    30,
  );
});

test("409 busy and 409 cancelled are told apart by the code", () => {
  assert.equal(describeComfyFailure({ status: 409, code: "busy", detail: "A job is running." }).kind, "busy");
  const cancelled = describeComfyFailure({ status: 409, code: "cancelled", detail: "Generation cancelled." });
  assert.equal(cancelled.kind, "cancelled");
  assert.match(cancelled.message, /Nothing was saved/);
});

test("422 missing models, 502 not running, 404 template, 504 timeout", () => {
  assert.equal(
    describeComfyFailure({ status: 422, code: "missing_models", detail: "ComfyUI is missing model files: vae/x." }).kind,
    "missing",
  );
  const down = describeComfyFailure({ status: 502, code: "not_running", detail: "x" });
  assert.equal(down.kind, "not_running");
  assert.equal(down.openSettings, true);
  const gone = describeComfyFailure({ status: 404, code: "template", detail: "Template not found." });
  assert.equal(gone.reloadTemplates, true);
  assert.match(describeComfyFailure({ status: 504, code: "timeout", detail: "Timed out." }).message, /queue/);
});

test("execution and parameter errors show the backend detail", () => {
  for (const [status, code] of [
    [502, "execution"],
    [422, "params"],
    [422, "graph"],
  ] as const) {
    const f = describeComfyFailure({ status, code, detail: "ComfyUI failed in KSampler: boom" });
    assert.equal(f.kind, "error");
    assert.equal(f.message, "ComfyUI failed in KSampler: boom");
  }
});

test("an unknown 503 with no code still reads as a retry", () => {
  const f = describeComfyFailure({ status: 503, code: null, detail: "", retryAfterS: null });
  assert.equal(f.kind, "busy");
  assert.match(f.message, /15 s/);
  assert.equal(describeComfyFailure({ status: 500, code: null, detail: "" }).kind, "error");
});

test("progress parses to the Images progress shape plus the queue slot", () => {
  const p = progressFromApi({
    active: true,
    step: 3,
    total_steps: 25,
    fraction: 0.12,
    eta_seconds: 40.5,
    phase: "denoise",
    preview: null,
    preview_seq: 2,
    queue_position: 0,
    prompt_id: "abc",
    engine: "comfyui",
  });
  assert.equal(p.step, 3);
  assert.equal(p.phase, "denoise");
  assert.equal(p.queue_position, 0);
  assert.equal(p.prompt_id, "abc");
  const idle = progressFromApi(null);
  assert.equal(idle.active, false);
  assert.equal(idle.phase, null);
  assert.equal(progressFromApi({ phase: "weird", fraction: 4 }).fraction, 1);
});

test("queue labels only appear for a job that has not started", () => {
  assert.equal(queueLabel(0), null);
  assert.equal(queueLabel(null), null);
  assert.equal(queueLabel(1), "Next in ComfyUI's queue");
  assert.equal(queueLabel(3), "Position 3 in ComfyUI's queue");
});

test("graph import accepts an API-format object and rejects the rest", () => {
  const ok = parseGraphJson('{"1": {"class_type": "KSampler", "inputs": {}}}');
  assert.ok(ok.ok);
  assert.deepEqual(Object.keys(ok.graph), ["1"]);
  assert.equal(parseGraphJson("").ok, false);
  assert.equal(parseGraphJson("{nope").ok, false);
  assert.equal(parseGraphJson("[]").ok, false);
  assert.equal(parseGraphJson("{}").ok, false);
  const editor = parseGraphJson('{"nodes": [], "links": []}');
  assert.equal(editor.ok, false);
  assert.ok(!editor.ok && /API Format/.test(editor.error));
});

test("sampler lists come from ComfyUI and fall back list by list", () => {
  const live = samplerOptionsFromApi({
    samplers: ["euler", "res_multistep", "euler", "", 7],
    schedulers: ["simple", "kl_optimal"],
    source: "comfyui",
  });
  assert.deepEqual(live.samplers, ["euler", "res_multistep"]);
  assert.deepEqual(live.schedulers, ["simple", "kl_optimal"]);
  const partial = samplerOptionsFromApi({ samplers: [], schedulers: ["beta"] });
  assert.deepEqual(partial.samplers, SAMPLER_OPTIONS);
  assert.deepEqual(partial.schedulers, ["beta"]);
  assert.deepEqual(samplerOptionsFromApi(null), FALLBACK_SAMPLER_OPTIONS);
  assert.deepEqual(samplerOptionsFromApi({ samplers: "euler" }), FALLBACK_SAMPLER_OPTIONS);
  assert.deepEqual(FALLBACK_SAMPLER_OPTIONS.schedulers, SCHEDULER_OPTIONS);
});

// ---------------------------------------------------------------- phase 7: image inputs

import {
  type ComfyInput,
  REFERENCE_RESOLUTION_CHOICES,
  applyRecall,
  editOutputSize,
  keepsInputSize,
  pickTemplateForInput,
  recallFromImage,
  sizeForInput,
} from "../src/features/images/comfyui/comfyui-panel-state.ts";

const IMG_LIMITS = {
  steps: [1, 100],
  cfg: [0, 20],
  side: [256, 2048],
  multiple: 16,
  batch_size: [1, 1],
  denoise: [0.01, 1],
  reference_resolution: [256, 2048],
  reference_multiple: 32,
};
const IMG2IMG = {
  ...QWEN,
  id: "qwen-image-2.1-img2img",
  kind: "img2img",
  defaults: { ...QWEN.defaults, denoise: 0.8 },
  limits: IMG_LIMITS,
  slots: ["prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "width", "height", "denoise"],
  image_slots: [{ name: "image", label: "Input image", required: true }],
};
const EDIT = {
  ...QWEN,
  id: "qwen-image-2.1-edit",
  kind: "edit",
  defaults: { steps: 25, cfg: 1, sampler: "euler", scheduler: "simple", reference_resolution: 1024 },
  limits: IMG_LIMITS,
  slots: ["prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "reference_resolution"],
  image_slots: [{ name: "image", label: "Input image", required: true }],
};
const all = () => templatesFromApi({ templates: [QWEN, IMG2IMG, EDIT] }).templates;
const DATA: ComfyInput = { kind: "data", dataUrl: "data:image/png;base64,AAAA", width: 600, height: 400 };
const GALLERY: ComfyInput = { kind: "gallery", galleryId: "abc_123", previewUrl: "blob:x", width: 600, height: 400 };

test("image slots, kinds and the new defaults parse; an old payload still does", () => {
  const [, img, edit] = all();
  assert.equal(img.kind, "img2img");
  assert.deepEqual(img.imageSlots, [{ name: "image", label: "Input image", required: true }]);
  assert.equal(img.defaults.denoise, 0.8);
  assert.deepEqual(img.limits.denoise, [0.01, 1]);
  assert.equal(edit.defaults.referenceResolution, 1024);
  assert.equal(edit.limits.referenceMultiple, 32);
  const old = qwen();
  assert.deepEqual(old.imageSlots, []);
  assert.equal(old.defaults.denoise, null);
  assert.deepEqual(old.limits.denoise, [0.01, 1]);
  assert.equal(old.limits.referenceMultiple, 32);
  assert.equal(defaultParams(img).denoise, 0.8);
  assert.equal(defaultParams(edit).referenceResolution, 1024);
});

test("a request carries data or gallery inputs, and only the image controls the template binds", () => {
  const [t2i, img, edit] = all();
  const p = (t: ComfyTemplate) => ({ ...defaultParams(t), prompt: "a cabin" });
  const withData = buildGenerateRequest(img, p(img), { image: DATA });
  assert.ok(withData.ok);
  assert.deepEqual(withData.body.input_images, { image: { data: DATA.kind === "data" ? DATA.dataUrl : "" } });
  assert.equal(withData.body.denoise, 0.8);
  assert.equal(withData.body.reference_resolution, undefined);
  const withGallery = buildGenerateRequest(img, p(img), { image: GALLERY });
  assert.ok(withGallery.ok);
  assert.deepEqual(withGallery.body.input_images, { image: { gallery_id: "abc_123" } });
  const editReq = buildGenerateRequest(edit, p(edit), { image: DATA });
  assert.ok(editReq.ok);
  assert.equal(editReq.body.reference_resolution, 1024);
  assert.equal(editReq.body.denoise, undefined);
  assert.equal(editReq.body.width, undefined);
  const plain = buildGenerateRequest(t2i, p(t2i));
  assert.ok(plain.ok);
  assert.equal(plain.body.input_images, undefined);
  assert.equal(plain.body.denoise, undefined);
});

test("a required image slot without an input is refused after the prompt check", () => {
  const [, img] = all();
  const missing = buildGenerateRequest(img, { ...defaultParams(img), prompt: "x" }, { image: null });
  assert.deepEqual(missing, { ok: false, error: "Add an input image" });
  const noPrompt = buildGenerateRequest(img, defaultParams(img), {});
  assert.deepEqual(noPrompt, { ok: false, error: "Prompt is empty" });
});

test("sizeForInput keeps the template area at the input's aspect", () => {
  const [, img] = all();
  assert.deepEqual(sizeForInput(1500, 1000, img), { width: 1248, height: 832 });
  assert.deepEqual(sizeForInput(1000, 1000, img), { width: 1024, height: 1024 });
  assert.deepEqual(sizeForInput(100000, 10, img), { width: 2048, height: 256 });
  assert.deepEqual(sizeForInput(0, 0, img), { width: 1024, height: 1024 });
});

test("editOutputSize follows ComfyUI's resize to resolution squared in multiples of 32", () => {
  assert.deepEqual(editOutputSize(1000, 1000, 1024), { width: 1024, height: 1024 });
  assert.deepEqual(editOutputSize(1500, 1000, 1024), { width: 1248, height: 832 });
  assert.deepEqual(editOutputSize(1000, 1500, 768), { width: 640, height: 928 });
  assert.deepEqual(REFERENCE_RESOLUTION_CHOICES, [512, 768, 1024]);
});

test("editOutputSize with resolution 0 follows the input size in multiples of 32", () => {
  assert.deepEqual(editOutputSize(1000, 700, 0), { width: 992, height: 704 });
  assert.deepEqual(editOutputSize(0, 0, 0), { width: 0, height: 0 });
});

test("a template defaulting to reference resolution 0 keeps 0 in defaults, reconcile and recall", () => {
  const keep = templatesFromApi({
    templates: [{ ...EDIT, id: "user:keep", source: "user", defaults: { ...EDIT.defaults, reference_resolution: 0 } }],
  }).templates[0];
  const [, , edit] = all();
  assert.equal(keepsInputSize(keep), true);
  assert.equal(keepsInputSize(edit), false);
  assert.equal(defaultParams(keep).referenceResolution, 0);
  assert.equal(reconcileParams({ ...defaultParams(keep), referenceResolution: 0 }, keep).referenceResolution, 0);
  assert.equal(reconcileParams({ ...defaultParams(keep), referenceResolution: 700 }, keep).referenceResolution, 704);
  const recalled = applyRecall(
    { templateId: "user:keep", params: { referenceResolution: 0 }, inputs: {} },
    [keep],
  );
  assert.equal(recalled?.params.referenceResolution, 0);
  const sent = buildGenerateRequest(keep, { ...defaultParams(keep), prompt: "a fox" }, { image: GALLERY });
  assert.equal(sent.ok && sent.body.reference_resolution, 0);
  // A template that does not default to 0 still refuses it (the backend would reject it).
  assert.equal(reconcileParams({ ...defaultParams(edit), referenceResolution: 0 }, edit).referenceResolution, 256);
});

test("pickTemplateForInput keeps an image template, else prefers img2img", () => {
  const [t2i, img, edit] = all();
  assert.equal(pickTemplateForInput(edit, all())?.id, "qwen-image-2.1-edit");
  assert.equal(pickTemplateForInput(t2i, all())?.id, "qwen-image-2.1-img2img");
  assert.equal(pickTemplateForInput(null, [t2i, edit])?.id, "qwen-image-2.1-edit");
  assert.equal(pickTemplateForInput(t2i, [t2i]), null);
  assert.equal(pickTemplateForInput(img, [t2i]), img);
});

test("reconcileParams clamps denoise and snaps the reference resolution", () => {
  const [, img, edit] = all();
  assert.equal(reconcileParams({ ...defaultParams(img), denoise: 0 }, img).denoise, 0.01);
  assert.equal(reconcileParams({ ...defaultParams(img), denoise: 3 }, img).denoise, 1);
  assert.equal(reconcileParams({ ...defaultParams(edit), referenceResolution: 1000 }, edit).referenceResolution, 992);
  assert.equal(reconcileParams({ ...defaultParams(edit), referenceResolution: 99999 }, edit).referenceResolution, 2048);
});

test("the cancel answer separates no job, confirmed and pending (a slow ComfyUI)", () => {
  assert.equal(cancelOutcomeFromApi({ cancelled: false, confirmed: false }), "none");
  assert.equal(cancelOutcomeFromApi({ cancelled: true, confirmed: true }), "confirmed");
  assert.equal(cancelOutcomeFromApi({ cancelled: true, confirmed: false }), "pending");
  // an older backend: only `cancelled`
  assert.equal(cancelOutcomeFromApi({ cancelled: true }), "confirmed");
  assert.equal(cancelOutcomeFromApi({ cancelled: false }), "none");
  assert.equal(cancelOutcomeFromApi(null), "none");
  assert.equal(cancelOutcomeFromApi("x"), "none");
});

// ---------------------------------------------------------------- Turbo: a fixed schedule

const TURBO = {
  ...QWEN,
  id: "qwen-image-2.1-turbo-t2i",
  name: "Qwen-Image 2.1 Turbo (text to image, 8 steps)",
  defaults: { steps: 8, cfg: 1.0, width: 1024, height: 1024, sampler: "euler", scheduler: "manual" },
  limits: { steps: [8, 8], cfg: [1, 1], side: [256, 2048], multiple: 16, batch_size: [1, 4] },
  slots: ["prompt", "seed", "width", "height", "batch_size"],
};
const turbo = () => templatesFromApi({ templates: [TURBO] }).templates[0];

test("a template without steps, cfg, sampler and scheduler slots hides them and says why", () => {
  const t = turbo();
  for (const slot of ["steps", "cfg", "sampler", "scheduler", "negative_prompt"]) {
    assert.equal(hasSlot(t, slot), false, slot);
  }
  assert.equal(fixedSamplingNote(t), "Steps and sampler are fixed by this template (8 steps).");
  assert.equal(fixedSamplingNote(qwen()), null);
  // Only one of the three missing is not "fixed": the other control still shows.
  assert.equal(fixedSamplingNote(qwen({ slots: QWEN.slots.filter((s) => s !== "steps") })), null);
});

test("the Turbo request carries no steps, cfg, sampler, scheduler or negative prompt", () => {
  const t = turbo();
  const p = { ...defaultParams(t), prompt: "a fox", negativePrompt: "blur", seed: "5", steps: 50, cfg: 7, sampler: "dpmpp_2m", scheduler: "karras" };
  assert.equal(defaultParams(t).steps, 8);
  assert.equal(reconcileParams(p, t).steps, 8);
  const sent = buildGenerateRequest(t, reconcileParams(p, t), {});
  assert.ok(sent.ok);
  for (const key of ["steps", "cfg", "sampler", "scheduler", "negative_prompt"]) {
    assert.equal(key in sent.body, false, key);
  }
  assert.equal(sent.body.template_id, "qwen-image-2.1-turbo-t2i");
  assert.equal(sent.body.seed, 5);
  assert.equal(sent.body.width, 1024);
});

test("Recipe > Restore of a Turbo record finds the template and still sends no fixed values", () => {
  const t = turbo();
  const recall = recallFromImage({
    engine: "comfyui",
    comfyui_template: "qwen-image-2.1-turbo-t2i",
    prompt: "a fox",
    negative_prompt: null,
    width: 768,
    height: 512,
    steps: 8,
    guidance: 1,
    seed: 3,
    batch_size: 1,
    loras: [],
    sampler: "euler",
    scheduler: "manual",
  });
  assert.ok(recall);
  const applied = applyRecall(recall, [qwen(), t]);
  assert.ok(applied);
  assert.equal(applied.found, true);
  assert.equal(applied.template.id, "qwen-image-2.1-turbo-t2i");
  assert.equal(applied.params.steps, 8);
  assert.equal(applied.params.width, 768);
  const sent = buildGenerateRequest(applied.template, applied.params, {});
  assert.ok(sent.ok);
  for (const key of ["steps", "cfg", "sampler", "scheduler"]) assert.equal(key in sent.body, false, key);
  assert.equal(sent.body.width, 768);
});

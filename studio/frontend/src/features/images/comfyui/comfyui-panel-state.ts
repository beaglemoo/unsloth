// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers behind the Images page ComfyUI panel: template parsing, parameter clamping, the
// generate request, error wording, recall from a gallery record. Free of "@/" imports so the node
// suite can drive it.

type Obj = Record<string, unknown>;

export const MAX_LORAS = 4;
export const MAX_SEED = 2 ** 53 - 1;

export type ComfyLoraRow = { name: string; strength: number };

export type ComfyLimits = {
  steps: [number, number];
  cfg: [number, number];
  side: [number, number];
  multiple: number;
  batchSize: [number, number];
};

export type ComfyDefaults = {
  steps: number;
  cfg: number;
  width: number;
  height: number;
  sampler: string | null;
  scheduler: string | null;
};

export type ComfyTemplate = {
  id: string;
  name: string;
  kind: string;
  /** "shipped" or "user"; only user templates can be deleted. */
  source: string;
  defaults: ComfyDefaults;
  limits: ComfyLimits;
  /** The parameter slots the graph binds (prompt, negative_prompt, seed, ...). */
  slots: string[];
  supportsLora: boolean;
  /** Model files by ComfyUI folder. */
  requiredModels: Record<string, string[]>;
  /** Files ComfyUI lacks by folder; null while ComfyUI cannot be asked, {} when nothing is missing. */
  missingModels: Record<string, string[]> | null;
};

/** What the panel edits. `seed` is the text field: empty means a fresh random seed. */
export type ComfyParams = {
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  seed: string;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  batchSize: number;
  loras: ComfyLoraRow[];
};

export const DEFAULT_LIMITS: ComfyLimits = {
  steps: [1, 100],
  cfg: [0, 20],
  side: [256, 2048],
  multiple: 16,
  batchSize: [1, 4],
};

export const SAMPLER_OPTIONS = [
  "euler",
  "euler_ancestral",
  "heun",
  "dpmpp_2m",
  "dpmpp_2m_sde",
  "dpmpp_sde",
  "dpmpp_3m_sde",
  "uni_pc",
  "lcm",
  "ddim",
];

export const SCHEDULER_OPTIONS = [
  "simple",
  "normal",
  "karras",
  "exponential",
  "sgm_uniform",
  "beta",
  "ddim_uniform",
  "linear_quadratic",
];

/** [label, width, height]; every side is a multiple of 16. */
export const SIZE_PRESETS: ReadonlyArray<readonly [string, number, number]> = [
  ["1:1", 1024, 1024],
  ["3:2", 1216, 832],
  ["2:3", 832, 1216],
  ["16:9", 1344, 768],
  ["9:16", 768, 1344],
];

const isObj = (value: unknown): value is Obj =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const strOr = (value: unknown, fallback: string): string =>
  typeof value === "string" ? value : fallback;

function pair(value: unknown, fallback: [number, number]): [number, number] {
  if (Array.isArray(value) && value.length === 2 && finite(value[0]) && finite(value[1])) {
    return value[0] <= value[1] ? [value[0], value[1]] : fallback;
  }
  return fallback;
}

function fileMap(value: unknown): Record<string, string[]> {
  if (!isObj(value)) return {};
  const out: Record<string, string[]> = {};
  for (const [folder, names] of Object.entries(value)) {
    if (Array.isArray(names)) {
      out[folder] = names.filter((n): n is string => typeof n === "string");
    }
  }
  return out;
}

function templateFromApi(raw: unknown): ComfyTemplate | null {
  if (!isObj(raw) || typeof raw.id !== "string" || raw.id === "") return null;
  const limits = isObj(raw.limits) ? raw.limits : {};
  const defaults = isObj(raw.defaults) ? raw.defaults : {};
  const merged: ComfyLimits = {
    steps: pair(limits.steps, DEFAULT_LIMITS.steps),
    cfg: pair(limits.cfg, DEFAULT_LIMITS.cfg),
    side: pair(limits.side, DEFAULT_LIMITS.side),
    multiple:
      finite(limits.multiple) && limits.multiple >= 1
        ? Math.floor(limits.multiple)
        : DEFAULT_LIMITS.multiple,
    batchSize: pair(limits.batch_size, DEFAULT_LIMITS.batchSize),
  };
  return {
    id: raw.id,
    name: strOr(raw.name, raw.id),
    kind: strOr(raw.kind, "t2i"),
    source: strOr(raw.source, "shipped"),
    defaults: {
      steps: finite(defaults.steps) ? defaults.steps : 25,
      cfg: finite(defaults.cfg) ? defaults.cfg : 1,
      width: finite(defaults.width) ? defaults.width : 1024,
      height: finite(defaults.height) ? defaults.height : 1024,
      sampler: typeof defaults.sampler === "string" ? defaults.sampler : null,
      scheduler: typeof defaults.scheduler === "string" ? defaults.scheduler : null,
    },
    limits: merged,
    slots: Array.isArray(raw.slots)
      ? raw.slots.filter((s): s is string => typeof s === "string")
      : [],
    supportsLora: raw.supports_lora === true,
    requiredModels: fileMap(raw.required_models),
    missingModels: isObj(raw.missing_models) ? fileMap(raw.missing_models) : null,
  };
}

/** GET /comfyui/templates. Entries without an id are dropped. */
export function templatesFromApi(raw: unknown): {
  reachable: boolean;
  templates: ComfyTemplate[];
} {
  const body = isObj(raw) ? raw : {};
  const rows = Array.isArray(body.templates) ? body.templates : [];
  return {
    reachable: body.comfyui_reachable === true,
    templates: rows.flatMap((row) => {
      const template = templateFromApi(row);
      return template ? [template] : [];
    }),
  };
}

export function hasSlot(template: ComfyTemplate, slot: string): boolean {
  return template.slots.includes(slot);
}

/** Every model file the template needs that ComfyUI lacks, as "folder/name". */
export function missingModelList(template: ComfyTemplate): string[] {
  const missing = template.missingModels;
  if (!missing) return [];
  return Object.entries(missing).flatMap(([folder, names]) =>
    names.map((name) => `${folder}/${name}`),
  );
}

export function clampNumber(value: number, range: readonly [number, number]): number {
  if (!Number.isFinite(value)) return range[0];
  return Math.min(range[1], Math.max(range[0], value));
}

/** One image side snapped DOWN to the template's multiple and kept inside its side range. */
export function snapSide(value: number, limits: ComfyLimits): number {
  const [lo, hi] = limits.side;
  const multiple = Math.max(1, limits.multiple);
  const floor = Math.ceil(lo / multiple) * multiple;
  const snapped = Math.floor(clampNumber(Math.round(value), [lo, hi]) / multiple) * multiple;
  return Math.max(snapped, floor);
}

export function defaultParams(template: ComfyTemplate): ComfyParams {
  return {
    prompt: "",
    negativePrompt: "",
    width: snapSide(template.defaults.width, template.limits),
    height: snapSide(template.defaults.height, template.limits),
    seed: "",
    steps: clampNumber(template.defaults.steps, template.limits.steps),
    cfg: clampNumber(template.defaults.cfg, template.limits.cfg),
    sampler: template.defaults.sampler ?? "",
    scheduler: template.defaults.scheduler ?? "",
    batchSize: 1,
    loras: [],
  };
}

/** Parameters made valid for a template: numbers clamped, sides snapped, LoRAs dropped when the
 *  template cannot take them, at most MAX_LORAS rows. Text fields are untouched. */
export function reconcileParams(params: ComfyParams, template: ComfyTemplate): ComfyParams {
  const { limits } = template;
  return {
    ...params,
    width: snapSide(params.width, limits),
    height: snapSide(params.height, limits),
    steps: Math.round(clampNumber(params.steps, limits.steps)),
    cfg: clampNumber(params.cfg, limits.cfg),
    batchSize: Math.round(clampNumber(params.batchSize, limits.batchSize)),
    loras: template.supportsLora ? params.loras.slice(0, MAX_LORAS) : [],
  };
}

/** The seed text as a number: null for empty ("random"), NaN-safe, and undefined when invalid. */
export function parseSeed(text: string): number | null | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) && value <= MAX_SEED ? value : undefined;
}

export function randomSeed(random: () => number = Math.random): string {
  return String(Math.floor(random() * 2 ** 32));
}

export type ComfyGenerateRequest = {
  template_id: string;
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  seed?: number;
  steps?: number;
  cfg?: number;
  sampler?: string;
  scheduler?: string;
  batch_size: number;
  loras: Array<{ name: string; strength: number }>;
};

/** The POST /comfyui/generate body, or an error text. Only parameters the template binds are sent. */
export function buildGenerateRequest(
  template: ComfyTemplate,
  raw: ComfyParams,
): { ok: true; body: ComfyGenerateRequest } | { ok: false; error: string } {
  const params = reconcileParams(raw, template);
  const prompt = params.prompt.trim();
  if (!prompt) return { ok: false, error: "Prompt is empty" };
  const seed = parseSeed(params.seed);
  if (seed === undefined) {
    return { ok: false, error: "Seed must be a whole number, or empty for a random one" };
  }
  const body: ComfyGenerateRequest = {
    template_id: template.id,
    prompt,
    batch_size: params.batchSize,
    loras: params.loras
      .map((row) => ({ name: row.name.trim(), strength: row.strength }))
      .filter((row) => row.name !== "" && Number.isFinite(row.strength)),
  };
  if (hasSlot(template, "negative_prompt") && params.negativePrompt.trim()) {
    body.negative_prompt = params.negativePrompt.trim();
  }
  if (hasSlot(template, "width")) body.width = params.width;
  if (hasSlot(template, "height")) body.height = params.height;
  if (hasSlot(template, "seed") && seed !== null) body.seed = seed;
  if (hasSlot(template, "steps")) body.steps = params.steps;
  if (hasSlot(template, "cfg")) body.cfg = params.cfg;
  if (hasSlot(template, "sampler") && params.sampler) body.sampler = params.sampler;
  if (hasSlot(template, "scheduler") && params.scheduler) body.scheduler = params.scheduler;
  return { ok: true, body };
}

// ---------------------------------------------------------------- recall

/** The slice of a gallery record a ComfyUI recall reads (api.ts GalleryImage satisfies it). */
export type ComfyRecallSource = {
  engine?: string | null;
  comfyui_template?: string | null;
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
  batch_seed?: number | null;
  batch_size: number;
  loras?: string[];
  sampler?: string | null;
  scheduler?: string | null;
};

export type ComfyRecall = {
  templateId: string;
  params: Partial<ComfyParams>;
};

/** "name:strength" recipe entries; splits on the LAST colon so a file name with ':' survives. */
export function loraRowsFromRecipe(entries: readonly string[] | undefined): ComfyLoraRow[] {
  const rows: ComfyLoraRow[] = [];
  for (const entry of entries ?? []) {
    const idx = entry.lastIndexOf(":");
    if (idx <= 0) continue;
    const name = entry.slice(0, idx);
    const strength = Number(entry.slice(idx + 1));
    if (name && Number.isFinite(strength)) rows.push({ name, strength });
  }
  return rows;
}

export function isComfyRecord(image: { engine?: string | null }): boolean {
  return image.engine === "comfyui";
}

/** The panel state that regenerates a ComfyUI gallery record. Null for a Studio record. */
export function recallFromImage(image: ComfyRecallSource): ComfyRecall | null {
  if (!isComfyRecord(image)) return null;
  const templateId = image.comfyui_template ?? "";
  if (!templateId) return null;
  const params: Partial<ComfyParams> = {
    prompt: image.prompt,
    negativePrompt: image.negative_prompt ?? "",
    width: image.width,
    height: image.height,
    steps: image.steps,
    cfg: image.guidance,
    // The batch's base seed, not this image's derived one, or a replay of a batch advances it.
    seed: String(image.batch_seed ?? image.seed),
    batchSize: image.batch_size ?? 1,
    loras: loraRowsFromRecipe(image.loras),
  };
  if (image.sampler) params.sampler = image.sampler;
  if (image.scheduler) params.scheduler = image.scheduler;
  return { templateId, params };
}

/** A recall applied to the templates now on offer: the recalled template when it still exists
 *  (`found`), else the fallback with the recalled values clamped to it. */
export function applyRecall(
  recall: ComfyRecall,
  templates: readonly ComfyTemplate[],
): { template: ComfyTemplate; params: ComfyParams; found: boolean } | null {
  const exact = templates.find((t) => t.id === recall.templateId);
  const template = exact ?? templates[0];
  if (!template) return null;
  const base = defaultParams(template);
  const merged = { ...base, ...recall.params } as ComfyParams;
  return { template, params: reconcileParams(merged, template), found: exact !== undefined };
}

// ---------------------------------------------------------------- availability

/** The engine switch is offered to the owner while the attached flag is on and ComfyUI answers or
 *  is meant to run. `status` is the poller's `comfyui` block. */
export function comfyuiAvailable(input: {
  isOwner: boolean;
  enabled: boolean;
  status: { reachable: boolean; helperWanted: boolean | null } | null | undefined;
}): boolean {
  if (!input.isOwner || !input.enabled || !input.status) return false;
  return input.status.reachable || input.status.helperWanted === true;
}

// ---------------------------------------------------------------- errors

export const DEFAULT_RETRY_AFTER_S = 15;

export type ComfyFailureKind =
  | "off"
  | "busy"
  | "cancelled"
  | "missing"
  | "not_running"
  | "error";

export type ComfyFailure = {
  kind: ComfyFailureKind;
  message: string;
  /** Seconds to wait before trying again, for busy and admission replies. */
  retryAfterS?: number;
  /** Offer a button that opens Settings > Engines. */
  openSettings?: boolean;
  /** The template id is gone; the list should be reloaded. */
  reloadTemplates?: boolean;
  /** Studio media is likely resident: offer the existing Unload action. */
  offerUnload?: boolean;
};

const sentence = (text: string): string => text.trim().replace(/[.\s]+$/, "");

/** How a failed ComfyUI request reads, from the HTTP status, the X-Comfy-Error code and the detail. */
export function describeComfyFailure(input: {
  status: number;
  code: string | null;
  detail: string;
  retryAfterS?: number | null;
}): ComfyFailure {
  const { status, code } = input;
  const detail = input.detail.trim();
  const retry =
    input.retryAfterS && input.retryAfterS > 0 ? input.retryAfterS : DEFAULT_RETRY_AFTER_S;
  switch (code) {
    case "off":
      return {
        kind: "off",
        message: "ComfyUI is off. Turn it on in Settings > Engines to generate with it.",
        openSettings: true,
      };
    case "cancelled":
      return { kind: "cancelled", message: "Generation cancelled. Nothing was saved." };
    case "busy":
      return {
        kind: "busy",
        message: `${sentence(detail || "ComfyUI is already generating an image")}. Wait for it to finish, then try again.`,
        retryAfterS: retry,
      };
    case "admission":
      return {
        kind: "busy",
        message: `${sentence(detail || "ComfyUI is waiting for memory")}. Try again in about ${retry} s.`,
        retryAfterS: retry,
        offerUnload: true,
      };
    case "missing_models":
      return {
        kind: "missing",
        message: detail || "ComfyUI is missing model files this template needs.",
      };
    case "not_running":
      return {
        kind: "not_running",
        message:
          "ComfyUI is not answering yet. If you just turned it on, give it a few seconds; otherwise check Settings > Engines.",
        openSettings: true,
      };
    case "unreachable":
      return {
        kind: "not_running",
        message: "Studio could not reach ComfyUI. Check that it is running in Settings > Engines.",
        openSettings: true,
      };
    case "template":
      return {
        kind: "error",
        message: detail || "That template is no longer available.",
        reloadTemplates: true,
      };
    case "timeout":
      return {
        kind: "error",
        message: `${sentence(detail || "ComfyUI did not finish in time")}. Its queue is in Settings > Engines.`,
        openSettings: true,
      };
    case "execution":
    case "params":
    case "graph":
      return { kind: "error", message: detail || "ComfyUI could not run this generation." };
    default:
      break;
  }
  if (status === 503) {
    return {
      kind: "busy",
      message: `${sentence(detail || "Busy right now")}. Try again in about ${retry} s.`,
      retryAfterS: retry,
      offerUnload: true,
    };
  }
  if (status === 409) return { kind: "busy", message: detail || "ComfyUI is busy.", retryAfterS: retry };
  return { kind: "error", message: detail || "ComfyUI generation failed." };
}

// ---------------------------------------------------------------- progress

/** The GET /comfyui/progress body in the shape the Images progress UI reads, plus the queue slot. */
export type ComfyProgress = {
  active: boolean;
  step: number;
  total_steps: number;
  fraction: number;
  eta_seconds: number | null;
  phase: "encode" | "denoise" | "decode" | null;
  preview: string | null;
  preview_seq: number;
  queue_position: number | null;
  prompt_id: string | null;
};

const PHASES = ["encode", "denoise", "decode"] as const;

export function progressFromApi(raw: unknown): ComfyProgress {
  const body = isObj(raw) ? raw : {};
  const phase = PHASES.find((p) => p === body.phase) ?? null;
  const step = finite(body.step) ? body.step : 0;
  const total = finite(body.total_steps) ? body.total_steps : 0;
  return {
    active: body.active === true,
    step,
    total_steps: total,
    fraction: finite(body.fraction) ? Math.min(1, Math.max(0, body.fraction)) : 0,
    eta_seconds: finite(body.eta_seconds) ? body.eta_seconds : null,
    phase,
    preview: typeof body.preview === "string" ? body.preview : null,
    preview_seq: finite(body.preview_seq) ? body.preview_seq : 0,
    queue_position: finite(body.queue_position) ? body.queue_position : null,
    prompt_id: typeof body.prompt_id === "string" ? body.prompt_id : null,
  };
}

/** "Next in ComfyUI's queue" for a job that has not started; null once it runs. */
export function queueLabel(position: number | null | undefined): string | null {
  if (position === null || position === undefined || position <= 0) return null;
  return position === 1 ? "Next in ComfyUI's queue" : `Position ${position} in ComfyUI's queue`;
}

// ---------------------------------------------------------------- import

/** The graph object from pasted or uploaded JSON, or an error text. The backend validates the rest. */
export function parseGraphJson(
  text: string,
): { ok: true; graph: Record<string, unknown> } | { ok: false; error: string } {
  const trimmed = text.trim();
  if (!trimmed) return { ok: false, error: "Paste or choose the graph JSON" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { ok: false, error: "That is not valid JSON" };
  }
  if (!isObj(parsed) || Object.keys(parsed).length === 0) {
    return { ok: false, error: "The graph must be a JSON object in ComfyUI's API format" };
  }
  if (Array.isArray((parsed as Obj).nodes) && Array.isArray((parsed as Obj).links)) {
    return {
      ok: false,
      error: 'That is the editor format. Export with "Save (API Format)" instead.',
    };
  }
  return { ok: true, graph: parsed };
}

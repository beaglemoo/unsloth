// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  AttachedDs4Status,
  AttachedEnginesSettings,
  AttachedNotice,
  AttachedOmlxModel,
  AttachedOmlxStatus,
  AttachedProvider,
  AttachedSyncResult,
  AttachedStatus,
  Ds4ApplyResult,
  Ds4ContextInfo,
  OmlxContextInfo,
} from "./types";

const STATUS_URL = "/api/engines/attached/status";
const SETTINGS_URL = "/api/settings/attached-engines";

// The wire shapes are the backend dataclasses run through asdict(): snake_case, nullable.
type Obj = Record<string, unknown>;

const num = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;
const numOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const strOrNull = (value: unknown): string | null =>
  typeof value === "string" ? value : null;
const arr = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

function omlxModelFromApi(raw: Obj): AttachedOmlxModel {
  return {
    id: String(raw.id ?? ""),
    modelPath: typeof raw.model_path === "string" ? raw.model_path : "",
    loaded: raw.loaded === true,
    isLoading: raw.is_loading === true,
    estimatedSize: num(raw.estimated_size),
    pinned: raw.pinned === true,
    engineType: strOrNull(raw.engine_type),
    isHelper: raw.is_helper === true,
    modelAlias: strOrNull(raw.model_alias),
    maxContextWindow: numOrNull(raw.max_context_window),
    modelContextLength: numOrNull(raw.model_context_length),
    maxTokens: numOrNull(raw.max_tokens),
    ttlS: numOrNull(raw.ttl_s),
    idleRemainingS: numOrNull(raw.idle_remaining_s),
  };
}

function omlxFromApi(raw: Obj): AttachedOmlxStatus {
  return {
    reachable: raw.reachable === true,
    models: arr(raw.models).map((row) => omlxModelFromApi(row as Obj)),
    memoryBytes: num(raw.memory_bytes),
    ceilingBytes: num(raw.ceiling_bytes),
    error: strOrNull(raw.error),
    chatModelIds: arr(raw.chat_model_ids).filter(
      (id): id is string => typeof id === "string",
    ),
  };
}

function ds4FromApi(raw: Obj): AttachedDs4Status {
  return {
    reachable: raw.reachable === true,
    loaded: raw.loaded === true,
    starting: raw.starting === true,
    pid: numOrNull(raw.pid),
    uptimeS: numOrNull(raw.uptime_s),
    inFlight: num(raw.in_flight),
    idleRemainingS: numOrNull(raw.idle_remaining_s),
    liveTps: numOrNull(raw.live_tps),
    lastGenTps: numOrNull(raw.last_gen_tps),
    lastTtftMs: numOrNull(raw.last_ttft_ms),
    startTimeoutS: num(raw.start_timeout_s, 120),
    ctx: numOrNull(raw.ctx),
    ctxActive: numOrNull(raw.ctx_active),
    pendingRestart: raw.pending_restart === true,
    error: strOrNull(raw.error),
  };
}

function noticeFromApi(raw: Obj): AttachedNotice {
  return {
    ts: num(raw.ts),
    reason: typeof raw.reason === "string" ? raw.reason : "",
    actions: arr(raw.actions).filter(
      (action): action is string => typeof action === "string",
    ),
    inFlightKilled: num(raw.in_flight_killed),
  };
}

/** Live engine status, or null when the backend answers 404 (the flag is off). */
export async function fetchAttachedStatus(
  since = 0,
): Promise<AttachedStatus | null> {
  const res = await authFetch(`${STATUS_URL}?since=${since}`);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to read engine status"));
  }
  const body = (await res.json()) as Obj;
  return {
    omlx: body.omlx ? omlxFromApi(body.omlx as Obj) : null,
    ds4: body.ds4 ? ds4FromApi(body.ds4 as Obj) : null,
    modelsHash: typeof body.models_hash === "string" ? body.models_hash : "",
    notices: arr(body.notices).map((row) => noticeFromApi(row as Obj)),
    receivedAt: Date.now(),
  };
}

async function post(path: string, body?: unknown, fallback = "Request failed") {
  const res = await authFetch(`/api/engines/attached${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readFastApiError(res, fallback));
  return res;
}

/** Upsert (or, with the flag off, delete) the saved provider rows the picker is built from. */
export async function syncAttachedProviders(): Promise<AttachedSyncResult> {
  const res = await post("/sync", undefined, "Failed to sync engine models");
  const body = (await res.json().catch(() => ({}))) as Obj;
  const rows = (body.rows ?? {}) as Obj;
  return {
    enabled: body.enabled === true,
    incomplete: Object.values(rows).includes("kept_models"),
  };
}

/** Fire-and-forget on selection: frees memory (oMLX) or pre-warms (DwarfStar). */
export async function prepareAttachedProvider(
  provider: AttachedProvider,
): Promise<void> {
  await post("/prepare", { provider }, "Failed to prepare engine");
}

export async function loadOmlxModel(modelId: string): Promise<void> {
  await post("/omlx/load", { model_id: modelId }, "Failed to load model");
}

export async function unloadOmlxModel(modelId: string): Promise<void> {
  await post("/omlx/unload", { model_id: modelId }, "Failed to unload model");
}

export async function unloadAllOmlxModels(): Promise<void> {
  await post("/omlx/unload-all", undefined, "Failed to unload models");
}

function omlxContextFromApi(raw: Obj): OmlxContextInfo {
  return {
    modelId: String(raw.model_id ?? ""),
    dir: String(raw.dir ?? ""),
    maxContextWindow: numOrNull(raw.max_context_window),
    nativeMax: numOrNull(raw.native_max),
    effective: numOrNull(raw.effective),
  };
}

const APPLIED = new Set<string>([
  "next_start",
  "restarted",
  "after_current_requests",
  "unchanged",
]);

function ds4ContextFromApi(raw: Obj): Ds4ContextInfo {
  return {
    ctx: num(raw.ctx),
    ctxActive: numOrNull(raw.ctx_active),
    ctxMin: num(raw.ctx_min, 4096),
    ctxMax: num(raw.ctx_max),
    pendingRestart: raw.pending_restart === true,
    applied:
      typeof raw.applied === "string" && APPLIED.has(raw.applied)
        ? (raw.applied as Ds4ApplyResult)
        : null,
  };
}

export async function getOmlxContext(modelId: string): Promise<OmlxContextInfo> {
  const res = await post(
    "/omlx/context/get",
    { model_id: modelId },
    "Failed to read the context length",
  );
  return omlxContextFromApi((await res.json()) as Obj);
}

/** `null` resets the model to its default cap. */
export async function setOmlxContext(
  modelId: string,
  maxContextWindow: number | null,
): Promise<OmlxContextInfo> {
  const res = await post(
    "/omlx/context",
    { model_id: modelId, max_context_window: maxContextWindow },
    "Failed to set the context length",
  );
  return omlxContextFromApi((await res.json()) as Obj);
}

export async function getDs4Context(): Promise<Ds4ContextInfo> {
  const res = await authFetch("/api/engines/attached/ds4/context");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to read the DwarfStar context length"),
    );
  }
  return ds4ContextFromApi((await res.json()) as Obj);
}

export async function setDs4Context(ctx: number): Promise<Ds4ContextInfo> {
  const res = await post(
    "/ds4/context",
    { ctx },
    "Failed to set the DwarfStar context length",
  );
  return ds4ContextFromApi((await res.json()) as Obj);
}

export async function startDs4(): Promise<void> {
  await post("/ds4/start", undefined, "Failed to start DwarfStar");
}

export async function stopDs4(): Promise<void> {
  await post("/ds4/stop", undefined, "Failed to stop DwarfStar");
}

function settingsFromApi(raw: Obj): AttachedEnginesSettings {
  return {
    enabled: raw.enabled === true,
    omlxUrl: typeof raw.omlx_url === "string" ? raw.omlx_url : "",
    ds4Url: typeof raw.ds4_url === "string" ? raw.ds4_url : "",
    scanDenylist: arr(raw.scan_denylist).filter(
      (entry): entry is string => typeof entry === "string",
    ),
    arbitrateLocalLoads: raw.arbitrate_local_loads !== false,
    prewarmDs4OnSelect: raw.prewarm_ds4_on_select !== false,
  };
}

export async function loadAttachedEnginesSettings(): Promise<AttachedEnginesSettings> {
  const res = await authFetch(SETTINGS_URL);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load attached engine settings"),
    );
  }
  return settingsFromApi((await res.json()) as Obj);
}

// Camel-cased field -> the API schema key it is sent as.
const UPDATE_KEYS = {
  enabled: "enabled",
  omlxUrl: "omlx_url",
  ds4Url: "ds4_url",
  scanDenylist: "scan_denylist",
  arbitrateLocalLoads: "arbitrate_local_loads",
  prewarmDs4OnSelect: "prewarm_ds4_on_select",
} as const;

/** A partial write: an omitted field keeps its stored value. */
export async function updateAttachedEnginesSettings(
  update: Partial<AttachedEnginesSettings>,
): Promise<AttachedEnginesSettings> {
  const body: Record<string, unknown> = {};
  for (const [field, key] of Object.entries(UPDATE_KEYS)) {
    const value = update[field as keyof typeof UPDATE_KEYS];
    if (value !== undefined) body[key] = value;
  }
  const res = await authFetch(SETTINGS_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update attached engine settings"),
    );
  }
  return settingsFromApi((await res.json()) as Obj);
}

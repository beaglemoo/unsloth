// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Wire parsing for the ComfyUI routes (snake_case, nullable). Pure and free of "@/" imports so the
// node suite can drive it.

import { failureFromApi } from "./failure.ts";
import type {
  ComfyuiDevice,
  ComfyuiModels,
  ComfyuiPeer,
  ComfyuiQueueRow,
  ComfyuiQueueState,
  ComfyuiStatus,
} from "./types.ts";

type Obj = Record<string, unknown>;

const num = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;
const numOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const strOrNull = (value: unknown): string | null =>
  typeof value === "string" ? value : null;
const arr = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const QUEUE_STATES: readonly ComfyuiQueueState[] = ["down", "unknown", "busy", "idle"];

function queueStateFromApi(value: unknown): ComfyuiQueueState {
  return QUEUE_STATES.includes(value as ComfyuiQueueState)
    ? (value as ComfyuiQueueState)
    : "unknown";
}

function peerFromApi(raw: Obj): ComfyuiPeer {
  return {
    url: typeof raw.url === "string" ? raw.url : "",
    reachable: raw.reachable === true,
    busy: raw.busy === true,
    state: queueStateFromApi(raw.state),
  };
}

function deviceFromApi(raw: Obj): ComfyuiDevice {
  return {
    name: typeof raw.name === "string" ? raw.name : "",
    type: typeof raw.type === "string" ? raw.type : "",
  };
}

/** The `comfyui` block of /status, or null when the backend predates the engine (or sent junk). */
export function comfyuiFromApi(raw: unknown): ComfyuiStatus | null {
  if (!raw || typeof raw !== "object") return null;
  const body = raw as Obj;
  return {
    url: typeof body.url === "string" ? body.url : "",
    reachable: body.reachable === true,
    state: queueStateFromApi(body.state),
    version: strOrNull(body.version),
    queueRunning: num(body.queue_running),
    queuePending: num(body.queue_pending),
    devices: arr(body.devices).map((row) => deviceFromApi((row ?? {}) as Obj)),
    ramTotal: numOrNull(body.ram_total),
    ramFree: numOrNull(body.ram_free),
    failure: failureFromApi(body.failure),
    helperWanted: typeof body.helper_wanted === "boolean" ? body.helper_wanted : null,
    peers: arr(body.peers).map((row) => peerFromApi((row ?? {}) as Obj)),
  };
}

/** GET /comfyui/queue: running rows first, then pending. Rows without an id are dropped. */
export function comfyuiQueueFromApi(raw: unknown): ComfyuiQueueRow[] {
  if (!raw || typeof raw !== "object") return [];
  const body = raw as Obj;
  const rows = (value: unknown, state: ComfyuiQueueRow["state"]): ComfyuiQueueRow[] =>
    arr(value).flatMap((row) => {
      const item = (row ?? {}) as Obj;
      if (typeof item.prompt_id !== "string" || item.prompt_id === "") return [];
      return [
        {
          promptId: item.prompt_id,
          number: numOrNull(item.number),
          state,
          studio: item.studio !== null && item.studio !== undefined,
        },
      ];
    });
  return [...rows(body.running, "running"), ...rows(body.pending, "pending")];
}

/** GET /comfyui/models: `{folders: {name: [files]}}`. Non-string entries are dropped. */
export function comfyuiModelsFromApi(raw: unknown): ComfyuiModels {
  const folders = raw && typeof raw === "object" ? (raw as Obj).folders : null;
  if (!folders || typeof folders !== "object") return {};
  const out: ComfyuiModels = {};
  for (const [name, files] of Object.entries(folders as Obj)) {
    out[name] = arr(files).filter((file): file is string => typeof file === "string");
  }
  return out;
}

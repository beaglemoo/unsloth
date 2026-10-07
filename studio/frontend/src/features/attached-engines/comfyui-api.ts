// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { engineRequest } from "./api";
import { comfyuiModelsFromApi, comfyuiQueueFromApi } from "./comfyui-wire";
import type { ComfyuiModels, ComfyuiQueueRow } from "./types";

/** Unload ComfyUI's models and free its memory. `deferred`: a job is running, so ComfyUI frees after it. */
export async function freeComfyui(): Promise<{ freed: boolean; deferred: boolean }> {
  const res = await engineRequest(
    "/comfyui/free",
    { method: "POST" },
    "Could not unload the ComfyUI models",
  );
  const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  return { freed: body.freed !== false, deferred: body.deferred === true };
}

export async function fetchComfyuiQueue(): Promise<ComfyuiQueueRow[]> {
  const res = await engineRequest(
    "/comfyui/queue",
    { method: "GET" },
    "Could not read the ComfyUI queue",
  );
  return comfyuiQueueFromApi(await res.json());
}

/** Interrupt a running job or drop a pending one on Studio's own ComfyUI. */
export async function cancelComfyuiJob(promptId: string): Promise<void> {
  await engineRequest(
    "/comfyui/queue/cancel",
    { method: "POST", body: { prompt_id: promptId } },
    "Could not cancel the job",
  );
}

export async function fetchComfyuiModels(): Promise<ComfyuiModels> {
  const res = await engineRequest(
    "/comfyui/models",
    { method: "GET" },
    "Could not list the ComfyUI models",
  );
  return comfyuiModelsFromApi(await res.json());
}

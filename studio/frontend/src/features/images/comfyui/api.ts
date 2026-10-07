// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { fetchComfyuiModels } from "@/features/attached-engines/comfyui-api";
import { parseRetryAfter } from "@/features/attached-engines/engine-errors";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { GalleryImage } from "../api";
import {
  type ComfyFailure,
  type ComfyGenerateRequest,
  type ComfyProgress,
  type ComfyTemplate,
  describeComfyFailure,
  progressFromApi,
  templatesFromApi,
} from "./comfyui-panel-state";

const BASE = "/api/engines/attached/comfyui";

/** A non-2xx ComfyUI reply with the machine-readable `X-Comfy-Error` code the routes attach. */
export class ComfyRequestError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly retryAfterS: number | null;

  constructor(detail: string, status: number, code: string | null, retryAfterS: number | null) {
    super(detail);
    this.name = "ComfyRequestError";
    this.status = status;
    this.code = code;
    this.retryAfterS = retryAfterS;
  }

  get failure(): ComfyFailure {
    return describeComfyFailure({
      status: this.status,
      code: this.code,
      detail: this.message,
      retryAfterS: this.retryAfterS,
    });
  }
}

/** The calm wording for any error a ComfyUI call threw (a network failure included). */
export function comfyFailureOf(error: unknown, fallback: string): ComfyFailure {
  if (error instanceof ComfyRequestError) return error.failure;
  return {
    kind: "error",
    message: error instanceof Error && error.message ? error.message : fallback,
  };
}

async function request(
  path: string,
  init: { method: "GET" | "POST" | "DELETE"; body?: unknown; signal?: AbortSignal },
  fallback: string,
): Promise<Response> {
  const res = await authFetch(`${BASE}${path}`, {
    method: init.method,
    headers: init.body === undefined ? undefined : { "Content-Type": "application/json" },
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    signal: init.signal,
  });
  if (!res.ok) {
    throw new ComfyRequestError(
      await readFastApiError(res, fallback),
      res.status,
      res.headers.get("X-Comfy-Error"),
      parseRetryAfter(res.headers.get("Retry-After")),
    );
  }
  return res;
}

export async function fetchComfyTemplates(): Promise<{
  reachable: boolean;
  templates: ComfyTemplate[];
}> {
  const res = await request("/templates", { method: "GET" }, "Could not list the ComfyUI templates");
  return templatesFromApi(await res.json());
}

/** The LoRA files ComfyUI lists in its `loras` folder; empty when it cannot be asked. */
export async function fetchComfyLoras(): Promise<string[]> {
  try {
    const models = await fetchComfyuiModels();
    return models.loras ?? [];
  } catch {
    return [];
  }
}

export type ComfyGenerateResult = {
  images: GalleryImage[];
  actions: string[];
  promptId: string;
  seed: number;
};

/** Blocks until the job ends. Poll {@link fetchComfyProgress} meanwhile. */
export async function generateWithComfy(body: ComfyGenerateRequest): Promise<ComfyGenerateResult> {
  const res = await request("/generate", { method: "POST", body }, "ComfyUI generation failed");
  const data = (await res.json()) as Record<string, unknown>;
  return {
    images: Array.isArray(data.images) ? (data.images as GalleryImage[]) : [],
    actions: Array.isArray(data.actions)
      ? data.actions.filter((a): a is string => typeof a === "string")
      : [],
    promptId: typeof data.prompt_id === "string" ? data.prompt_id : "",
    seed: typeof data.seed === "number" ? data.seed : 0,
  };
}

export async function fetchComfyProgress(signal?: AbortSignal): Promise<ComfyProgress> {
  const res = await request("/progress", { method: "GET", signal }, "Could not read the progress");
  return progressFromApi(await res.json());
}

export async function cancelComfyGeneration(): Promise<boolean> {
  const res = await request("/generate/cancel", { method: "POST" }, "Could not cancel the job");
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  return data.cancelled === true;
}

export async function importComfyGraph(name: string, graph: unknown): Promise<ComfyTemplate | null> {
  const res = await request(
    "/templates/import",
    { method: "POST", body: { name, graph } },
    "Could not import the graph",
  );
  const data = (await res.json()) as Record<string, unknown>;
  return templatesFromApi({ templates: [data.template] }).templates[0] ?? null;
}

export async function deleteComfyTemplate(id: string): Promise<void> {
  await request(`/templates/${encodeURIComponent(id)}`, { method: "DELETE" }, "Could not delete the template");
}

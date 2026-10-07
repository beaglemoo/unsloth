// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which engine the Images page generates with, remembered across visits. Pure and free of "@/"
// imports so the node suite can drive it.

export type ImageEngine = "studio" | "comfyui";

export const IMAGE_ENGINE_KEY = "unsloth.images.engine";

/** The remembered choice; anything unreadable or unknown is "studio". */
export function readStoredEngine(storage?: Pick<Storage, "getItem"> | null): ImageEngine {
  try {
    const store = storage === undefined ? globalThis.localStorage : storage;
    return store?.getItem(IMAGE_ENGINE_KEY) === "comfyui" ? "comfyui" : "studio";
  } catch {
    return "studio";
  }
}

export function persistEngine(
  engine: ImageEngine,
  storage?: Pick<Storage, "setItem"> | null,
): void {
  try {
    const store = storage === undefined ? globalThis.localStorage : storage;
    store?.setItem(IMAGE_ENGINE_KEY, engine);
  } catch {
    // storage can be unavailable in private browsing
  }
}

/**
 * What the page actually uses. A remembered "comfyui" with ComfyUI gone (flag off, helper off, not
 * the owner) falls back to Studio instead of leaving a blank page; the choice itself is kept, so it
 * comes back with the engine. ComfyUI only has the Create workflow.
 */
export function effectiveEngine(
  chosen: ImageEngine,
  available: boolean,
  pageMode: "create" | "train",
): ImageEngine {
  return chosen === "comfyui" && available && pageMode === "create" ? "comfyui" : "studio";
}

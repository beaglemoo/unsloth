// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the chat needs to know about the attached model that is selected: its effective context
// window and when the engine will unload it. Pure, so the node suite can drive it.

import type {
  AttachedDs4Status,
  AttachedOmlxModel,
  AttachedOmlxStatus,
  AttachedProvider,
  AttachedStatus,
} from "./types.ts";

export type ActiveAttached = { kind: AttachedProvider; modelId: string };

/** Rows for the selected oMLX id: itself plus every alias or profile sharing its weights. */
export function omlxRowsFor(
  status: AttachedOmlxStatus,
  modelId: string,
): AttachedOmlxModel[] {
  const row =
    status.models.find((model) => model.id === modelId) ??
    status.models.find((model) => model.modelAlias === modelId);
  if (!row) return [];
  if (!row.modelPath) return [row];
  return status.models.filter((model) => model.modelPath === row.modelPath);
}

/** The prompt cap in force for the selected attached model, or null when it is not known. */
export function activeContextWindow(
  status: AttachedStatus | null,
  active: ActiveAttached | null,
): number | null {
  if (!status || !active) return null;
  if (active.kind === "dwarfstar") {
    const ds4 = status.ds4;
    if (!ds4?.reachable) return null;
    return ds4.ctxActive ?? ds4.ctx ?? null;
  }
  if (!status.omlx?.reachable) return null;
  const rows = omlxRowsFor(status.omlx, active.modelId);
  const withCap = rows.find((row) => row.maxContextWindow !== null);
  return withCap?.maxContextWindow ?? rows[0]?.modelContextLength ?? null;
}

/** The most output tokens the selected attached model accepts: oMLX's own `max_tokens` for the
 *  model, DwarfStar's running context. Null when the engine has not said. */
export function attachedMaxOutputTokens(
  status: AttachedStatus | null,
  active: ActiveAttached | null,
): number | null {
  if (!status || !active) return null;
  if (active.kind === "dwarfstar") {
    return status.ds4?.reachable ? (status.ds4.ctxActive ?? status.ds4.ctx ?? null) : null;
  }
  if (!status.omlx?.reachable) return null;
  return omlxRowsFor(status.omlx, active.modelId).find((row) => row.maxTokens !== null)?.maxTokens ?? null;
}

/** Max Tokens as it goes on the wire: bounded by the selected attached model's output ceiling
 *  when the engine has reported one. Applied at send time only; the saved value is never lowered,
 *  so a small DwarfStar context does not shrink the setting for every other model. */
export function capMaxTokensForAttached(
  maxTokens: number,
  status: AttachedStatus | null,
  active: ActiveAttached | null,
): number {
  const cap = attachedMaxOutputTokens(status, active);
  return cap !== null && cap > 0 && maxTokens > cap ? cap : maxTokens;
}

export type UnloadState =
  | { kind: "offline" }
  | { kind: "not-loaded" }
  | { kind: "loading" }
  | { kind: "pinned" }
  /** DwarfStar mid-request: the idle timer is paused. */
  | { kind: "busy" }
  /** Loaded, but the engine reported no idle timer. */
  | { kind: "untimed" }
  | { kind: "countdown"; remainingS: number };

function ds4Unload(ds4: AttachedDs4Status | null): UnloadState {
  if (!ds4?.reachable) return { kind: "offline" };
  if (ds4.starting) return { kind: "loading" };
  if (!ds4.loaded) return { kind: "not-loaded" };
  if (ds4.inFlight > 0) return { kind: "busy" };
  if (ds4.idleRemainingS === null) return { kind: "untimed" };
  return { kind: "countdown", remainingS: ds4.idleRemainingS };
}

function omlxUnload(
  omlx: AttachedOmlxStatus | null,
  modelId: string,
): UnloadState {
  if (!omlx?.reachable) return { kind: "offline" };
  const rows = omlxRowsFor(omlx, modelId);
  if (rows.some((row) => row.isLoading && !row.loaded)) return { kind: "loading" };
  const loaded = rows.filter((row) => row.loaded);
  if (loaded.length === 0) return { kind: "not-loaded" };
  if (loaded.some((row) => row.pinned)) return { kind: "pinned" };
  const remaining = loaded
    .map((row) => row.idleRemainingS)
    .filter((value): value is number => value !== null);
  if (remaining.length === 0) return { kind: "untimed" };
  return { kind: "countdown", remainingS: Math.min(...remaining) };
}

/** The unload state as of the last poll; `remainingS` still needs `remainingNow` to tick. */
export function unloadStateFor(
  status: AttachedStatus | null,
  active: ActiveAttached | null,
): UnloadState | null {
  if (!status || !active) return null;
  return active.kind === "dwarfstar"
    ? ds4Unload(status.ds4)
    : omlxUnload(status.omlx, active.modelId);
}

/** Seconds left now, counted down from the poll that reported `baseS`. Never negative. */
export function remainingNow(
  baseS: number,
  receivedAt: number | undefined,
  now: number,
): number {
  const elapsed = receivedAt === undefined ? 0 : Math.max(0, (now - receivedAt) / 1000);
  return Math.max(0, baseS - elapsed);
}

/** m:ss, rounded up so a countdown never shows 0:00 while time remains. */
export function formatCountdown(seconds: number): string {
  const whole = Math.max(0, Math.ceil(seconds));
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

/** The one-line text for the chat indicator and the engines panel; null when there is nothing to say. */
export function describeUnload(
  state: UnloadState | null,
  receivedAt: number | undefined,
  now: number,
): string | null {
  if (!state) return null;
  switch (state.kind) {
    case "offline":
      return null;
    case "loading":
      return "Loading";
    case "not-loaded":
      return "Not loaded";
    case "pinned":
      return "Pinned, stays loaded";
    case "busy":
      return "In use";
    case "untimed":
      return "Stays loaded";
    case "countdown": {
      const left = remainingNow(state.remainingS, receivedAt, now);
      return left <= 0 ? "Unloading" : `Unloads in ${formatCountdown(left)}`;
    }
  }
}

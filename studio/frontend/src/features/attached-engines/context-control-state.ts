// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The numbers and wording behind the chat settings "Context length" control for oMLX and DwarfStar.

import type { Ds4ApplyResult, Ds4ContextInfo, OmlxContextInfo } from "./types.ts";

export const CONTEXT_MIN = 4096;
export const CONTEXT_STEP = 1024;

export type ContextRange = {
  min: number;
  max: number;
  /** The value the slider starts on: the saved setting, else the one in force, else the top. */
  current: number;
  /** oMLX only: whether a per-model value is saved (null in the engine means "default"). */
  hasOverride: boolean;
};

export function omlxContextRange(info: OmlxContextInfo): ContextRange | null {
  if (info.nativeMax === null || info.nativeMax < CONTEXT_MIN) return null;
  const current = info.maxContextWindow ?? info.effective ?? info.nativeMax;
  return {
    min: CONTEXT_MIN,
    max: info.nativeMax,
    current: Math.min(info.nativeMax, Math.max(CONTEXT_MIN, current)),
    hasOverride: info.maxContextWindow !== null,
  };
}

export function ds4ContextRange(info: Ds4ContextInfo): ContextRange | null {
  if (info.ctxMax < CONTEXT_MIN) return null;
  const min = Math.max(CONTEXT_MIN, info.ctxMin);
  return {
    min,
    max: info.ctxMax,
    current: Math.min(info.ctxMax, Math.max(min, info.ctx)),
    hasOverride: false,
  };
}

/** Round to the slider step and clamp to the range, for the numeric input. */
export function clampContext(value: number, range: Pick<ContextRange, "min" | "max">): number {
  if (!Number.isFinite(value)) return range.min;
  return Math.min(range.max, Math.max(range.min, Math.round(value)));
}

const fmt = (n: number): string => n.toLocaleString();

/** The toast after a DwarfStar change, by what the launcher did with it. */
export function ds4AppliedMessage(applied: Ds4ApplyResult | null, ctx: number): string {
  switch (applied) {
    case "unchanged":
      return `Already using ${fmt(ctx)} tokens.`;
    case "restarted":
      return `Context length set to ${fmt(ctx)}. DwarfStar restarted; the next message reloads the model (about 1-2 min).`;
    case "after_current_requests":
      return `Context length set to ${fmt(ctx)}. It applies after the current reply.`;
    case "next_start":
      return `Context length set to ${fmt(ctx)}. It applies on the next start.`;
    default:
      return `Context length set to ${fmt(ctx)}.`;
  }
}

/** The note under the DwarfStar control when a change is waiting on a restart. */
export function ds4PendingNote(info: Ds4ContextInfo): string | null {
  if (!info.pendingRestart) return null;
  const running = info.ctxActive !== null ? fmt(info.ctxActive) : "the old value";
  return `Restart pending: running with ${running}, the next start uses ${fmt(info.ctx)}.`;
}

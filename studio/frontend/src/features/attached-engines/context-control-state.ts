// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The numbers and wording behind the chat settings "Context length" control for oMLX.

import type { OmlxContextInfo } from "./types.ts";

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

/** Round to the slider step and clamp to the range, for the numeric input. */
export function clampContext(value: number, range: Pick<ContextRange, "min" | "max">): number {
  if (!Number.isFinite(value)) return range.min;
  return Math.min(range.max, Math.max(range.min, Math.round(value)));
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The launch failure the engine wrappers record (see failures.py in the backend). Pure, so the
// node suite can drive it.

import type { AttachedEngineFailure } from "./types.ts";

/** Null unless the marker is complete: a reason, and a failure count of at least one. */
export function failureFromApi(raw: unknown): AttachedEngineFailure | null {
  if (!raw || typeof raw !== "object") return null;
  const { reason, ts, count } = raw as Record<string, unknown>;
  if (typeof reason !== "string" || reason.trim() === "") return null;
  if (typeof count !== "number" || !Number.isInteger(count) || count < 1) return null;
  return {
    reason: reason.trim(),
    ts: typeof ts === "number" && Number.isFinite(ts) ? ts : 0,
    count,
  };
}

/** "oMLX failing: <reason>", with the consecutive count once it has repeated. */
export function failureMessage(
  engine: string,
  failure: AttachedEngineFailure,
): string {
  const repeats = failure.count > 1 ? ` (failed ${failure.count} times)` : "";
  return `${engine} failing: ${failure.reason}${repeats}`;
}

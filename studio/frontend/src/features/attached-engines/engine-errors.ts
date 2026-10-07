// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A 503 from the engine routes means "busy, retry shortly" (a model is being served, a job is
// running, memory is shared): not an error the owner has to act on. Pure, so the node suite can
// drive it.

export const DEFAULT_RETRY_AFTER_S = 15;

export class EngineBusyError extends Error {
  readonly retryAfterS: number;

  constructor(detail: string, retryAfterS: number | null) {
    super(detail);
    this.name = "EngineBusyError";
    this.retryAfterS =
      retryAfterS !== null && retryAfterS > 0 ? retryAfterS : DEFAULT_RETRY_AFTER_S;
  }
}

/** Any other non-2xx reply, with its status so callers can tell "gone" (404) from a real failure. */
export class EngineHttpError extends Error {
  readonly status: number;

  constructor(detail: string, status: number) {
    super(detail);
    this.name = "EngineHttpError";
    this.status = status;
  }
}

/** Seconds from a `Retry-After` header; null for an absent or non-numeric value (HTTP dates too). */
export function parseRetryAfter(header: string | null | undefined): number | null {
  if (typeof header !== "string") return null;
  const trimmed = header.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const seconds = Number(trimmed);
  return seconds > 0 ? seconds : null;
}

/** The retry hint shown instead of an error toast. */
export function busyMessage(error: EngineBusyError): string {
  const detail = error.message.trim();
  const retry = `Try again in about ${error.retryAfterS} s.`;
  return detail ? `${detail.replace(/[.\s]+$/, "")}. ${retry}` : `Busy right now. ${retry}`;
}

/** How a failed engine action should be shown: a calm message for a busy engine, else an error. */
export function describeEngineFailure(
  error: unknown,
  fallback: string,
): { kind: "busy" | "error"; message: string } {
  if (error instanceof EngineBusyError) {
    return { kind: "busy", message: busyMessage(error) };
  }
  return {
    kind: "error",
    message: error instanceof Error && error.message ? error.message : fallback,
  };
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AttachedNotice } from "./types";

const REASON_LABELS: Record<string, string> = {
  training: "Freed engine memory for training",
  local_load: "Freed engine memory for a local model",
  omlx_use: "Switched to oMLX",
};

/** One toast line for an arbiter notice. */
export function noticeMessage(notice: AttachedNotice): string {
  const parts = [REASON_LABELS[notice.reason] ?? "Engine memory changed"];
  if (notice.actions.length > 0) parts.push(notice.actions.join("; "));
  if (notice.inFlightKilled > 0) {
    const count = notice.inFlightKilled;
    parts.push(`${count} in-flight request${count === 1 ? "" : "s"} stopped`);
  }
  return parts.join(". ");
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AttachedProvider, AttachedStatus } from "./types.ts";

/**
 * True only when the engines have been polled and this one answered nothing. Before the first
 * poll, with the flag off, or for a non-owner there is no status, and a row is left alone.
 */
export function isAttachedEngineOffline(
  _provider: AttachedProvider,
  status: AttachedStatus | null,
): boolean {
  if (status === null) return false;
  const engine = status.omlx;
  return engine === null || !engine.reachable;
}

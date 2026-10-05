// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { prepareAttachedProvider } from "./api";
import { useAttachedEnginesStore } from "./attached-engines-store";
import type { AttachedProvider } from "./types";
import { refreshAttachedStatus } from "./use-attached-engines";

/**
 * Picking an attached engine's row tells the backend so it can free oMLX memory. Fire and forget:
 * a failure here must never block the selection, and with the
 * flag off nothing is sent.
 */
export function prepareAttachedEngineOnSelect(provider: AttachedProvider): void {
  if (!useAttachedEnginesStore.getState().settings?.enabled) return;
  void prepareAttachedProvider(provider)
    .then(() => refreshAttachedStatus())
    .catch(() => undefined);
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AttachedEnginesSettings } from "./types.ts";

export type SettingsSaveSync = "remove" | "sync" | "none";

/**
 * What a saved settings change owes the saved provider rows. models_hash covers model ids only,
 * so the poller never notices an engine URL change that serves the same models: the rows keep
 * the old base_url until a /sync rewrites it.
 */
export function settingsSaveSyncAction(
  previous: AttachedEnginesSettings | null,
  saved: AttachedEnginesSettings,
  update: Partial<AttachedEnginesSettings>,
): SettingsSaveSync {
  // Turning the flag off takes the seeded rows out of the picker.
  if (update.enabled === false && !saved.enabled) return "remove";
  if (!saved.enabled || previous === null) return "none";
  const urlChanged = previous.omlxUrl !== saved.omlxUrl;
  return urlChanged ? "sync" : "none";
}

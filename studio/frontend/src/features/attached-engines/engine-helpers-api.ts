// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import type {
  EngineHelpersStatus,
  EngineLifetime,
} from "./engine-helpers-state";

async function call(
  command: string,
  args?: Record<string, unknown>,
): Promise<EngineHelpersStatus> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<EngineHelpersStatus>(command, args);
}

/**
 * Helper status from the desktop shell, or null outside it. A shell built without the
 * `attached-engines` Cargo feature does not register the command, so the rejection also
 * means "feature absent" and hides the toggle.
 */
export async function loadEngineHelpersStatus(): Promise<EngineHelpersStatus | null> {
  if (!isTauri) return null;
  try {
    return await call("engine_helpers_status");
  } catch {
    return null;
  }
}

export function setEngineHelpersEnabled(
  enabled: boolean,
): Promise<EngineHelpersStatus> {
  return call(enabled ? "engine_helpers_enable" : "engine_helpers_disable");
}

/** Saves the engine lifetime in the desktop shell and returns the status with it applied. */
export function setEngineLifetime(
  lifetime: EngineLifetime,
): Promise<EngineHelpersStatus> {
  return call("engine_lifetime_set", { lifetime });
}

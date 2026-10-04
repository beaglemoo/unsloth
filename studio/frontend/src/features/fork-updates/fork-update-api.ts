// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { type ForkUpdateInfo, isForkUpdateInfo } from "./fork-update-state";

async function invokeInfo(
  command: string,
  args?: Record<string, unknown>,
): Promise<ForkUpdateInfo | null> {
  const { invoke } = await import("@tauri-apps/api/core");
  const result = await invoke<unknown>(command, args);
  return isForkUpdateInfo(result) ? result : null;
}

/**
 * The cached fork update state from the desktop shell, or null outside it or in a shell that
 * predates the command (the rejection means "not a fork build" and keeps the upstream UI).
 * Makes no network request.
 */
export async function loadForkUpdateInfo(): Promise<ForkUpdateInfo | null> {
  if (!isTauri) return null;
  try {
    return await invokeInfo("fork_update_info");
  } catch {
    return null;
  }
}

/** Asks GitHub when the cached answer is over a day old, or always when `force`. */
export async function checkForkUpdates(
  force: boolean,
): Promise<ForkUpdateInfo | null> {
  if (!isTauri) return null;
  try {
    return await invokeInfo("fork_update_check", { force });
  } catch {
    return null;
  }
}

/** Opens Terminal running update-fork.sh. Rejects with the reason it could not. */
export async function openForkUpdateTerminal(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("fork_update_open_terminal");
}

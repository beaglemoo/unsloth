// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Mirrors `EngineHelpersStatus` in src-tauri/src/engine_helpers.rs (snake_case states). */
export type EngineHelperState =
  | "not_registered"
  | "enabled"
  | "requires_approval"
  | "not_found"
  | "unsupported"
  | "partial";

export interface EngineHelpersStatus {
  supported: boolean;
  state: EngineHelperState;
  helpers: Array<{ name: string; plist: string; state: EngineHelperState }>;
  error: string | null;
}

export interface EngineHelpersToggle {
  checked: boolean;
  message: string | null;
  /** True when the message reports a failure rather than guidance. */
  isError: boolean;
}

/**
 * What the "Background engines" row shows, or null when the row must stay hidden: not inside
 * the desktop shell, a shell built without the feature (the command is missing), or a
 * platform without SMAppService.
 */
export function engineHelpersToggle(
  status: EngineHelpersStatus | null,
): EngineHelpersToggle | null {
  if (!status || !status.supported) return null;
  const error = status.error ? `Failed: ${status.error}` : null;
  const isError = error !== null;
  switch (status.state) {
    case "enabled":
      return { checked: true, message: error, isError };
    case "requires_approval":
      return {
        checked: true,
        message:
          error ??
          "Approve Unsloth under System Settings > General > Login Items & Extensions to start the engines.",
        isError,
      };
    case "partial":
      return {
        checked: true,
        message:
          error ??
          "Only some of the engine helpers are registered. Turn this off and on again.",
        isError,
      };
    case "not_found":
      return {
        checked: false,
        message: error ?? "This build does not include the engine helpers.",
        isError,
      };
    default:
      return { checked: false, message: error, isError };
  }
}

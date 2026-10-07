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

/** Whether the engines stop when Unsloth quits (`with_app`, the default) or keep serving. */
export type EngineLifetime = "with_app" | "always";

/** The engines the desktop shell can run as helpers. */
export type EngineName = "omlx" | "comfyui";

export interface EngineHelperInfo {
  name: string;
  plist: string;
  state: EngineHelperState;
  /** The master switch and this engine's own choice are both on. Absent from a shell with `cli` below 2. */
  wanted?: boolean;
}

export interface EngineHelpersStatus {
  supported: boolean;
  state: EngineHelperState;
  helpers: EngineHelperInfo[];
  error: string | null;
  /** 2 = per-engine commands (`engine_helper_set_enabled`) exist. Absent from an older shell. */
  cli?: number;
  /** The master "Engines enabled" choice. Absent from a shell older than the lifetime setting. */
  engines_enabled?: boolean;
  /** Absent from a shell older than the lifetime setting, which behaves as "always". */
  engine_lifetime?: EngineLifetime;
}

export interface EngineHelpersToggle {
  checked: boolean;
  message: string | null;
  /** True when the message reports a failure rather than guidance. */
  isError: boolean;
}

/**
 * What the "Engines enabled" row shows, or null when the row must stay hidden: not inside
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

export const LOGIN_ITEMS_NOTE =
  "macOS may ask you to approve Unsloth under System Settings > General > Login Items & Extensions the first time.";

export interface EngineHelperRow extends EngineHelpersToggle {
  /** The helper's own registration state, for the status line ("starting" and the like). */
  state: EngineHelperState;
}

/**
 * The per-engine switch, or null when it must stay hidden: not in the desktop shell, a shell
 * without the per-engine command (`cli` below 2), or a helper the build does not know. The switch
 * shows the user's choice (`wanted`), not the registration state: with the master "Engines
 * enabled" switch off nothing runs, and the message says so.
 */
export function engineHelperRow(
  status: EngineHelpersStatus | null,
  name: EngineName,
): EngineHelperRow | null {
  if (!status || !status.supported || (status.cli ?? 1) < 2) return null;
  const helper = status.helpers.find((entry) => entry.name === name);
  if (!helper) return null;
  const checked = helper.wanted === true;
  const error = status.error ? `Failed: ${status.error}` : null;
  let message: string | null = error;
  if (!message && checked && helper.state === "requires_approval") {
    message =
      "Approve Unsloth under System Settings > General > Login Items & Extensions to start it.";
  } else if (!message && helper.state === "not_found") {
    message = "This build does not include this engine's helper.";
  } else if (!message && !checked && status.engines_enabled === false) {
    message = "Engines are turned off above; turn them on for this engine to run.";
  } else if (!message && !checked && name === "comfyui") {
    message = LOGIN_ITEMS_NOTE;
  }
  return { checked, message, isError: error !== null, state: helper.state };
}

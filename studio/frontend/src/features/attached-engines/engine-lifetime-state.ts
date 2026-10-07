// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  EngineHelpersStatus,
  EngineLifetime,
} from "./engine-helpers-state";

export interface EngineLifetimeOption {
  value: EngineLifetime;
  label: string;
}

export const ENGINE_LIFETIME_OPTIONS: readonly EngineLifetimeOption[] = [
  { value: "with_app", label: "With Unsloth (stop when Unsloth quits)" },
  {
    value: "always",
    label: "Always running (keep serving after Unsloth quits)",
  },
];

export function isEngineLifetime(value: unknown): value is EngineLifetime {
  return value === "with_app" || value === "always";
}

/**
 * The lifetime the control shows. A desktop shell that predates the setting reports none and
 * behaves as "always" (its helpers outlive the app); one that has it defaults to "with_app".
 */
export function engineLifetimeOf(
  status: EngineHelpersStatus | null,
): EngineLifetime {
  if (!status) return "with_app";
  return isEngineLifetime(status.engine_lifetime)
    ? status.engine_lifetime
    : "always";
}

/** The trade-off, shown under the control. Names the clients that care. */
export function engineLifetimeNote(lifetime: EngineLifetime): string {
  return lifetime === "with_app"
    ? "pi, Claude Code and OpenCode only get local models while Unsloth is open. On quit Unsloth unloads the oMLX models and takes the helper down; the next launch starts it again."
    : "pi, Claude Code and OpenCode keep their local models after Unsloth quits, at the cost of the engines holding memory and a login item until you turn them off.";
}

/** What the "Engines enabled" row says about the lifetime. */
export function enginesEnabledDescription(lifetime: EngineLifetime): string {
  return lifetime === "with_app"
    ? "Run the engines you turn on below as macOS login items while Unsloth is open. Each starts with Unsloth and stops when it quits."
    : "Run the engines you turn on below as macOS login items. Each keeps serving after Unsloth quits.";
}

export interface EngineLifetimeRow {
  value: EngineLifetime;
  options: readonly EngineLifetimeOption[];
  note: string;
}

/**
 * The "Engine lifetime" row, or null when it must stay hidden: the same conditions as the
 * Engines enabled row (outside the desktop shell, no feature, no SMAppService), because the
 * setting lives in the desktop shell.
 */
export function engineLifetimeRow(
  status: EngineHelpersStatus | null,
): EngineLifetimeRow | null {
  if (!status || !status.supported) return null;
  const value = engineLifetimeOf(status);
  return {
    value,
    options: ENGINE_LIFETIME_OPTIONS,
    note: engineLifetimeNote(value),
  };
}

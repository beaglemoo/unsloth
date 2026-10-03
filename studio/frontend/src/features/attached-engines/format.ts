// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function formatTps(value: number | null): string {
  return value === null ? "-" : `${value.toFixed(1)} tok/s`;
}

export function formatTtft(ms: number | null): string {
  if (ms === null) return "-";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

export function formatIdleRemaining(seconds: number | null): string {
  if (seconds === null) return "-";
  const whole = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(whole / 60);
  return minutes > 0 ? `${minutes} m ${whole % 60} s` : `${whole} s`;
}

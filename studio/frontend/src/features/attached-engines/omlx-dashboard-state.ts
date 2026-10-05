// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure logic for the oMLX dashboard embedded on the oMLX settings tab.

export const OMLX_DASHBOARD_EXPANDED_KEY = "unsloth.omlxDashboard.expanded";

/** The oMLX admin UI for the configured base URL (`omlx_url`, no `/v1`), or null when the URL is
 *  empty or not http(s), so nothing unexpected is ever framed. */
export function omlxDashboardUrl(baseUrl: string | null | undefined): string | null {
  const base = (baseUrl ?? "").trim().replace(/\/+$/, "");
  if (!/^https?:\/\/\S+$/i.test(base)) return null;
  return `${base}/admin/dashboard`;
}

export type OmlxDashboardView =
  /** The attached-engines flag is off or the settings are not loaded: render nothing. */
  | { kind: "hidden" }
  /** Flag on, oMLX not answering: the muted "not running" line. */
  | { kind: "unreachable" }
  | { kind: "ready"; url: string };

export function omlxDashboardView(input: {
  enabled: boolean;
  reachable: boolean;
  omlxUrl: string | null | undefined;
}): OmlxDashboardView {
  if (!input.enabled) return { kind: "hidden" };
  const url = omlxDashboardUrl(input.omlxUrl);
  if (!input.reachable || url === null) return { kind: "unreachable" };
  return { kind: "ready", url };
}

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function defaultStorage(): StorageLike | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/** Collapsed unless the viewer expanded it before; storage failures read as collapsed. */
export function readDashboardExpanded(storage: StorageLike | null = defaultStorage()): boolean {
  try {
    return storage?.getItem(OMLX_DASHBOARD_EXPANDED_KEY) === "1";
  } catch {
    return false;
  }
}

export function writeDashboardExpanded(
  expanded: boolean,
  storage: StorageLike | null = defaultStorage(),
): void {
  try {
    storage?.setItem(OMLX_DASHBOARD_EXPANDED_KEY, expanded ? "1" : "0");
  } catch {
    // Private mode or a full quota: the state lasts for this viewer session only.
  }
}

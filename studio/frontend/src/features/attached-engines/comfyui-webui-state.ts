// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure logic for the ComfyUI web UI embedded on the Engines settings tab.

export const COMFYUI_WEBUI_EXPANDED_KEY = "unsloth.comfyuiWebUi.expanded";

/**
 * The ComfyUI web UI for the configured base URL, or null unless it is `http://127.0.0.1[:port]`
 * (or `localhost`/`[::1]`): ComfyUI has no auth, so nothing else is ever framed.
 */
export function comfyuiWebUiUrl(baseUrl: string | null | undefined): string | null {
  const base = (baseUrl ?? "").trim();
  const match = /^http:\/\/(127\.0\.0\.1|localhost|\[::1\])(?::(\d{1,5}))?\/?$/i.exec(base);
  if (!match) return null;
  const port = match[2] === undefined ? "" : `:${match[2]}`;
  return `http://${match[1].toLowerCase()}${port}/`;
}

/** The app CSP frames `http://127.0.0.1:*` and `http://localhost:*` only. */
export function comfyuiWebUiFrameable(url: string): boolean {
  return /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?\/$/.test(url);
}

export type ComfyuiWebUiView =
  /** The attached-engines flag is off: render nothing. */
  | { kind: "hidden" }
  /** Flag on, ComfyUI not answering (or no usable URL): the muted "not running" line. */
  | { kind: "unreachable" }
  /** Answering, but the URL cannot be framed (an IPv6 loopback): only "Open in browser". */
  | { kind: "link-only"; url: string }
  | { kind: "ready"; url: string };

export function comfyuiWebUiView(input: {
  enabled: boolean;
  reachable: boolean;
  comfyuiUrl: string | null | undefined;
}): ComfyuiWebUiView {
  if (!input.enabled) return { kind: "hidden" };
  const url = comfyuiWebUiUrl(input.comfyuiUrl);
  if (!input.reachable || url === null) return { kind: "unreachable" };
  return comfyuiWebUiFrameable(url) ? { kind: "ready", url } : { kind: "link-only", url };
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
export function readWebUiExpanded(storage: StorageLike | null = defaultStorage()): boolean {
  try {
    return storage?.getItem(COMFYUI_WEBUI_EXPANDED_KEY) === "1";
  } catch {
    return false;
  }
}

export function writeWebUiExpanded(
  expanded: boolean,
  storage: StorageLike | null = defaultStorage(),
): void {
  try {
    storage?.setItem(COMFYUI_WEBUI_EXPANDED_KEY, expanded ? "1" : "0");
  } catch {
    // Private mode or a full quota: the state lasts for this viewer session only.
  }
}

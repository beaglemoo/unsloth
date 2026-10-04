// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Settings > About > Updates for the beaglemoo fork. The desktop shell (fork_updates.rs) asks
 * github.com/beaglemoo/unsloth how far the branch is ahead of this build. That repo is the only
 * update source this feature knows.
 */

export interface ForkCommit {
  sha: string;
  message: string;
}

/** What the `fork_update_*` commands return. `ahead_by` is null while the answer is unknown. */
export interface ForkUpdateInfo {
  enabled: boolean;
  build_sha: string | null;
  short_sha: string | null;
  branch: string | null;
  checked_at: number | null;
  ahead_by: number | null;
  commits: ForkCommit[];
  html_url: string | null;
  error: string | null;
  can_update: boolean;
}

/** Shown when the app cannot open the script itself (a browser, or no checkout path baked in). */
export const FORK_UPDATE_SCRIPT_HINT =
  "~/Homelab/unsloth/unsloth/studio/scripts/update-fork.sh";

export function isForkUpdateInfo(value: unknown): value is ForkUpdateInfo {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return typeof v.enabled === "boolean" && Array.isArray(v.commits);
}

export type ForkUpdateKind =
  | "disabled"
  | "unbaked"
  | "unknown"
  | "current"
  | "available";

export function forkUpdateKind(info: ForkUpdateInfo): ForkUpdateKind {
  if (!info.enabled) return "disabled";
  if (!info.build_sha) return "unbaked";
  if (info.ahead_by === null) return "unknown";
  return info.ahead_by > 0 ? "available" : "current";
}

/** "Fork build 5426c052d" */
export function forkBuildLabel(info: ForkUpdateInfo): string {
  const short = info.short_sha ?? info.build_sha?.slice(0, 9) ?? null;
  return short ? `Fork build ${short}` : "Fork build (commit unknown)";
}

export function forkUpdateHeadline(info: ForkUpdateInfo): string {
  switch (forkUpdateKind(info)) {
    case "available": {
      const n = info.ahead_by ?? 0;
      return `${n} ${n === 1 ? "update" : "updates"} available`;
    }
    case "current":
      return "Up to date";
    case "unbaked":
      return "This build did not record its commit. Rebuild with build-fork-mac.sh.";
    case "unknown":
      return info.error ?? "Not checked yet";
    default:
      return "";
  }
}

/** The error to show under the headline, when it is not already the headline. */
export function forkUpdateNote(info: ForkUpdateInfo): string | null {
  const kind = forkUpdateKind(info);
  if (kind === "unknown" || kind === "unbaked") return null;
  return info.error;
}

export interface ForkCommitRows {
  rows: { sha: string; message: string }[];
  more: number;
}

/** At most `max` commits for the list, with 7 character shas; `more` counts the rest. */
export function forkCommitRows(info: ForkUpdateInfo, max = 8): ForkCommitRows {
  const total = Math.max(info.ahead_by ?? 0, info.commits.length);
  const rows = info.commits.slice(0, max).map((c) => ({
    sha: c.sha.slice(0, 7),
    message: c.message || "(no message)",
  }));
  return { rows, more: Math.max(0, total - rows.length) };
}

/** The Update button needs updates to install and a script the app can open. */
export function forkCanOpenUpdate(info: ForkUpdateInfo): boolean {
  return forkUpdateKind(info) === "available" && info.can_update;
}

/** Shown when there are updates but the app has no script path to open. */
export function forkUpdateManualHint(info: ForkUpdateInfo): string | null {
  if (forkUpdateKind(info) !== "available" || info.can_update) return null;
  return `Run ${FORK_UPDATE_SCRIPT_HINT} in a terminal.`;
}

export function forkCheckedLabel(
  checkedAt: number | null,
  nowSeconds: number,
): string | null {
  if (checkedAt === null) return null;
  const age = Math.max(0, nowSeconds - checkedAt);
  if (age < 60) return "Checked just now";
  if (age < 3600) return `Checked ${Math.floor(age / 60)} min ago`;
  if (age < 86400) return `Checked ${Math.floor(age / 3600)} h ago`;
  return `Checked ${Math.floor(age / 86400)} d ago`;
}

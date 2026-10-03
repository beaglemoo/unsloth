// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// oMLX lists a model once per directory and again for every alias or profile that shares its
// weights (swift-1.5-27b, swift-1.5-27b:fast, ...). The panel wants one row per set of weights.

import type { AttachedOmlxModel, AttachedOmlxStatus } from "./types.ts";

export type OmlxModelGroup = {
  /** The id Load and Unload send; the backend resolves it to the directory. */
  id: string;
  rowIds: string[];
  loaded: boolean;
  loading: boolean;
  sizeBytes: number;
  pinned: boolean;
  /** Seconds left before the idle unload as of the poll, from the loaded row; null when none applies. */
  idleRemainingS: number | null;
  kind: "chat" | "embedding" | "helper";
};

function baseName(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? "";
}

function groupKind(
  rows: AttachedOmlxModel[],
  chatIds: Set<string>,
): OmlxModelGroup["kind"] {
  if (rows.some((row) => chatIds.has(row.id))) return "chat";
  if (rows.some((row) => row.engineType === "embedding")) return "embedding";
  if (rows.some((row) => row.isHelper)) return "helper";
  return "chat";
}

function minIdleRemaining(rows: AttachedOmlxModel[]): number | null {
  const values = rows
    .map((row) => row.idleRemainingS)
    .filter((value): value is number => value !== null);
  return values.length > 0 ? Math.min(...values) : null;
}

export function groupOmlxModels(status: AttachedOmlxStatus): OmlxModelGroup[] {
  const chatIds = new Set(status.chatModelIds);
  const byPath = new Map<string, AttachedOmlxModel[]>();
  for (const row of status.models) {
    const key = row.modelPath || row.id;
    const rows = byPath.get(key);
    if (rows) rows.push(row);
    else byPath.set(key, [row]);
  }
  const groups: OmlxModelGroup[] = [];
  for (const [path, rows] of byPath) {
    const directory = baseName(path);
    const named = rows.find((row) => row.id === directory) ?? rows[0];
    groups.push({
      id: named.id,
      rowIds: rows.map((row) => row.id),
      loaded: rows.some((row) => row.loaded),
      loading: rows.some((row) => row.isLoading),
      sizeBytes: Math.max(...rows.map((row) => row.estimatedSize)),
      pinned: rows.some((row) => row.pinned),
      idleRemainingS: minIdleRemaining(rows),
      kind: groupKind(rows, chatIds),
    });
  }
  return groups;
}

/** Loaded or loading sets of weights, whatever their kind: they all count against the ceiling. */
export function residentGroups(groups: OmlxModelGroup[]): OmlxModelGroup[] {
  return groups.filter((group) => group.loaded || group.loading);
}

/** Chat models that can be loaded now. */
export function loadableGroups(groups: OmlxModelGroup[]): OmlxModelGroup[] {
  return groups.filter(
    (group) => group.kind === "chat" && !group.loaded && !group.loading,
  );
}

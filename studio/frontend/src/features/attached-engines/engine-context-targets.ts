// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which context-length controls the Attached engines page shows: one per oMLX chat model (the
// prompt cap is per model).

import { groupOmlxModels } from "./omlx-groups.ts";
import type { AttachedProvider, AttachedStatus } from "./types.ts";

export type EngineContextTarget = {
  kind: AttachedProvider;
  modelId: string;
  label: string;
};

export function engineContextTargets(
  status: AttachedStatus | null,
): EngineContextTarget[] {
  if (!status) return [];
  const targets: EngineContextTarget[] = [];
  if (status.omlx?.reachable) {
    for (const group of groupOmlxModels(status.omlx)) {
      if (group.kind !== "chat") continue;
      targets.push({ kind: "omlx", modelId: group.id, label: group.id });
    }
  }
  return targets;
}

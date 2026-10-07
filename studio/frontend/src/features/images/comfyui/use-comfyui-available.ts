// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import {
  selectAttachedEnabled,
  useAttachedEnginesStore,
} from "@/features/attached-engines/attached-engines-store";
import { comfyuiAvailable } from "./comfyui-panel-state";

/** Owner, attached flag on, and ComfyUI either answering or meant to run (its helper is wanted). */
export function useComfyuiAvailable(): boolean {
  const isOwner = useIsAccountOwner();
  const enabled = useAttachedEnginesStore(selectAttachedEnabled);
  const status = useAttachedEnginesStore((s) => s.status?.comfyui ?? null);
  return comfyuiAvailable({ isOwner, enabled, status });
}

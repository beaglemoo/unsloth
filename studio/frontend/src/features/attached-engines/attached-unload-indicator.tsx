// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAuiState } from "@assistant-ui/react";
import { useEffect, useRef } from "react";
import { describeUnload, unloadStateFor } from "./attached-active";
import { useAttachedEnginesStore } from "./attached-engines-store";
import { useActiveAttached, useNow } from "./use-active-attached";
import { refreshAttachedStatus } from "./use-attached-engines";

/** Shown above the composer while an oMLX model is selected: when the engine will
 *  unload it ("Unloads in 4:12"), or that it is loading or not loaded. The countdown ticks every
 *  second and is resynced by each 5 s poll; a poll is also forced when a reply ends, because the
 *  engine restarts the idle clock then. Renders nothing for any other model. */
export function AttachedUnloadIndicator() {
  const active = useActiveAttached();
  const status = useAttachedEnginesStore((s) => s.status);
  const running = useAuiState(({ thread }) => thread.isRunning);
  const wasRunning = useRef(false);
  useEffect(() => {
    const was = wasRunning.current;
    wasRunning.current = running;
    if (!was || running || !active) return undefined;
    const id = setTimeout(() => void refreshAttachedStatus(), 400);
    return () => clearTimeout(id);
  }, [running, active]);

  const state = unloadStateFor(status, active);
  const now = useNow(state?.kind === "countdown");
  const text = describeUnload(state, status?.receivedAt, now);
  if (!text) return null;
  return (
    <div
      role="status"
      data-slot="attached-unload-indicator"
      className="mb-1 px-3 text-xs tabular-nums text-muted-foreground"
    >
      {text}
    </div>
  );
}

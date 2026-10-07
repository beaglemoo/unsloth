// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useAttachedEnginesStore } from "./attached-engines-store";
import { loadEngineHelpersStatus } from "./engine-helpers-api";

/**
 * Keep the shared helper status (the store's `helpers`) fresh while the Engines tab is open: read
 * on mount and again after every engine poll, so "starting" turns into "idle" and an approval in
 * Login Items shows up without reopening the tab. Outside the desktop shell it stays null.
 */
export function useEngineHelpers(): void {
  const receivedAt = useAttachedEnginesStore((s) => s.status?.receivedAt);
  const setHelpers = useAttachedEnginesStore((s) => s.setHelpers);

  useEffect(() => {
    let cancelled = false;
    void loadEngineHelpersStatus().then((status) => {
      if (!cancelled) setHelpers(status);
    });
    return () => {
      cancelled = true;
    };
  }, [receivedAt, setHelpers]);
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { syncExternalProvidersFromBackend } from "@/features/chat/sync-external-providers";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import {
  fetchAttachedStatus,
  loadAttachedEnginesSettings,
  syncAttachedProviders,
} from "./api";
import {
  type AttachedPoller,
  createAttachedPoller,
} from "./attached-engines-poller";
import {
  selectAttachedEnabled,
  useAttachedEnginesStore,
} from "./attached-engines-store";
import { noticeMessage } from "./notice-message";

let activePoller: AttachedPoller | null = null;

/** Pull the saved provider rows into the picker, the way credential bootstrap does. */
export async function syncAttachedProviderRows(): Promise<void> {
  const store = useExternalProvidersStore.getState();
  const synced = await syncExternalProvidersFromBackend(store.providers);
  store.setProviders(synced);
}

/** Poll the engines now, e.g. right after a Load or Start, instead of waiting for the next tick. */
export function refreshAttachedStatus(): Promise<void> {
  return activePoller ? activePoller.tick() : Promise.resolve();
}

/** What the owner turning the flag off must do: the saved rows come out of the picker. */
export async function removeAttachedProviderRows(): Promise<void> {
  await syncAttachedProviders();
  await syncAttachedProviderRows();
  useAttachedEnginesStore.getState().setStatus(null);
}

/**
 * Mount once. Reads the owner's settings, and polls /status every 5 s only while the flag is on:
 * with it off (or for any other account) no request goes to the engine routes at all.
 */
export function useAttachedEngines(): void {
  const isOwner = useIsAccountOwner();
  const enabled = useAttachedEnginesStore(selectAttachedEnabled);

  useEffect(() => {
    if (!isOwner) {
      useAttachedEnginesStore.getState().setSettings(null);
      useAttachedEnginesStore.getState().setStatus(null);
      return;
    }
    let cancelled = false;
    void loadAttachedEnginesSettings()
      .then((settings) => {
        if (!cancelled) useAttachedEnginesStore.getState().setSettings(settings);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [isOwner]);

  useEffect(() => {
    if (!isOwner || !enabled) return;
    const poller = createAttachedPoller({
      fetchStatus: fetchAttachedStatus,
      postSync: syncAttachedProviders,
      syncProviders: syncAttachedProviderRows,
      onStatus: (status) => {
        const store = useAttachedEnginesStore.getState();
        store.setStatus(status);
        // A 404 means the flag went off behind our back (an env override, another session):
        // re-read the settings so the toggle and the panel follow it instead of hanging.
        if (status === null) {
          void loadAttachedEnginesSettings()
            .then((settings) => store.setSettings(settings))
            .catch(() => undefined);
        }
      },
      onNotice: (notice) => {
        toast.info(noticeMessage(notice));
      },
    });
    activePoller = poller;
    poller.start();
    return () => {
      poller.stop();
      if (activePoller === poller) activePoller = null;
    };
  }, [isOwner, enabled]);
}

/** Null-rendering mount point for the root layout. */
export function AttachedEnginesMount(): null {
  useAttachedEngines();
  return null;
}

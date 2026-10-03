// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The poll loop behind useAttachedEngines, free of React and of "@/" imports so a node test can
// drive it with fake timers and stubbed requests.

import type {
  AttachedNotice,
  AttachedStatus,
  AttachedSyncResult,
} from "./types.ts";

export const ATTACHED_POLL_INTERVAL_MS = 5000;

// Bound on remembered notice stamps; the backend keeps 20.
const MAX_SEEN_NOTICES = 64;

export type AttachedPollerDeps = {
  /** null when the backend answers 404: the flag is off. */
  fetchStatus: (since: number) => Promise<AttachedStatus | null>;
  postSync: () => Promise<AttachedSyncResult | void>;
  /** Pull the saved provider rows into the picker store. */
  syncProviders: () => Promise<void>;
  onStatus: (status: AttachedStatus | null) => void;
  onNotice: (notice: AttachedNotice) => void;
  /** The backend says the feature is off (404): run the disabled-feature sync and reload the
   *  provider rows so the attached models leave the picker. */
  onDisabled?: () => Promise<void>;
  setTimer?: (fn: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
  intervalMs?: number;
};

export type AttachedPoller = {
  start: () => void;
  stop: () => void;
  /** One poll now, e.g. right after a Load or Start so the UI need not wait for the next tick. */
  tick: () => Promise<void>;
};

export function createAttachedPoller(deps: AttachedPollerDeps): AttachedPoller {
  const setTimer = deps.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
  const clearTimer =
    deps.clearTimer ?? ((handle) => clearTimeout(handle as number));
  const intervalMs = deps.intervalMs ?? ATTACHED_POLL_INTERVAL_MS;

  let running = false;
  let generation = 0;
  let timer: unknown = null;
  let busy: Promise<void> | null = null;
  let lastHash: string | null = null;
  // The first reply only establishes what already happened: notices from before this page
  // opened are not news, and the browser clock is not trusted to filter them.
  let primed = false;
  let lastTs = 0;
  const seen = new Set<number>();

  const schedule = (gen: number) => {
    if (!running || gen !== generation) return;
    timer = setTimer(() => {
      timer = null;
      void runTick(gen).then(() => schedule(gen));
    }, intervalMs);
  };

  const runTick = (gen: number): Promise<void> => {
    if (busy) return busy;
    busy = (async () => {
      let status: AttachedStatus | null;
      try {
        status = await deps.fetchStatus(lastTs);
      } catch {
        // Transient (network, auth refresh): keep the last status and try again next tick.
        return;
      }
      if (gen !== generation) return;
      if (status === null) {
        lastHash = null;
        primed = false;
        deps.onStatus(null);
        // The flag is off; nothing to poll until the settings say otherwise.
        running = false;
        try {
          await deps.onDisabled?.();
        } catch {
          // Best effort: the rows are only cosmetic once the engines are off.
        }
        return;
      }
      deps.onStatus(status);
      for (const notice of status.notices) {
        if (seen.has(notice.ts)) continue;
        seen.add(notice.ts);
        if (seen.size > MAX_SEEN_NOTICES) {
          const oldest = seen.values().next().value;
          if (oldest !== undefined) seen.delete(oldest);
        }
        if (notice.ts > lastTs) lastTs = notice.ts;
        if (primed) deps.onNotice(notice);
      }
      primed = true;
      if (status.modelsHash !== lastHash) {
        try {
          const result = await deps.postSync();
          await deps.syncProviders();
          // An incomplete catalog fetch kept the old models: leave lastHash stale so the next
          // tick syncs again instead of treating the picker as up to date.
          if (!(result && result.incomplete)) lastHash = status.modelsHash;
        } catch {
          // lastHash stays stale, so the next tick retries the sync.
        }
      }
    })().finally(() => {
      busy = null;
    });
    return busy;
  };

  return {
    start() {
      if (running) return;
      running = true;
      generation += 1;
      const gen = generation;
      void runTick(gen).then(() => schedule(gen));
    },
    stop() {
      running = false;
      generation += 1;
      if (timer !== null) {
        clearTimer(timer);
        timer = null;
      }
    },
    tick() {
      return runTick(generation);
    },
  };
}

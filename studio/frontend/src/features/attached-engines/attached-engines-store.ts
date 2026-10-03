// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { AttachedEnginesSettings, AttachedStatus } from "./types";

interface AttachedEnginesState {
  /** Null until the owner's settings are read, and for accounts that cannot read them. */
  settings: AttachedEnginesSettings | null;
  /** Null while the flag is off, nothing has been polled yet, or the account is not the owner. */
  status: AttachedStatus | null;
  setSettings: (settings: AttachedEnginesSettings | null) => void;
  setStatus: (status: AttachedStatus | null) => void;
}

export const useAttachedEnginesStore = create<AttachedEnginesState>((set) => ({
  settings: null,
  status: null,
  setSettings: (settings) => set({ settings }),
  setStatus: (status) => set({ status }),
}));

export const selectAttachedEnabled = (state: AttachedEnginesState): boolean =>
  state.settings?.enabled === true;

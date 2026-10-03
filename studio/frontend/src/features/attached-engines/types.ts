// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type AttachedOmlxModel = {
  id: string;
  modelPath: string;
  loaded: boolean;
  isLoading: boolean;
  estimatedSize: number;
  pinned: boolean;
  engineType: string | null;
  isHelper: boolean;
  modelAlias: string | null;
};

export type AttachedOmlxStatus = {
  reachable: boolean;
  models: AttachedOmlxModel[];
  memoryBytes: number;
  ceilingBytes: number;
  error: string | null;
  // Ids /v1/models lists minus embedding and helper models: what the picker offers.
  chatModelIds: string[];
};

export type AttachedDs4Status = {
  reachable: boolean;
  loaded: boolean;
  starting: boolean;
  pid: number | null;
  uptimeS: number | null;
  inFlight: number;
  idleRemainingS: number | null;
  liveTps: number | null;
  lastGenTps: number | null;
  lastTtftMs: number | null;
  startTimeoutS: number;
  error: string | null;
};

export type AttachedNotice = {
  // Epoch seconds, as the backend stamps it.
  ts: number;
  reason: string;
  actions: string[];
  inFlightKilled: number;
};

export type AttachedStatus = {
  omlx: AttachedOmlxStatus | null;
  ds4: AttachedDs4Status | null;
  modelsHash: string;
  notices: AttachedNotice[];
};

export type AttachedEnginesSettings = {
  enabled: boolean;
  omlxUrl: string;
  ds4Url: string;
  scanDenylist: string[];
  arbitrateLocalLoads: boolean;
  prewarmDs4OnSelect: boolean;
};

export type AttachedProvider = "omlx" | "dwarfstar";

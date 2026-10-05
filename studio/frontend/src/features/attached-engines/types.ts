// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The launch failure marker the engine wrappers leave when a precondition stops a start (missing
 *  venv, config or model, port in use): `reason`, epoch seconds, and consecutive failures. */
export type AttachedEngineFailure = {
  reason: string;
  ts: number;
  count: number;
};

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
  /** Resolved prompt cap, the model's native length and its output cap, when the engine reports them. */
  maxContextWindow: number | null;
  modelContextLength: number | null;
  maxTokens: number | null;
  /** Idle-unload TTL that applies, and seconds left as of the poll. Null while unloaded, pinned or untimed. */
  ttlS: number | null;
  idleRemainingS: number | null;
};

export type AttachedOmlxStatus = {
  reachable: boolean;
  models: AttachedOmlxModel[];
  memoryBytes: number;
  ceilingBytes: number;
  error: string | null;
  // Ids /v1/models lists minus embedding and helper models: what the picker offers.
  chatModelIds: string[];
  failure: AttachedEngineFailure | null;
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
  modelsHash: string;
  notices: AttachedNotice[];
  /** Epoch ms when this status was read: the base idle countdowns run down from. */
  receivedAt?: number;
};

/** The per-model oMLX context cap (POST /omlx/context/get). `maxContextWindow` is the saved
 *  setting (null = default), `effective` the value in force. */
export type OmlxContextInfo = {
  modelId: string;
  dir: string;
  maxContextWindow: number | null;
  nativeMax: number | null;
  effective: number | null;
};

export type AttachedEnginesSettings = {
  enabled: boolean;
  omlxUrl: string;
  scanDenylist: string[];
  arbitrateLocalLoads: boolean;
};

/** What POST /sync reports. `incomplete`: an engine's model catalog could not be read, so its
 *  saved models were kept (`kept_models`) and the sync should be retried. */
export type AttachedSyncResult = {
  enabled: boolean;
  incomplete: boolean;
};

export type AttachedProvider = "omlx";

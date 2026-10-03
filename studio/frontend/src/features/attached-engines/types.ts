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
  /** Configured context length, the one the running server started with, and a queued restart. */
  ctx: number | null;
  ctxActive: number | null;
  pendingRestart: boolean;
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

export type Ds4ApplyResult =
  | "next_start"
  | "restarted"
  | "after_current_requests"
  | "unchanged";

/** The DwarfStar launcher's context config (GET/POST /ds4/context). */
export type Ds4ContextInfo = {
  ctx: number;
  ctxActive: number | null;
  ctxMin: number;
  ctxMax: number;
  pendingRestart: boolean;
  applied: Ds4ApplyResult | null;
};

export type AttachedEnginesSettings = {
  enabled: boolean;
  omlxUrl: string;
  ds4Url: string;
  scanDenylist: string[];
  arbitrateLocalLoads: boolean;
  prewarmDs4OnSelect: boolean;
};

/** What POST /sync reports. `incomplete`: an engine's model catalog could not be read, so its
 *  saved models were kept (`kept_models`) and the sync should be retried. */
export type AttachedSyncResult = {
  enabled: boolean;
  incomplete: boolean;
};

export type AttachedProvider = "omlx" | "dwarfstar";

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

/** ComfyUI's queue as the backend classifies it: `down` is confirmed not running, `unknown` means
 *  it answered without a usable queue. */
export type ComfyuiQueueState = "down" | "unknown" | "busy" | "idle";

/** A ComfyUI Studio does not own (StoryPress's on :8188). Studio only reads its queue. */
export type ComfyuiPeer = {
  url: string;
  reachable: boolean;
  busy: boolean;
  state: ComfyuiQueueState;
};

export type ComfyuiDevice = {
  name: string;
  type: string;
};

/** The `comfyui` block of GET /status. */
export type ComfyuiStatus = {
  url: string;
  reachable: boolean;
  state: ComfyuiQueueState;
  version: string | null;
  queueRunning: number;
  queuePending: number;
  devices: ComfyuiDevice[];
  ramTotal: number | null;
  ramFree: number | null;
  failure: AttachedEngineFailure | null;
  /** The desktop shell's choice for the helper as the backend reads it; null when unknown. */
  helperWanted: boolean | null;
  peers: ComfyuiPeer[];
};

/** One row of GET /comfyui/queue. `studio` is the marker of a job Studio submitted. */
export type ComfyuiQueueRow = {
  promptId: string;
  number: number | null;
  state: "running" | "pending";
  studio: boolean;
};

/** GET /comfyui/models: file names per ComfyUI model folder. */
export type ComfyuiModels = Record<string, string[]>;

export type AttachedStatus = {
  omlx: AttachedOmlxStatus | null;
  /** Absent from a backend older than the ComfyUI engine. */
  comfyui?: ComfyuiStatus | null;
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
  comfyuiUrl: string;
  comfyuiPeerUrls: string[];
  arbitrateComfyui: boolean;
  /** Seconds of idleness before Studio frees its ComfyUI; 0 turns it off. */
  comfyuiIdleFreeS: number;
};

/** What POST /sync reports. `incomplete`: an engine's model catalog could not be read, so its
 *  saved models were kept (`kept_models`) and the sync should be retried. */
export type AttachedSyncResult = {
  enabled: boolean;
  incomplete: boolean;
};

export type AttachedProvider = "omlx";

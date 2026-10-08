// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure logic for the ComfyUI section of the Engines settings tab: the status line, the unload
// rule, the queue and model lists, and the wording of the replies. No React and no "@/" imports,
// so the node suite can drive it.

import { formatCountdown, remainingNow } from "./attached-active.ts";
import { EngineHttpError } from "./engine-errors.ts";
import { failureMessage } from "./failure.ts";
import type { EngineHelperRow } from "./engine-helpers-state.ts";
import type {
  ComfyuiModels,
  ComfyuiPeer,
  ComfyuiQueueRow,
  ComfyuiStatus,
} from "./types.ts";

export type StatusTone = "ok" | "busy" | "muted" | "warn" | "error";

export interface StatusLine {
  text: string;
  tone: StatusTone;
}

/** The hint added when the marker says another ComfyUI (StoryPress's) holds the port. */
export const STORYPRESS_PORT_HINT =
  "StoryPress's ComfyUI holds port 8844. Quit it, or change the ComfyUI port in engines.toml.";

export function failureText(status: ComfyuiStatus): string | null {
  if (!status.failure) return null;
  const base = failureMessage("ComfyUI", status.failure);
  return /storypress|8844/i.test(status.failure.reason)
    ? `${base}. ${STORYPRESS_PORT_HINT}`
    : base;
}

/**
 * One line for the state of Studio's own ComfyUI. A launch-failure marker only counts while
 * ComfyUI is not answering (the wrapper removes it right before it execs ComfyUI).
 */
export function comfyuiStatusLine(input: {
  status: ComfyuiStatus | null | undefined;
  helper: EngineHelperRow | null;
  /** The poll time and the current time, to count the idle-free timer down between polls. */
  receivedAt?: number;
  now?: number;
}): StatusLine {
  const { status, helper } = input;
  if (!status) return { text: "Status unavailable", tone: "muted" };
  if (!status.reachable) {
    const failing = failureText(status);
    if (failing) return { text: failing, tone: "error" };
  }
  switch (status.state) {
    case "idle": {
      const base = status.idleFreeInS;
      if (base === null || base === undefined) return { text: "Idle", tone: "ok" };
      const left = remainingNow(base, input.receivedAt, input.now ?? Date.now());
      return {
        text: left <= 0 ? "Idle, freeing memory" : `Idle, frees memory in ${formatCountdown(left)}`,
        tone: "ok",
      };
    }
    case "busy":
      return status.queueRunning > 0
        ? { text: `Generating (${status.queuePending} queued)`, tone: "busy" }
        : { text: `Queued (${status.queuePending})`, tone: "busy" };
    case "unknown":
      return {
        text: "Unknown: ComfyUI answered but did not report its queue",
        tone: "warn",
      };
    default: {
      if (helper?.checked && helper.state === "requires_approval") {
        return {
          text: "Waiting for approval in Login Items",
          tone: "warn",
        };
      }
      const wanted = helper ? helper.checked : status.helperWanted === true;
      const registered = helper ? helper.state === "enabled" : wanted;
      return wanted && registered
        ? { text: "Starting", tone: "muted" }
        : { text: "Not running", tone: "muted" };
    }
  }
}

export interface UnloadRule {
  enabled: boolean;
  /** Why it is off, shown under the button. Null when it is on, or when ComfyUI is simply down. */
  reason: string | null;
}

/** The Unload (Free memory) button: only on an idle ComfyUI. The tray applies the same rule. */
export function comfyuiUnloadRule(status: ComfyuiStatus | null | undefined): UnloadRule {
  if (!status || status.state === "down") return { enabled: false, reason: null };
  if (status.state === "unknown") {
    return { enabled: false, reason: "ComfyUI did not report its queue." };
  }
  if (status.state === "busy") {
    return {
      enabled: false,
      reason: "A job is running or queued. Unload after it ends; nothing is interrupted.",
    };
  }
  return { enabled: true, reason: null };
}

/** POST /comfyui/free answers 200 with `deferred` while a job runs (ComfyUI applies it after). */
export function freeResultMessage(result: { freed: boolean; deferred: boolean }): string {
  return result.deferred
    ? "Memory will be freed when the current job finishes."
    : "ComfyUI models unloaded and memory freed.";
}

export function formatBytes(bytes: number | null): string {
  if (bytes === null || !Number.isFinite(bytes) || bytes < 0) return "-";
  const gib = bytes / 1024 ** 3;
  if (gib >= 1) return `${gib.toFixed(1)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

/** "Apple M5 Pro (mps), 12.3 GB free", the memory being system-wide on Apple Silicon. */
export function comfyuiDeviceLine(status: ComfyuiStatus): string | null {
  const parts: string[] = [];
  const device = status.devices[0];
  if (device?.name) parts.push(device.type ? `${device.name} (${device.type})` : device.name);
  if (status.ramFree !== null) parts.push(`${formatBytes(status.ramFree)} free`);
  return parts.length > 0 ? parts.join(", ") : null;
}

function portOf(url: string): string {
  const match = /:(\d+)\/?$/.exec(url.trim());
  return match ? match[1] : url;
}

/** One line per reachable peer ComfyUI. Studio waits for a generating peer before it starts work. */
export function peerNotes(peers: readonly ComfyuiPeer[]): string[] {
  return peers
    .filter((peer) => peer.reachable)
    .map((peer) => {
      const port = portOf(peer.url);
      const name = port === "8188" ? "StoryPress ComfyUI" : "Another ComfyUI";
      return peer.busy
        ? `${name} on :${port} is generating (shares memory); Studio waits for it before starting local work.`
        : `${name} is also running on :${port} (shares memory).`;
    });
}

export function queueRowTitle(row: ComfyuiQueueRow): string {
  const id = row.promptId.length > 8 ? row.promptId.slice(0, 8) : row.promptId;
  const origin = row.studio ? "Studio job" : "Job";
  return `${origin} ${id}`;
}

export function queueRowDetail(row: ComfyuiQueueRow, position: number): string {
  return row.state === "running" ? "Running" : `Queued, position ${position}`;
}

/** The pending rows are numbered from 1 in queue order. */
export function pendingPosition(rows: readonly ComfyuiQueueRow[], row: ComfyuiQueueRow): number {
  return rows.filter((r) => r.state === "pending").indexOf(row) + 1;
}

/** A cancel can find the job already gone (404): the queue moved on, which is not an error. */
export function cancelOutcome(error: unknown): "gone" | "error" {
  return error instanceof EngineHttpError && error.status === 404 ? "gone" : "error";
}

export interface ModelGroup {
  folder: string;
  label: string;
  files: string[];
}

const FOLDER_LABELS: ReadonlyArray<readonly [string, string]> = [
  ["diffusion_models", "Diffusion models"],
  ["checkpoints", "Checkpoints"],
  ["text_encoders", "Text encoders"],
  ["vae", "VAE"],
  ["loras", "LoRA"],
];

/** The model folders in display order, empty ones left out. Unknown folders follow, by name. */
export function groupComfyuiModels(models: ComfyuiModels): ModelGroup[] {
  const known = new Set(FOLDER_LABELS.map(([folder]) => folder));
  const groups: ModelGroup[] = FOLDER_LABELS.map(([folder, label]) => ({
    folder,
    label,
    files: models[folder] ?? [],
  }));
  for (const folder of Object.keys(models).sort()) {
    if (!known.has(folder)) groups.push({ folder, label: folder, files: models[folder] });
  }
  return groups.filter((group) => group.files.length > 0);
}

export function modelCountText(groups: readonly ModelGroup[]): string {
  const total = groups.reduce((sum, group) => sum + group.files.length, 0);
  return total === 0 ? "No models found" : `${total} model file${total === 1 ? "" : "s"}`;
}

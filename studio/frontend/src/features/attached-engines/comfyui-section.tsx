// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { useEffect, useState } from "react";
import { updateAttachedEnginesSettings } from "./api";
import {
  selectAttachedEnabled,
  useAttachedEnginesStore,
} from "./attached-engines-store";
import {
  cancelComfyuiJob,
  fetchComfyuiModels,
  fetchComfyuiQueue,
  freeComfyui,
} from "./comfyui-api";
import {
  cancelOutcome,
  comfyuiDeviceLine,
  comfyuiStatusLine,
  comfyuiUnloadRule,
  freeResultMessage,
  groupComfyuiModels,
  modelCountText,
  pendingPosition,
  peerNotes,
  queueRowDetail,
  queueRowTitle,
  type StatusTone,
} from "./comfyui-state";
import { setEngineHelperEnabled } from "./engine-helpers-api";
import { engineHelperRow } from "./engine-helpers-state";
import { describeEngineFailure } from "./engine-errors";
import { useNow } from "./use-active-attached";
import { refreshAttachedStatus } from "./use-attached-engines";
import type { ComfyuiModels, ComfyuiQueueRow } from "./types";

const INFO_CLASS =
  "max-w-[calc(320px*var(--ui-space-scale,1))] text-right text-xs text-muted-foreground";
const ERROR_CLASS =
  "max-w-[calc(320px*var(--ui-space-scale,1))] text-right text-xs text-destructive";

const TONE_CLASS: Record<StatusTone, string> = {
  ok: "text-emerald-600 dark:text-emerald-400",
  busy: "text-amber-600 dark:text-amber-400",
  warn: "text-amber-600 dark:text-amber-400",
  error: "text-destructive",
  muted: "text-muted-foreground",
};

type Message = { kind: "info" | "error"; text: string } | null;

function MessageLine({ message }: { message: Message }) {
  if (!message) return null;
  return (
    <span className={message.kind === "error" ? ERROR_CLASS : INFO_CLASS}>
      {message.text}
    </span>
  );
}

/** A busy engine (503) reads as a retry hint, any other failure as an error. */
function messageFromFailure(error: unknown, fallback: string): Message {
  const shown = describeEngineFailure(error, fallback);
  return { kind: shown.kind === "busy" ? "info" : "error", text: shown.message };
}

function QueueRows({
  rows,
  onChange,
}: {
  rows: ComfyuiQueueRow[];
  onChange: () => void;
}) {
  const [cancelling, setCancelling] = useState<string | null>(null);
  const [message, setMessage] = useState<Message>(null);

  const cancel = (row: ComfyuiQueueRow) => {
    setCancelling(row.promptId);
    setMessage(null);
    void cancelComfyuiJob(row.promptId)
      .catch((error) => {
        if (cancelOutcome(error) === "gone") {
          setMessage({ kind: "info", text: "That job already finished." });
        } else {
          setMessage(messageFromFailure(error, "Could not cancel the job"));
        }
      })
      .then(() => onChange())
      .finally(() => setCancelling(null));
  };

  return (
    <>
      {rows.map((row) => (
        <SettingsRow
          key={row.promptId}
          label={queueRowTitle(row)}
          description={queueRowDetail(row, pendingPosition(rows, row))}
        >
          <Button
            variant="outline"
            size="sm"
            disabled={cancelling !== null}
            onClick={() => cancel(row)}
          >
            {cancelling === row.promptId
              ? "Cancelling"
              : row.state === "running"
                ? "Interrupt"
                : "Cancel"}
          </Button>
        </SettingsRow>
      ))}
      {message ? (
        <div className="flex justify-end pb-2">
          <MessageLine message={message} />
        </div>
      ) : null}
    </>
  );
}

function ModelsList({ reachable }: { reachable: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const [models, setModels] = useState<ComfyuiModels | null>(null);
  const [message, setMessage] = useState<Message>(null);
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    setMessage(null);
    void fetchComfyuiModels()
      .then(setModels)
      .catch((error) =>
        setMessage(messageFromFailure(error, "Could not list the ComfyUI models")),
      )
      .finally(() => setLoading(false));
  };

  // Listed when the section opens, and again when ComfyUI comes back up.
  useEffect(() => {
    if (expanded && reachable) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expanded, reachable]);

  const groups = models ? groupComfyuiModels(models) : [];
  return (
    <div className="flex flex-col gap-2 py-2">
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          aria-expanded={expanded}
          disabled={!reachable}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Hide models" : "Show models"}
        </Button>
        {expanded ? (
          <Button variant="outline" size="sm" disabled={loading || !reachable} onClick={load}>
            {loading ? "Loading" : "Refresh"}
          </Button>
        ) : null}
        {!reachable ? (
          <span className="text-xs text-muted-foreground">
            Available while ComfyUI is running.
          </span>
        ) : null}
      </div>
      <MessageLine message={message} />
      {expanded && models ? (
        <div className="flex flex-col gap-3 text-xs">
          <span className="text-muted-foreground">{modelCountText(groups)}</span>
          {groups.map((group) => (
            <div key={group.folder} className="flex flex-col gap-1">
              <span className="font-medium text-foreground">
                {group.label} ({group.files.length})
              </span>
              <ul className="flex flex-col gap-0.5 text-muted-foreground">
                {group.files.map((file) => (
                  <li key={file} className="break-all font-mono">
                    {file}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** ComfyUI's own settings: where Studio reaches it, memory arbitration and the idle free. */
function ComfyuiSettingsRows() {
  const settings = useAttachedEnginesStore((s) => s.settings);
  const setSettings = useAttachedEnginesStore((s) => s.setSettings);
  const [urlDraft, setUrlDraft] = useState<string | null>(null);
  const [idleDraft, setIdleDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<Message>(null);
  if (!settings) return null;

  const url = urlDraft ?? settings.comfyuiUrl;
  const idle = idleDraft ?? String(settings.comfyuiIdleFreeS);
  const idleNumber = Number(idle);
  const idleValid =
    idle.trim() !== "" && Number.isInteger(idleNumber) && idleNumber >= 0 && idleNumber <= 86400;
  const urlDirty = urlDraft !== null && urlDraft.trim() !== settings.comfyuiUrl;
  const idleDirty = idleDraft !== null && idleValid && idleNumber !== settings.comfyuiIdleFreeS;

  const persist = async (update: Parameters<typeof updateAttachedEnginesSettings>[0]) => {
    setSaving(true);
    setMessage(null);
    try {
      setSettings(await updateAttachedEnginesSettings(update));
      setUrlDraft(null);
      setIdleDraft(null);
      await refreshAttachedStatus();
    } catch (error) {
      setMessage(messageFromFailure(error, "Failed to update the ComfyUI settings"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <SettingsRow
        label="ComfyUI URL"
        description="Loopback address of Studio's ComfyUI."
        below={<MessageLine message={message} />}
      >
        <Input
          value={url}
          aria-label="ComfyUI URL"
          disabled={saving}
          onChange={(event) => setUrlDraft(event.target.value)}
          className="h-8 w-64"
        />
      </SettingsRow>
      {urlDirty ? (
        <SettingsRow label="Apply ComfyUI URL">
          <Button
            variant="outline"
            size="sm"
            disabled={saving}
            onClick={() => void persist({ comfyuiUrl: url.trim() })}
          >
            {saving ? "Saving" : "Save"}
          </Button>
        </SettingsRow>
      ) : null}
      <SettingsRow
        label="Arbitrate ComfyUI memory"
        description="Unload oMLX before a ComfyUI job, free ComfyUI before a local load, and wait while a ComfyUI is generating."
      >
        <Switch
          checked={settings.arbitrateComfyui}
          disabled={saving}
          onCheckedChange={(arbitrateComfyui) => void persist({ arbitrateComfyui })}
        />
      </SettingsRow>
      <SettingsRow
        label="Free ComfyUI after idle (s)"
        description="Seconds after a Studio ComfyUI job before its models are unloaded. 0 turns it off. Takes effect once Studio starts ComfyUI jobs itself."
      >
        <Input
          value={idle}
          inputMode="numeric"
          aria-label="Free ComfyUI after idle (seconds)"
          aria-invalid={!idleValid}
          disabled={saving}
          onChange={(event) => setIdleDraft(event.target.value)}
          className="h-8 w-24"
        />
      </SettingsRow>
      {idleDirty ? (
        <SettingsRow label="Apply idle free">
          <Button
            variant="outline"
            size="sm"
            disabled={saving}
            onClick={() => void persist({ comfyuiIdleFreeS: idleNumber })}
          >
            {saving ? "Saving" : "Save"}
          </Button>
        </SettingsRow>
      ) : null}
    </>
  );
}

/**
 * ComfyUI on the Engines tab: the per-engine switch, live state, unload, queue, models and settings.
 * Owner-only; hidden while the attached-engines flag is off. State comes from the engine poll
 * (`status.comfyui`); the queue is read once per change of the queue counts, the models on demand.
 */
export function ComfyuiSection() {
  const enabled = useAttachedEnginesStore(selectAttachedEnabled);
  const status = useAttachedEnginesStore((s) => s.status);
  const helpers = useAttachedEnginesStore((s) => s.helpers);
  const setHelpers = useAttachedEnginesStore((s) => s.setHelpers);
  const comfyui = status?.comfyui ?? null;
  const [helperBusy, setHelperBusy] = useState(false);
  const [helperError, setHelperError] = useState<string | null>(null);
  const [unloading, setUnloading] = useState(false);
  const [unloadMessage, setUnloadMessage] = useState<Message>(null);
  const [queue, setQueue] = useState<ComfyuiQueueRow[]>([]);

  const queueKey = comfyui
    ? `${comfyui.reachable}:${comfyui.queueRunning}:${comfyui.queuePending}`
    : "none";
  const loadQueue = () => {
    void fetchComfyuiQueue()
      .then(setQueue)
      .catch(() => setQueue([]));
  };
  useEffect(() => {
    if (!comfyui?.reachable || comfyui.queueRunning + comfyui.queuePending === 0) {
      setQueue([]);
      return;
    }
    loadQueue();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queueKey]);

  // Ticks once a second only while an idle-free timer is counting down.
  const now = useNow(comfyui?.state === "idle" && comfyui.idleFreeInS != null);

  if (!enabled) return null;

  const helperRow = engineHelperRow(helpers, "comfyui");
  const line = comfyuiStatusLine({
    status: comfyui,
    helper: helperRow,
    receivedAt: status?.receivedAt,
    now,
  });
  const details = comfyui?.reachable ? comfyuiDeviceLine(comfyui) : null;
  const rule = comfyuiUnloadRule(comfyui);
  const peers = comfyui ? peerNotes(comfyui.peers) : [];

  const toggleHelper = async (next: boolean) => {
    setHelperBusy(true);
    setHelperError(null);
    try {
      setHelpers(await setEngineHelperEnabled("comfyui", next));
      await refreshAttachedStatus();
    } catch (error) {
      setHelperError(
        error instanceof Error ? error.message : "Failed to change the ComfyUI engine",
      );
    } finally {
      setHelperBusy(false);
    }
  };

  const unload = () => {
    setUnloading(true);
    setUnloadMessage(null);
    void freeComfyui()
      .then((result) =>
        setUnloadMessage({ kind: "info", text: freeResultMessage(result) }),
      )
      .catch((error) =>
        setUnloadMessage(messageFromFailure(error, "Could not unload the ComfyUI models")),
      )
      .then(() => refreshAttachedStatus())
      .finally(() => setUnloading(false));
  };

  return (
    <SettingsSection
      title="ComfyUI"
      description="Image generation engine. It shares unified memory with oMLX, so Studio keeps the two apart."
    >
      {helperRow ? (
        <SettingsRow
          label="Run ComfyUI"
          description="Start ComfyUI as a background helper on port 8844. Off by default. Turning it off interrupts a running job and frees its memory."
          below={
            helperError ? (
              <span className={ERROR_CLASS}>{helperError}</span>
            ) : helperRow.message ? (
              <span className={helperRow.isError ? ERROR_CLASS : INFO_CLASS}>
                {helperRow.message}
              </span>
            ) : null
          }
        >
          <Switch
            checked={helperRow.checked}
            disabled={helperBusy}
            onCheckedChange={(next) => void toggleHelper(next)}
          />
        </SettingsRow>
      ) : null}
      <SettingsRow
        label="ComfyUI status"
        description={<span className={TONE_CLASS[line.tone]}>{line.text}</span>}
      />
      {comfyui?.reachable ? (
        <SettingsRow
          label="Details"
          description={[comfyui.version ? `ComfyUI ${comfyui.version}` : null, details]
            .filter(Boolean)
            .join(", ") || "Running"}
        />
      ) : null}
      {peers.map((note) => (
        <SettingsRow key={note} label="Peer" description={note} />
      ))}
      <SettingsRow
        label="Unload models"
        description={
          rule.reason ??
          "Unload ComfyUI's models and free their memory. A running job is never interrupted."
        }
        below={<MessageLine message={unloadMessage} />}
      >
        <Button
          variant="outline"
          size="sm"
          disabled={!rule.enabled || unloading}
          onClick={unload}
        >
          {unloading ? "Unloading" : "Unload"}
        </Button>
      </SettingsRow>
      {queue.length > 0 ? <QueueRows rows={queue} onChange={loadQueue} /> : null}
      <ModelsList reachable={comfyui?.reachable === true} />
      <ComfyuiSettingsRows />
    </SettingsSection>
  );
}

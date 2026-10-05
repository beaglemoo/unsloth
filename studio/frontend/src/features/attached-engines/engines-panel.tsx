// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { formatBytes } from "@/features/hub";
import { toast } from "@/lib/toast";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { useState } from "react";
import { loadOmlxModel, unloadAllOmlxModels, unloadOmlxModel } from "./api";
import {
  selectAttachedEnabled,
  useAttachedEnginesStore,
} from "./attached-engines-store";
import { formatCountdown, remainingNow } from "./attached-active";
import { failureMessage } from "./failure";
import {
  groupOmlxModels,
  loadableGroups,
  residentGroups,
  type OmlxModelGroup,
} from "./omlx-groups";
import type { AttachedOmlxStatus } from "./types";
import { useNow } from "./use-active-attached";
import { refreshAttachedStatus } from "./use-attached-engines";

function groupLabel(group: OmlxModelGroup): string {
  const tags = [
    group.kind !== "chat" ? group.kind : null,
    group.pinned ? "pinned" : null,
    group.loading && !group.loaded ? "loading" : null,
  ].filter(Boolean);
  return tags.length > 0 ? `${group.id} (${tags.join(", ")})` : group.id;
}

/** "unloads in m:ss", counted down from the poll that reported it, or null when no timer applies. */
function unloadsIn(
  seconds: number | null,
  receivedAt: number | undefined,
  now: number,
): string | null {
  if (seconds === null) return null;
  const left = remainingNow(seconds, receivedAt, now);
  return left <= 0 ? "unloading" : `unloads in ${formatCountdown(left)}`;
}

function OmlxBlock({
  status,
  receivedAt,
  now,
  busy,
  run,
}: {
  receivedAt: number | undefined;
  now: number;
  status: AttachedOmlxStatus | null;
  busy: string | null;
  run: (key: string, action: () => Promise<void>, failure: string) => void;
}) {
  if (!status?.reachable) {
    return (
      <SettingsRow
        label="oMLX"
        description={
          status?.failure
            ? failureMessage("oMLX", status.failure)
            : "Not reachable. Check that the oMLX server is running and the URL is right."
        }
      />
    );
  }
  const groups = groupOmlxModels(status);
  const resident = residentGroups(groups);
  const loadable = loadableGroups(groups);
  const percent =
    status.ceilingBytes > 0
      ? Math.min(100, (status.memoryBytes / status.ceilingBytes) * 100)
      : 0;
  return (
    <>
      <SettingsRow
        label="oMLX memory"
        description={
          status.ceilingBytes > 0
            ? `${formatBytes(status.memoryBytes)} of ${formatBytes(status.ceilingBytes)} ceiling`
            : `${formatBytes(status.memoryBytes)} in use`
        }
      >
        <Progress value={percent} className="w-48" />
      </SettingsRow>
      {resident.length === 0 ? (
        <SettingsRow label="Loaded models" description="No model is loaded." />
      ) : (
        resident.map((group) => (
          <SettingsRow
            key={group.id}
            label={groupLabel(group)}
            description={[
              formatBytes(group.sizeBytes),
              group.pinned
                ? "pinned, never unloads"
                : group.loaded
                  ? unloadsIn(group.idleRemainingS, receivedAt, now)
                  : null,
            ]
              .filter(Boolean)
              .join(", ")}
          >
            <Button
              variant="outline"
              size="sm"
              disabled={busy !== null}
              onClick={() =>
                run(
                  `unload:${group.id}`,
                  () => unloadOmlxModel(group.id),
                  "Could not unload the model",
                )
              }
            >
              {busy === `unload:${group.id}` ? "Unloading" : "Unload"}
            </Button>
          </SettingsRow>
        ))
      )}
      {resident.length > 1 ? (
        <SettingsRow label="Unload everything">
          <Button
            variant="outline"
            size="sm"
            disabled={busy !== null}
            onClick={() =>
              run(
                "unload-all",
                unloadAllOmlxModels,
                "Could not unload the models",
              )
            }
          >
            {busy === "unload-all" ? "Unloading" : "Unload all"}
          </Button>
        </SettingsRow>
      ) : null}
      {loadable.map((group) => (
        <SettingsRow
          key={group.id}
          label={group.id}
          description={
            group.sizeBytes > 0
              ? `Not loaded, ${formatBytes(group.sizeBytes)}`
              : "Not loaded"
          }
        >
          <Button
            variant="outline"
            size="sm"
            disabled={busy !== null}
            onClick={() =>
              run(
                `load:${group.id}`,
                () => loadOmlxModel(group.id),
                "Could not load the model",
              )
            }
          >
            {busy === `load:${group.id}` ? "Loading" : "Load"}
          </Button>
        </SettingsRow>
      ))}
    </>
  );
}

/** Live oMLX state with the controls Studio has over it. Owner-only; hidden while the flag is off. */
export function AttachedEnginesPanel() {
  const enabled = useAttachedEnginesStore(selectAttachedEnabled);
  const status = useAttachedEnginesStore((s) => s.status);
  const [busy, setBusy] = useState<string | null>(null);
  const now = useNow(enabled);

  if (!enabled) return null;

  const run = (
    key: string,
    action: () => Promise<void>,
    failure: string,
  ) => {
    setBusy(key);
    void action()
      .catch((error) => {
        toast.error(error instanceof Error ? error.message : failure);
      })
      .then(() => refreshAttachedStatus())
      .finally(() => setBusy(null));
  };

  return (
    <SettingsSection
      title="Engines"
      description="Live state of the attached oMLX engine."
    >
      {status === null ? (
        <SettingsRow label="Engines" description="Checking engine status." />
      ) : (
        <>
          <OmlxBlock
            status={status.omlx}
            receivedAt={status.receivedAt}
            now={now}
            busy={busy}
            run={run}
          />
        </>
      )}
    </SettingsSection>
  );
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { useEffect, useState } from "react";
import { checkForkUpdates, openForkUpdateTerminal } from "./fork-update-api";
import {
  type ForkUpdateInfo,
  FORK_UPDATE_SCRIPT_HINT,
  forkBuildLabel,
  forkCanOpenUpdate,
  forkCheckedLabel,
  forkCommitRows,
  forkUpdateHeadline,
  forkUpdateKind,
  forkUpdateManualHint,
  forkUpdateNote,
} from "./fork-update-state";

/**
 * The fork's replacement for the upstream "how to update" block. The Update button opens
 * Terminal on update-fork.sh: the build needs cargo and npm from the user's shell, so the app
 * does not run it itself.
 */
export function ForkUpdateSection({ initial }: { initial: ForkUpdateInfo }) {
  const [info, setInfo] = useState<ForkUpdateInfo>(initial);
  const [checking, setChecking] = useState(false);
  const [opened, setOpened] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);

  // The shell already checks once a day at launch; this covers a stale cache on open.
  useEffect(() => {
    let canceled = false;
    checkForkUpdates(false).then((next) => {
      if (!canceled && next) setInfo(next);
    });
    return () => {
      canceled = true;
    };
  }, []);

  async function checkNow() {
    setChecking(true);
    try {
      const next = await checkForkUpdates(true);
      if (next) setInfo(next);
    } finally {
      setChecking(false);
    }
  }

  async function openUpdate() {
    setOpenError(null);
    try {
      await openForkUpdateTerminal();
      setOpened(true);
    } catch (error) {
      setOpened(false);
      setOpenError(String(error));
    }
  }

  const kind = forkUpdateKind(info);
  const { rows, more } = forkCommitRows(info);
  const checked = forkCheckedLabel(info.checked_at, Math.floor(Date.now() / 1000));
  const note = forkUpdateNote(info);
  const manual = forkUpdateManualHint(info);

  return (
    <div className="flex flex-col" data-fork-update-kind={kind}>
      <SettingsRow
        label={forkBuildLabel(info)}
        description={
          info.branch
            ? `Updates come only from github.com/beaglemoo/unsloth (${info.branch}).`
            : "Updates come only from github.com/beaglemoo/unsloth."
        }
      />
      <SettingsRow
        label={forkUpdateHeadline(info)}
        description={[checked, note].filter(Boolean).join(" · ") || undefined}
      >
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={checking}
            onClick={checkNow}
          >
            {checking ? "Checking..." : "Check for updates"}
          </Button>
          {forkCanOpenUpdate(info) ? (
            <Button size="sm" onClick={openUpdate}>
              Update
            </Button>
          ) : null}
        </div>
      </SettingsRow>
      {rows.length > 0 ? (
        <ul className="flex flex-col gap-1 pb-2 text-xs text-muted-foreground">
          {rows.map((row) => (
            <li key={row.sha} className="flex gap-2">
              <code className="font-mono text-foreground/70">{row.sha}</code>
              <span className="min-w-0 truncate">{row.message}</span>
            </li>
          ))}
          {more > 0 ? <li>and {more} more</li> : null}
        </ul>
      ) : null}
      {opened ? (
        <p className="pb-2 text-xs text-muted-foreground">
          Terminal opened. Follow the prompts there: the install quits Unsloth and replaces the
          app.
        </p>
      ) : null}
      {openError ? (
        <p className="pb-2 text-xs text-destructive">
          {openError} Run {FORK_UPDATE_SCRIPT_HINT} in a terminal instead.
        </p>
      ) : null}
      {manual ? (
        <p className="pb-2 text-xs text-muted-foreground">{manual}</p>
      ) : null}
    </div>
  );
}

/** The browser-served Studio cannot open Terminal; it names the script to run. */
export function ForkUpdateInstructions() {
  return (
    <div className="flex flex-col gap-1 py-2 text-xs text-muted-foreground">
      <p>
        This is the beaglemoo fork of Unsloth. It updates only from its own repo
        (github.com/beaglemoo/unsloth). Update with:
      </p>
      <code className="font-mono text-foreground/80">{FORK_UPDATE_SCRIPT_HINT}</code>
    </div>
  );
}

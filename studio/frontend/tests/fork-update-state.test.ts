// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import {
  FORK_UPDATE_SCRIPT_HINT,
  type ForkUpdateInfo,
  forkBuildLabel,
  forkCanOpenUpdate,
  forkCheckedLabel,
  forkCommitRows,
  forkUpdateHeadline,
  forkUpdateKind,
  forkUpdateManualHint,
  forkUpdateNote,
  isForkUpdateInfo,
} from "../src/features/fork-updates/fork-update-state.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

const SHA = "5426c052d0bfab4d8ada7c6dc55bebb92ffa219e";

function info(extra: Partial<ForkUpdateInfo> = {}): ForkUpdateInfo {
  return {
    enabled: true,
    build_sha: SHA,
    short_sha: SHA.slice(0, 9),
    branch: "feat/omlx-ds4-engines",
    checked_at: 1_000,
    ahead_by: 0,
    commits: [],
    html_url: null,
    error: null,
    can_update: true,
    ...extra,
  };
}

function commits(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    sha: `${String(i).padStart(2, "0")}abcdef0123456789`,
    message: `feat: change ${i}`,
  }));
}

test("a build shows 'Fork build <short sha>'", () => {
  assert.equal(forkBuildLabel(info()), "Fork build 5426c052d");
  assert.equal(
    forkBuildLabel(info({ short_sha: null })),
    "Fork build 5426c052d",
  );
  assert.equal(
    forkBuildLabel(info({ short_sha: null, build_sha: null })),
    "Fork build (commit unknown)",
  );
});

test("the kind follows the build commit and the answer", () => {
  assert.equal(forkUpdateKind(info({ enabled: false })), "disabled");
  assert.equal(forkUpdateKind(info({ build_sha: null })), "unbaked");
  assert.equal(forkUpdateKind(info({ ahead_by: null })), "unknown");
  assert.equal(forkUpdateKind(info({ ahead_by: 0 })), "current");
  assert.equal(forkUpdateKind(info({ ahead_by: 3 })), "available");
});

test("'N updates available' with the singular handled", () => {
  assert.equal(
    forkUpdateHeadline(info({ ahead_by: 3, commits: commits(3) })),
    "3 updates available",
  );
  assert.equal(forkUpdateHeadline(info({ ahead_by: 1 })), "1 update available");
  assert.equal(forkUpdateHeadline(info({ ahead_by: 0 })), "Up to date");
});

test("an unknown answer shows the error, or says it was not checked", () => {
  assert.equal(
    forkUpdateHeadline(info({ ahead_by: null })),
    "Not checked yet",
  );
  assert.equal(
    forkUpdateHeadline(info({ ahead_by: null, error: "GitHub did not answer within 10 s" })),
    "GitHub did not answer within 10 s",
  );
  assert.match(
    forkUpdateHeadline(info({ build_sha: null, ahead_by: null })),
    /did not record its commit/,
  );
});

test("the note under the headline is only the error that is not the headline", () => {
  assert.equal(forkUpdateNote(info({ ahead_by: null, error: "x" })), null);
  assert.equal(forkUpdateNote(info({ ahead_by: 2, error: "stale" })), "stale");
  assert.equal(forkUpdateNote(info({ ahead_by: 0 })), null);
});

test("the commit list shows 7 character shas and counts what it hides", () => {
  const { rows, more } = forkCommitRows(
    info({ ahead_by: 12, commits: commits(12) }),
    8,
  );
  assert.equal(rows.length, 8);
  assert.equal(rows[0].sha.length, 7);
  assert.equal(rows[0].message, "feat: change 0");
  assert.equal(more, 4);
});

test("an empty commit message is not a blank row, and the count survives a capped list", () => {
  const { rows, more } = forkCommitRows(
    info({ ahead_by: 300, commits: [{ sha: "abcdef0123", message: "" }] }),
  );
  assert.equal(rows[0].message, "(no message)");
  assert.equal(more, 299);
  assert.deepEqual(forkCommitRows(info()), { rows: [], more: 0 });
});

test("the Update button needs updates and a script the app can open", () => {
  assert.equal(forkCanOpenUpdate(info({ ahead_by: 2 })), true);
  assert.equal(forkCanOpenUpdate(info({ ahead_by: 2, can_update: false })), false);
  assert.equal(forkCanOpenUpdate(info({ ahead_by: 0 })), false);
  assert.equal(forkCanOpenUpdate(info({ ahead_by: null })), false);
  assert.equal(forkCanOpenUpdate(info({ enabled: false, ahead_by: 2 })), false);
});

test("without a script path the user is told what to run", () => {
  assert.equal(
    forkUpdateManualHint(info({ ahead_by: 2, can_update: false })),
    `Run ${FORK_UPDATE_SCRIPT_HINT} in a terminal.`,
  );
  assert.equal(forkUpdateManualHint(info({ ahead_by: 2 })), null);
  assert.equal(forkUpdateManualHint(info({ ahead_by: 0, can_update: false })), null);
});

test("the script hint is the fork's update script, never an upstream installer", () => {
  assert.equal(
    FORK_UPDATE_SCRIPT_HINT,
    "~/Homelab/unsloth/unsloth/studio/scripts/update-fork.sh",
  );
});

test("how long ago a check ran", () => {
  assert.equal(forkCheckedLabel(null, 5), null);
  assert.equal(forkCheckedLabel(100, 130), "Checked just now");
  assert.equal(forkCheckedLabel(0, 600), "Checked 10 min ago");
  assert.equal(forkCheckedLabel(0, 7200), "Checked 2 h ago");
  assert.equal(forkCheckedLabel(0, 3 * 86400), "Checked 3 d ago");
  assert.equal(forkCheckedLabel(500, 100), "Checked just now");
});

test("only a complete info object is accepted from the shell", () => {
  assert.equal(isForkUpdateInfo(info()), true);
  assert.equal(isForkUpdateInfo(null), false);
  assert.equal(isForkUpdateInfo({ enabled: true }), false);
  assert.equal(isForkUpdateInfo("x"), false);
});

test("the fork update code never names PyPI, upstream Unsloth or an installer", () => {
  const dir = new URL("../src/features/fork-updates/", import.meta.url);
  for (const file of readdirSync(dir)) {
    const text = readFileSync(new URL(file, dir), "utf8");
    assert.doesNotMatch(text, /pypi|unslothai|unsloth\.ai|install\.sh|install\.ps1/i, file);
  }
});

test("the loaders stay out of a browser and treat a missing command as 'not a fork build'", () => {
  const api = src("features/fork-updates/fork-update-api.ts");
  assert.match(api, /if \(!isTauri\) return null;/);
  assert.match(api, /catch \{\s*return null;/);
  assert.match(api, /"fork_update_info"/);
  assert.match(api, /"fork_update_check", \{ force \}/);
  assert.match(api, /"fork_update_open_terminal"/);
});

test("the About tab swaps the upstream instructions for the fork panel", () => {
  const about = src("features/settings/tabs/about-tab.tsx");
  assert.match(about, /loadForkUpdateInfo\(\)/);
  assert.match(about, /<ForkUpdateSection initial=\{forkInfo\} \/>/);
  // while the shell is being asked nothing renders, so the upstream curl command never flashes
  assert.match(about, /forkInfo === undefined \? null/);
  // a browser-served fork says to run the script
  assert.match(about, /data\["reason"\] === FORK_REASON/);
  assert.match(about, /<ForkUpdateInstructions \/>/);
});

test("the panel has a Check for updates button and an Update button that opens Terminal", () => {
  const section = src("features/fork-updates/fork-update-section.tsx");
  assert.match(section, /Check for updates/);
  assert.match(section, />\s*Update\s*</);
  assert.match(section, /openForkUpdateTerminal\(\)/);
  assert.match(section, /checkForkUpdates\(true\)/);
  assert.match(section, /checkForkUpdates\(false\)/);
});

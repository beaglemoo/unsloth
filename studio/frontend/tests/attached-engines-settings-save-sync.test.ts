// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { settingsSaveSyncAction } from "../src/features/attached-engines/settings-save-sync.ts";
import type { AttachedEnginesSettings } from "../src/features/attached-engines/types.ts";

function settings(
  patch: Partial<AttachedEnginesSettings> = {},
): AttachedEnginesSettings {
  return {
    enabled: true,
    omlxUrl: "http://127.0.0.1:8843",
    scanDenylist: [],
    arbitrateLocalLoads: true,
    comfyuiUrl: "http://127.0.0.1:8844",
    comfyuiPeerUrls: [],
    arbitrateComfyui: true,
    comfyuiIdleFreeS: 300,
    ...patch,
  };
}

test("a changed oMLX URL triggers a sync even though the model ids are the same", () => {
  const previous = settings();
  const saved = settings({ omlxUrl: "http://127.0.0.1:9999" });
  assert.equal(
    settingsSaveSyncAction(previous, saved, { omlxUrl: saved.omlxUrl }),
    "sync",
  );
});

test("saving the URLs unchanged, or a toggle, needs no sync", () => {
  const previous = settings();
  assert.equal(
    settingsSaveSyncAction(previous, settings(), { omlxUrl: previous.omlxUrl }),
    "none",
  );
});

test("turning the flag off removes the rows", () => {
  assert.equal(
    settingsSaveSyncAction(settings(), settings({ enabled: false }), {
      enabled: false,
    }),
    "remove",
  );
});

test("an env override that keeps the flag on is not treated as off", () => {
  assert.equal(
    settingsSaveSyncAction(settings(), settings({ enabled: true }), {
      enabled: false,
    }),
    "none",
  );
});

test("no sync while the flag is off or before settings were read", () => {
  assert.equal(
    settingsSaveSyncAction(
      settings({ enabled: false }),
      settings({ enabled: false, omlxUrl: "http://127.0.0.1:1" }),
      { omlxUrl: "http://127.0.0.1:1" },
    ),
    "none",
  );
  assert.equal(settingsSaveSyncAction(null, settings(), {}), "none");
});

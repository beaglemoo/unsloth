// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { applyProviderSync } from "../src/features/attached-engines/provider-sync.ts";

test("a provider sync that outlives its account is not applied", async () => {
  let epoch = 1;
  const applied: string[][] = [];
  let release: (providers: string[]) => void = () => undefined;
  const pending = applyProviderSync<string>({
    captureSession: () => {
      const captured = epoch;
      return () => epoch === captured;
    },
    fetchProviders: () =>
      new Promise<string[]>((resolve) => {
        release = resolve;
      }),
    apply: (providers) => applied.push(providers),
  });
  // The user switches account while the sync is still pending.
  epoch = 2;
  release(["previous-account-provider"]);
  assert.equal(await pending, false);
  assert.deepEqual(applied, []);
});

test("a provider sync for the current account is applied", async () => {
  const applied: string[][] = [];
  const result = await applyProviderSync<string>({
    captureSession: () => () => true,
    fetchProviders: async (isCurrent) => (isCurrent() ? ["p"] : []),
    apply: (providers) => applied.push(providers),
  });
  assert.equal(result, true);
  assert.deepEqual(applied, [["p"]]);
});

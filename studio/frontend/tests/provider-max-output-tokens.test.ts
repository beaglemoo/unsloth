// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(relativePath: string): string {
  return readFileSync(
    new URL(`../src/features/chat/${relativePath}`, import.meta.url),
    "utf8",
  );
}

test("all max-output cap callers pass the selected connection override", () => {
  const settings = source("chat-settings-sheet.tsx");
  const runtime = source("stores/chat-runtime-store.ts");
  const adapter = source("api/chat-adapter.ts");

  assert.match(
    settings,
    /getExternalMaxOutputTokens\([\s\S]*?activeExternalProvider\?\.maxOutputTokens/,
  );
  // the store no longer clamps: the saved value is only bounded on the wire
  assert.doesNotMatch(runtime, /getExternalMaxOutputTokens/);
  assert.match(
    adapter,
    /getExternalMaxOutputTokens\([\s\S]*?externalProvider\?\.maxOutputTokens/,
  );

  // re-gating this on the UI provider type makes the feature a no-op on the next sync
  assert.match(
    source("sync-external-providers.ts"),
    /maxOutputTokens: config\.max_output_tokens \?\? undefined,/,
  );
});

test("the connection editor exposes a bounded optional cap and warning", () => {
  const dialog = source("chat-providers-dialog.tsx");

  // Match the predicate: the LEGACY_CUSTOM_PROVIDER_TYPE line is a display-name lookup.
  assert.match(
    dialog,
    /const supportsMaxOutputTokens = supportsProviderMaxOutputTokens\(/,
  );
  assert.match(dialog, /\{supportsMaxOutputTokens \? \(/);
  assert.match(dialog, /Max Tokens limit/);
  assert.match(
    dialog,
    /Caps Max Tokens for this connection\. Never raises it past a\s+model's documented limit\. Leave blank to use that limit, or\s+32,768 for a model without one\./,
  );
  assert.match(
    dialog,
    /If the upstream provider does not support this value,\s+requests may fail\./,
  );
  // A number input sanitizes "131,072" to empty, which would silently clear the override.
  assert.match(
    dialog,
    /id="provider-max-output-tokens"\s+type="text"\s+inputMode="numeric"/,
  );
  assert.doesNotMatch(
    dialog,
    /id="provider-max-output-tokens"\s+type="number"/,
  );
  assert.match(
    dialog,
    /const floor = Math\.max\(\s*PROVIDER_MAX_OUTPUT_TOKENS_MIN,\s*getExternalMinOutputTokens\(providerType\),\s*\);\s*if \(value < floor\)/,
  );
  assert.match(dialog, /Number\.isSafeInteger\(value\)/);
  assert.match(dialog, /\/\^\\d\+\$\/\.test\(trimmed\)/);

  // Seeding the draft raw would wedge edits of a row stored below the floor.
  assert.match(
    dialog,
    /setMaxOutputTokensDraft\(\s*provider\.maxOutputTokens == null\s*\? ""\s*: Math\.max\(\s*provider\.maxOutputTokens,\s*getExternalMinOutputTokens\(provider\.providerType\),\s*\)\.toString\(\),\s*\);/,
  );
});

test("preset application clamps live Max Tokens to the active external cap", () => {
  const settings = source("chat-settings-sheet.tsx");

  assert.match(
    settings,
    // Unresolved, the cap is the 32,768 fallback and a preset would lower the value for good.
    /function applyPresetParamsWithinCurrentLimits\([\s\S]*?if \(!isExternalModel \|\| activeExternalProvider == null\) return nextParams;[\s\S]*?Math\.min\(nextParams\.maxTokens, maxTokensMax\)/,
  );
  assert.match(
    settings,
    /setParams\(applyPresetParamsWithinCurrentLimits\(p\),\s*\{\s*minPChoiceEdited: true,\s*fromPreset: true,\s*\}\)/,
  );
  assert.match(
    settings,
    /setParams\(\s*applyPresetParamsWithinCurrentLimits\(fallbackPreset\),\s*\{ minPChoiceEdited: true, fromPreset: true \},\s*\)/,
  );
});

test("a lower cap never rewrites the saved Max Tokens", () => {
  const settings = source("chat-settings-sheet.tsx");
  const runtime = source("stores/chat-runtime-store.ts");
  const adapter = source("api/chat-adapter.ts");

  // no effect writes a lowered value back through onParamsChange
  assert.doesNotMatch(settings, /resolveExternalMaxTokensClamp/);
  assert.doesNotMatch(settings, /maxTokens: clampedMaxTokens/);
  // the external switch keeps the saved value
  assert.match(runtime, /const nextMaxTokens = baseParams\.maxTokens;/);
  // the request is where the attached engine's ceiling applies
  assert.match(
    adapter,
    /max_tokens: capMaxTokensForAttached\([\s\S]*?useAttachedEnginesStore\.getState\(\)\.status/,
  );
});

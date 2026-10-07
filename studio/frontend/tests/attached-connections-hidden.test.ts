// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  ATTACHED_CONNECTIONS_LINK,
  ATTACHED_CONNECTIONS_NOTICE,
  connectionsListView,
  isAttachedConnection,
  settingsTabForConnection,
} from "../src/features/chat/attached-connections.ts";

const src = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

const omlx = { id: "attachedomlx0001", providerType: "omlx", models: ["a", "b", "c"] };
const openai = { id: "p1", providerType: "openai", models: ["gpt-x", "gpt-y"] };
const custom = { id: "p2", providerType: "custom", models: ["m"] };

test("attached rows are recognised by type or by their seeded id", () => {
  assert.equal(isAttachedConnection(omlx), true);
  assert.equal(isAttachedConnection({ id: "attachedomlx0001", providerType: "custom" }), true);
  assert.equal(isAttachedConnection(openai), false);
  assert.equal(isAttachedConnection(custom), false);
});

test("the Connections list drops attached rows and their models from the counts", () => {
  const view = connectionsListView([omlx, openai, custom]);
  assert.deepEqual(
    view.visible.map((p) => p.id),
    ["p1", "p2"],
  );
  assert.equal(view.connectionCount, 2);
  assert.equal(view.modelCount, 3);
  assert.equal(view.hasAttached, true);
});

test("without attached rows nothing is hidden and no pointer is shown", () => {
  const view = connectionsListView([openai, custom]);
  assert.equal(view.visible.length, 2);
  assert.equal(view.connectionCount, 2);
  assert.equal(view.modelCount, 3);
  assert.equal(view.hasAttached, false);
});

test("only attached rows leaves an empty list with the pointer", () => {
  const view = connectionsListView([omlx]);
  assert.equal(view.visible.length, 0);
  assert.equal(view.connectionCount, 0);
  assert.equal(view.modelCount, 0);
  assert.equal(view.hasAttached, true);
});

test("the input array is not mutated, so the picker keeps its attached rows", () => {
  const all = [omlx, openai];
  connectionsListView(all);
  assert.equal(all.length, 2);
});

test("a request to configure an attached connection lands on the Engines tab", () => {
  const all = [omlx, openai];
  assert.equal(settingsTabForConnection(all, "attachedomlx0001"), "attached-engines");
  assert.equal(settingsTabForConnection(all, "p1"), "connections");
  assert.equal(settingsTabForConnection([], "attachedomlx0001"), "attached-engines");
  assert.equal(settingsTabForConnection(all, "unknown"), "connections");
});

test("the notice wording names the engine and the tab", () => {
  assert.equal(
    `${ATTACHED_CONNECTIONS_NOTICE} ${ATTACHED_CONNECTIONS_LINK}`,
    "oMLX is managed in Settings > Engines",
  );
});

test("the Connections list renders only the filtered rows and links to the tab", () => {
  const dialog = src("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /connectionsListView\(providers\)/);
  assert.match(dialog, /listView\.visible\.map\(\(provider\) =>/);
  assert.match(dialog, /listView\.connectionCount\} connections/);
  assert.match(dialog, /openDialog\("attached-engines"\)/);
  // no list or count reads the unfiltered rows
  assert.doesNotMatch(dialog, /\{providers\.map\(/);
  assert.doesNotMatch(dialog, /\{providers\.length\} connections/);
});

test("a deep link to an attached connection never opens its edit form", () => {
  const dialog = src("features/chat/chat-providers-dialog.tsx");
  const guard = dialog.indexOf("if (isAttachedConnection(provider))");
  const edit = dialog.indexOf("void editProvider(provider);\n    onOpenProviderConsumed");
  assert.ok(guard > 0 && edit > guard);
});

test("the picker gear routes attached connections to their tab and the store still holds them", () => {
  const selector = src("features/model-picker/components/model-selector.tsx");
  assert.match(selector, /settingsTabForConnection\(/);
  assert.match(selector, /openDialog\("attached-engines"\)/);
  const tab = src("features/settings/tabs/connections-tab.tsx");
  // the tab hands the store's rows to the dialog; filtering is display-only inside it
  assert.match(tab, /providers=\{providers\}/);
  assert.doesNotMatch(tab, /isAttached/);
});

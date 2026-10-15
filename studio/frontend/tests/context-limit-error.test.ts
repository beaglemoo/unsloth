// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// isContextLimitError decides whether the chat shows "Context limit reached".
// oMLX 0.7.1.dev2 answers an overflow with 400 `context_length_exceeded` and
// "Prompt is too long: N tokens > M maximum context length"; Studio appends
// "(context_length_exceeded)". The KV-starvation sentence must stay out: it
// is worded around the markers on purpose (see stream_errors.py).

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// chat-adapter.ts pulls in the whole chat runtime, so evaluate just the
// function body (types stripped) instead of importing the module.
const source = readSrc("features/chat/api/chat-adapter.ts");
const match = source.match(
  /export function isContextLimitError\(message: string\): boolean \{([\s\S]*?)\n\}\n/,
);
assert.ok(match, "isContextLimitError not found in chat-adapter.ts");
const isContextLimitError = new Function("message", match[1]) as (
  message: string,
) => boolean;

test("recognises the oMLX 0.7.1.dev2 overflow text", () => {
  assert.equal(
    isContextLimitError(
      "Prompt is too long: 9000 tokens > 8192 maximum context length (context_length_exceeded)",
    ),
    true,
  );
  assert.equal(isContextLimitError("Error: CONTEXT_LENGTH_EXCEEDED"), true);
  assert.equal(
    isContextLimitError("9000 tokens > 8192 Maximum Context Length"),
    true,
  );
});

test("still recognises the older wording", () => {
  for (const text of [
    "request exceeds the available context size",
    "Message too long: 9000 tokens",
    "the conversation has filled the context window",
    "context shift is disabled",
    "n_ctx exceeded",
  ]) {
    assert.equal(isContextLimitError(text), true, text);
  }
});

test("does not match the KV-starvation sentence or unrelated errors", () => {
  assert.equal(
    isContextLimitError(
      "The model ran out of context space while generating. This happens when several " +
        "chats or research runs generate at the same time, because they all draw on one " +
        "shared pool of context. Try again with fewer running at once, or raise the " +
        "Context Length in Model settings.",
    ),
    false,
  );
  assert.equal(isContextLimitError(""), false);
  assert.equal(isContextLimitError("Connection refused"), false);
});

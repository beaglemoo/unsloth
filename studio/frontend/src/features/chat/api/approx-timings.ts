// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Providers that report no llama.cpp `timings` still send `usage`. The tok/s badge reads
// `serverTimings`, so build an approximate one from the usage counts and the client's own
// stream clock. It is flagged `approx` so the badge can show "~": network and tool time
// are inside the measured window.

export interface ApproxServerTimings {
  prompt_n: number;
  cache_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
  approx: true;
}

interface UsageLike {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

export function approximateServerTimings(input: {
  usage: UsageLike | undefined;
  /** Milliseconds from the stream start to the first token, when one arrived. */
  firstTokenMs: number | undefined;
  /** Milliseconds from the stream start to the end of the stream. */
  totalMs: number;
}): ApproxServerTimings | undefined {
  const completion = input.usage?.completion_tokens;
  if (typeof completion !== "number" || !(completion > 0)) return undefined;
  const firstTokenMs = Math.max(0, input.firstTokenMs ?? 0);
  const predictedMs = input.totalMs - firstTokenMs;
  if (!Number.isFinite(predictedMs) || predictedMs <= 0) return undefined;
  const prompt = Math.max(0, input.usage?.prompt_tokens ?? 0);
  const cached = Math.min(
    prompt,
    Math.max(0, input.usage?.prompt_tokens_details?.cached_tokens ?? 0),
  );
  const promptN = prompt - cached;
  const promptRate =
    firstTokenMs > 0 && promptN > 0 ? promptN / (firstTokenMs / 1000) : 0;
  return {
    prompt_n: promptN,
    cache_n: cached,
    prompt_ms: firstTokenMs,
    prompt_per_token_ms: promptN > 0 ? firstTokenMs / promptN : 0,
    prompt_per_second: promptRate,
    predicted_n: completion,
    predicted_ms: predictedMs,
    predicted_per_token_ms: predictedMs / completion,
    predicted_per_second: completion / (predictedMs / 1000),
    approx: true,
  };
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  attachedProviderKind,
  parseExternalModelId,
} from "@/features/chat/external-providers";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { useEffect, useMemo, useState } from "react";
import { type ActiveAttached, activeContextWindow } from "./attached-active";
import { useAttachedEnginesStore } from "./attached-engines-store";

/** The attached engine and model the chat has selected, or null for any other model or while the flag is off. */
export function useActiveAttached(): ActiveAttached | null {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const enabled = useAttachedEnginesStore((s) => s.settings?.enabled === true);
  const selection = parseExternalModelId(checkpoint);
  const providerId = selection?.providerId ?? null;
  const modelId = selection?.modelId ?? null;
  const providerType = useExternalProvidersStore((s) =>
    providerId === null
      ? null
      : (s.providers.find((p) => p.id === providerId)?.providerType ?? null),
  );
  return useMemo(() => {
    if (!enabled || providerId === null || modelId === null) return null;
    const kind = attachedProviderKind(providerId, providerType);
    return kind ? { kind, modelId } : null;
  }, [enabled, providerId, providerType, modelId]);
}

/** Epoch ms, re-read every `intervalMs` while `active`, so a countdown ticks between polls. */
export function useNow(active: boolean, intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [active, intervalMs]);
  return now;
}

/** The prompt cap in force for the selected oMLX model, or null for any other model. */
export function useActiveAttachedContext(): number | null {
  const active = useActiveAttached();
  const status = useAttachedEnginesStore((s) => s.status);
  return activeContextWindow(status, active);
}

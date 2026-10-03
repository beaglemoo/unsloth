// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { parseExternalModelId } from "@/features/chat/external-providers";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { isDwarfStarWarming } from "./attached-state";
import { useAttachedEnginesStore } from "./attached-engines-store";

const DWARFSTAR_PROVIDER_TYPE = "dwarfstar";

/** Shown above the composer while ds4 is cold-starting and a DwarfStar model is the selected one. Renders nothing when the flag is off. */
export function DwarfStarWarmingIndicator() {
  const warming = useAttachedEnginesStore((s) => isDwarfStarWarming(s.status));
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const providerId = parseExternalModelId(checkpoint)?.providerId ?? null;
  const selectedIsDwarfStar = useExternalProvidersStore((s) =>
    providerId === null
      ? false
      : s.providers.find((p) => p.id === providerId)?.providerType ===
        DWARFSTAR_PROVIDER_TYPE,
  );
  if (!warming || !selectedIsDwarfStar) return null;
  return (
    <div role="status" className="mb-1 px-3 text-xs text-muted-foreground">
      DwarfStar warming up. A cold start can take a couple of minutes.
    </div>
  );
}

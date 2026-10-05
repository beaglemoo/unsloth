// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The attached engine (oMLX) is a seeded connection row, managed on the
// Attached engines settings tab. Settings > Connections hides them; the model picker keeps them.

import { attachedProviderKind } from "./external-providers.ts";

type ConnectionLike = {
  id: string;
  providerType: string;
  models: readonly string[];
};

export const ATTACHED_CONNECTIONS_NOTICE =
  "oMLX is managed in";
export const ATTACHED_CONNECTIONS_LINK = "Attached engines";

export function isAttachedConnection(
  provider: Pick<ConnectionLike, "id" | "providerType">,
): boolean {
  return attachedProviderKind(provider.id, provider.providerType) !== null;
}

export type ConnectionsListView<T extends ConnectionLike> = {
  /** The rows Connections lists, edits and deletes. */
  visible: T[];
  /** "N connections" and "M models" count only these. */
  connectionCount: number;
  modelCount: number;
  /** Whether to show the pointer to the Attached engines tab. */
  hasAttached: boolean;
};

export function connectionsListView<T extends ConnectionLike>(
  providers: readonly T[],
): ConnectionsListView<T> {
  const visible = providers.filter((provider) => !isAttachedConnection(provider));
  return {
    visible,
    connectionCount: visible.length,
    modelCount: visible.reduce((count, provider) => count + provider.models.length, 0),
    hasAttached: visible.length !== providers.length,
  };
}

/** Where a request to configure one connection (the picker's gear) should land. */
export function settingsTabForConnection(
  providers: readonly Pick<ConnectionLike, "id" | "providerType">[],
  providerId: string,
): "attached-engines" | "connections" {
  const provider = providers.find((candidate) => candidate.id === providerId);
  return (provider && isAttachedConnection(provider)) ||
    attachedProviderKind(providerId) !== null
    ? "attached-engines"
    : "connections";
}

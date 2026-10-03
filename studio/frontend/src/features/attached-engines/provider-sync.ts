// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Applying a provider sync under an account guard, free of "@/" imports so a node test can drive it.

/**
 * Fetch the synced provider list and hand it to `apply`, unless the account changed while the
 * fetch was in flight: a reply for the previous account must not land in the new account's store.
 * `captureSession` is called before the fetch and returns the recheck, the way credential
 * bootstrap captures the auth session epoch. Returns whether the result was applied.
 */
export async function applyProviderSync<P>(deps: {
  captureSession: () => () => boolean;
  fetchProviders: (isCurrent: () => boolean) => Promise<P[]>;
  apply: (providers: P[]) => void;
}): Promise<boolean> {
  const isCurrent = deps.captureSession();
  const providers = await deps.fetchProviders(isCurrent);
  if (!isCurrent()) return false;
  deps.apply(providers);
  return true;
}

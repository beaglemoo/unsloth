// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AttachedContextControl } from "@/features/chat/components/attached-context-control";
import { SettingsSection } from "@/features/settings/components/settings-section";
import {
  selectAttachedEnabled,
  useAttachedEnginesStore,
} from "./attached-engines-store";
import { engineContextTargets } from "./engine-context-targets";

/**
 * Context length for every attached model in one place: the same control the chat settings show
 * for the selected model, once per oMLX chat model. Owner-only; hidden
 * while the flag is off. Each control renders nothing while its engine cannot be read.
 */
export function AttachedEnginesContextSection() {
  const enabled = useAttachedEnginesStore(selectAttachedEnabled);
  const status = useAttachedEnginesStore((s) => s.status);
  if (!enabled) return null;
  const targets = engineContextTargets(status);
  if (targets.length === 0) return null;
  return (
    <SettingsSection
      title="Context length"
      description="The prompt cap for each oMLX model. Changes wait for Apply."
    >
      <div className="flex flex-col gap-4 py-3">
        {targets.map((target) => (
          <div
            key={`${target.kind}:${target.modelId}`}
            className="flex flex-col gap-1.5"
          >
            <span
              data-settings-label={`Context length: ${target.label}`}
              className="text-sm font-medium text-foreground"
            >
              {target.label}
            </span>
            <AttachedContextControl
              active={{ kind: target.kind, modelId: target.modelId }}
            />
          </div>
        ))}
      </div>
    </SettingsSection>
  );
}

// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AttachedEnginesContextSection } from "@/features/attached-engines/engine-context-section";
import {
  AttachedEnginesPanel,
  AttachedEnginesSettingsSection,
  ComfyuiSection,
  ComfyuiWebUiSection,
  OmlxDashboardSection,
} from "@/features/attached-engines";
import { useEngineHelpers } from "@/features/attached-engines/use-engine-helpers";
import { useT } from "@/i18n";

/** The engines in one place: shared toggles, oMLX (URL, live state, load and unload, context
 *  length, dashboard), then ComfyUI (switch, state, queue, models, web UI).
 *  Owner-only (see settings-tab-visibility). While the feature flag is off only the enable toggle shows. */
export function AttachedEnginesTab() {
  const t = useT();
  useEngineHelpers();

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.tabs.attachedEngines")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.tabs.attachedEngines")}
        </h1>
        <p className="text-xs text-muted-foreground">
          oMLX and ComfyUI run beside Studio. The oMLX models appear in the chat
          model picker; the engines are managed here rather than in Connections.
        </p>
      </header>

      <AttachedEnginesSettingsSection />
      <AttachedEnginesPanel />
      <AttachedEnginesContextSection />
      <OmlxDashboardSection />
      <ComfyuiSection />
      <ComfyuiWebUiSection />
    </div>
  );
}

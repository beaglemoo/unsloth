// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { openLink } from "@/lib/open-link";
import { useState } from "react";
import { useAttachedEnginesStore } from "./attached-engines-store";
import {
  omlxDashboardView,
  readDashboardExpanded,
  writeDashboardExpanded,
} from "./omlx-dashboard-state";

/**
 * oMLX's own admin UI, framed. Nothing loads until the owner expands it (the choice is
 * remembered), and it is only framed while the flag is on and oMLX answers.
 */
export function OmlxDashboardSection() {
  const settings = useAttachedEnginesStore((s) => s.settings);
  const status = useAttachedEnginesStore((s) => s.status);
  const [expanded, setExpanded] = useState(() => readDashboardExpanded());
  // Bumping the key re-mounts the iframe, which reloads the dashboard.
  const [reloadKey, setReloadKey] = useState(0);

  const view = omlxDashboardView({
    enabled: settings?.enabled === true,
    reachable: status?.omlx?.reachable === true,
    omlxUrl: settings?.omlxUrl,
  });
  if (view.kind === "hidden") return null;

  const toggle = () => {
    const next = !expanded;
    setExpanded(next);
    writeDashboardExpanded(next);
  };

  return (
    <SettingsSection
      title="oMLX dashboard"
      description="oMLX's own admin page: models, settings and logs."
    >
      {view.kind === "unreachable" ? (
        <SettingsRow
          label="oMLX dashboard"
          description="oMLX is not running."
        />
      ) : (
        <div className="flex flex-col gap-2 py-2">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              aria-expanded={expanded}
              onClick={toggle}
            >
              {expanded ? "Hide dashboard" : "Show dashboard"}
            </Button>
            {expanded ? (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setReloadKey((key) => key + 1)}
                >
                  Reload
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void openLink(view.url)}
                >
                  Open in browser
                </Button>
              </>
            ) : null}
          </div>
          {expanded ? (
            <iframe
              key={reloadKey}
              src={view.url}
              title="oMLX dashboard"
              referrerPolicy="no-referrer"
              className="h-[70vh] min-h-[calc(520px*var(--ui-space-scale,1))] w-full rounded-lg border border-border bg-background"
            />
          ) : null}
        </div>
      )}
    </SettingsSection>
  );
}

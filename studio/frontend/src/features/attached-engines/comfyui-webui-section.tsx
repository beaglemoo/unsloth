// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { openLink } from "@/lib/open-link";
import { useState } from "react";
import { useAttachedEnginesStore } from "./attached-engines-store";
import {
  comfyuiWebUiView,
  readWebUiExpanded,
  writeWebUiExpanded,
} from "./comfyui-webui-state";

/**
 * ComfyUI's own web UI, framed. Nothing loads until the owner expands it (the choice is
 * remembered), and it is only framed while the flag is on and ComfyUI answers on a loopback URL.
 */
export function ComfyuiWebUiSection() {
  const settings = useAttachedEnginesStore((s) => s.settings);
  const status = useAttachedEnginesStore((s) => s.status);
  const [expanded, setExpanded] = useState(() => readWebUiExpanded());
  // Bumping the key re-mounts the iframe, which reloads the web UI.
  const [reloadKey, setReloadKey] = useState(0);

  const view = comfyuiWebUiView({
    enabled: settings?.enabled === true,
    reachable: status?.comfyui?.reachable === true,
    comfyuiUrl: settings?.comfyuiUrl,
  });
  if (view.kind === "hidden") return null;

  const toggle = () => {
    const next = !expanded;
    setExpanded(next);
    writeWebUiExpanded(next);
  };

  return (
    <SettingsSection
      title="ComfyUI web UI"
      description="ComfyUI's own node editor. Jobs started here do not go through Studio's memory arbiter."
    >
      {view.kind === "unreachable" ? (
        <SettingsRow label="ComfyUI web UI" description="ComfyUI is not running." />
      ) : (
        <div className="flex flex-col gap-2 py-2">
          <div className="flex items-center gap-2">
            {view.kind === "ready" ? (
              <Button
                variant="outline"
                size="sm"
                aria-expanded={expanded}
                onClick={toggle}
              >
                {expanded ? "Hide web UI" : "Show web UI"}
              </Button>
            ) : null}
            {view.kind === "link-only" || expanded ? (
              <>
                {view.kind === "ready" ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setReloadKey((key) => key + 1)}
                  >
                    Reload
                  </Button>
                ) : null}
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
          {view.kind === "ready" && expanded ? (
            <iframe
              key={reloadKey}
              src={view.url}
              title="ComfyUI web UI"
              referrerPolicy="no-referrer"
              allow="clipboard-read; clipboard-write"
              className="h-[70vh] min-h-[calc(520px*var(--ui-space-scale,1))] w-full rounded-lg border border-border bg-background"
            />
          ) : null}
        </div>
      )}
    </SettingsSection>
  );
}

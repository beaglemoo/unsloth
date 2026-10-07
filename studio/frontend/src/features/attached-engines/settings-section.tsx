// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { useEffect, useState } from "react";
import {
  loadAttachedEnginesSettings,
  updateAttachedEnginesSettings,
} from "./api";
import { useAttachedEnginesStore } from "./attached-engines-store";
import {
  setEngineHelperEnabled,
  setEngineHelpersEnabled,
  setEngineLifetime,
} from "./engine-helpers-api";
import {
  engineHelperRow,
  engineHelpersToggle,
} from "./engine-helpers-state";
import {
  engineLifetimeOf,
  engineLifetimeRow,
  enginesEnabledDescription,
  isEngineLifetime,
} from "./engine-lifetime-state";
import type { AttachedEnginesSettings } from "./types";
import { settingsSaveSyncAction } from "./settings-save-sync";
import {
  removeAttachedProviderRows,
  resyncAttachedProviderRows,
} from "./use-attached-engines";

const INFO_CLASS =
  "max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-muted-foreground";
const ERROR_CLASS =
  "max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-destructive";

/**
 * Owner-only shared settings and the oMLX settings on the Engines tab. Always rendered for the
 * owner, since the enable toggle lives here: everything else, including the Engines enabled
 * switch, the Engine lifetime control and the per-engine switch, is hidden while the feature is off.
 */
export function AttachedEnginesSettingsSection() {
  const settings = useAttachedEnginesStore((s) => s.settings);
  const setSettings = useAttachedEnginesStore((s) => s.setSettings);
  // null follows the saved value; a string is what the owner has typed since.
  const [omlxDraft, setOmlxUrl] = useState<string | null>(null);
  const omlxUrl = omlxDraft ?? settings?.omlxUrl ?? "";
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  // null: not the desktop shell, or a shell built without the engine helpers.
  const helpers = useAttachedEnginesStore((s) => s.helpers);
  const setHelpers = useAttachedEnginesStore((s) => s.setHelpers);
  const [isHelpersBusy, setIsHelpersBusy] = useState(false);
  const helpersToggle = engineHelpersToggle(helpers);
  const omlxHelperRow = engineHelperRow(helpers, "omlx");
  const lifetimeRow = engineLifetimeRow(helpers);

  const toggleHelpers = async (enabled: boolean) => {
    setIsHelpersBusy(true);
    try {
      setHelpers(await setEngineHelpersEnabled(enabled));
    } catch (helpersError) {
      setError(
        helpersError instanceof Error
          ? helpersError.message
          : "Failed to change the background engines",
      );
    } finally {
      setIsHelpersBusy(false);
    }
  };

  const toggleOmlxHelper = async (enabled: boolean) => {
    setIsHelpersBusy(true);
    try {
      setHelpers(await setEngineHelperEnabled("omlx", enabled));
    } catch (helperError) {
      setError(
        helperError instanceof Error
          ? helperError.message
          : "Failed to change the oMLX engine",
      );
    } finally {
      setIsHelpersBusy(false);
    }
  };

  const changeLifetime = async (value: string) => {
    if (!isEngineLifetime(value)) return;
    setIsHelpersBusy(true);
    try {
      setHelpers(await setEngineLifetime(value));
    } catch (lifetimeError) {
      setError(
        lifetimeError instanceof Error
          ? lifetimeError.message
          : "Failed to change the engine lifetime",
      );
    } finally {
      setIsHelpersBusy(false);
    }
  };

  useEffect(() => {
    // The root mount normally has read these already; this covers a store that is still empty.
    if (settings) return;
    let cancelled = false;
    void loadAttachedEnginesSettings()
      .then((loaded) => {
        if (!cancelled) setSettings(loaded);
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Failed to load attached engine settings",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [settings, setSettings]);

  const persist = async (update: Partial<AttachedEnginesSettings>) => {
    setIsSaving(true);
    setError(null);
    try {
      const saved = await updateAttachedEnginesSettings(update);
      setSettings(saved);
      setOmlxUrl(null);
      // Turning the flag off takes the seeded rows out of the picker; a changed URL rewrites
      // their base_url, which the poller's models_hash would never trigger.
      const action = settingsSaveSyncAction(settings, saved, update);
      if (action === "remove") await removeAttachedProviderRows();
      else if (action === "sync") await resyncAttachedProviderRows();
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Failed to update attached engine settings",
      );
    } finally {
      setIsSaving(false);
    }
  };

  const urlsDirty =
    settings !== null &&
    omlxUrl.trim() !== settings.omlxUrl;

  return (
    <SettingsSection
      title="Configuration"
      description="Show the engines in Studio, set where Studio reaches oMLX and how engines share memory with local loads."
    >
      <SettingsRow
        label="Show engines in Studio"
        description="Show oMLX in the model picker, free engine memory before a local load or training run, and manage ComfyUI here."
        below={error ? <span className={ERROR_CLASS}>{error}</span> : null}
      >
        <Switch
          checked={settings?.enabled ?? false}
          disabled={!settings || isSaving}
          onCheckedChange={(enabled) => void persist({ enabled })}
        />
      </SettingsRow>
      {settings?.enabled ? (
        <>
          {helpersToggle ? (
            <SettingsRow
              label="Engines enabled"
              description={enginesEnabledDescription(engineLifetimeOf(helpers))}
              below={
                helpersToggle.message ? (
                  <span
                    className={helpersToggle.isError ? ERROR_CLASS : INFO_CLASS}
                  >
                    {helpersToggle.message}
                  </span>
                ) : null
              }
            >
              <Switch
                checked={helpersToggle.checked}
                disabled={isHelpersBusy}
                onCheckedChange={(enabled) => void toggleHelpers(enabled)}
              />
            </SettingsRow>
          ) : null}
          {omlxHelperRow ? (
            <SettingsRow
              label="Run oMLX"
              description="Start oMLX as a background helper. Turn it off to stop oMLX while leaving ComfyUI as it is."
              below={
                omlxHelperRow.message ? (
                  <span
                    className={omlxHelperRow.isError ? ERROR_CLASS : INFO_CLASS}
                  >
                    {omlxHelperRow.message}
                  </span>
                ) : null
              }
            >
              <Switch
                checked={omlxHelperRow.checked}
                disabled={isHelpersBusy}
                onCheckedChange={(enabled) => void toggleOmlxHelper(enabled)}
              />
            </SettingsRow>
          ) : null}
          {lifetimeRow ? (
            <SettingsRow
              label="Engine lifetime"
              description="Whether the engines run only while Unsloth is open, or stay up after it quits."
              below={<span className={INFO_CLASS}>{lifetimeRow.note}</span>}
            >
              <Select
                value={lifetimeRow.value}
                disabled={isHelpersBusy}
                onValueChange={(value) => void changeLifetime(value)}
              >
                <SelectTrigger
                  aria-label="Engine lifetime"
                  className="w-80"
                  size="sm"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {lifetimeRow.options.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </SettingsRow>
          ) : null}
          <SettingsRow
            label="oMLX URL"
            description="Loopback address of the oMLX server."
          >
            <Input
              value={omlxUrl}
              aria-label="oMLX URL"
              disabled={isSaving}
              onChange={(event) => setOmlxUrl(event.target.value)}
              className="h-8 w-64"
            />
          </SettingsRow>
          {urlsDirty ? (
            <SettingsRow label="Apply engine URLs">
              <Button
                variant="outline"
                size="sm"
                disabled={isSaving}
                onClick={() =>
                  void persist({
                    omlxUrl: omlxUrl.trim(),
                  })
                }
              >
                {isSaving ? "Saving" : "Save"}
              </Button>
            </SettingsRow>
          ) : null}
          <SettingsRow
            label="Free engines for local loads"
            description="Unload oMLX models before Studio loads a local model or starts training."
          >
            <Switch
              checked={settings.arbitrateLocalLoads}
              disabled={isSaving}
              onCheckedChange={(arbitrateLocalLoads) =>
                void persist({ arbitrateLocalLoads })
              }
            />
          </SettingsRow>
        </>
      ) : null}
    </SettingsSection>
  );
}

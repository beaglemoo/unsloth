// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import {
  getDs4Context,
  getOmlxContext,
  setDs4Context,
  setOmlxContext,
} from "@/features/attached-engines/api";
import type { ActiveAttached } from "@/features/attached-engines/attached-active";
import {
  CONTEXT_STEP,
  type ContextRange,
  clampContext,
  ds4AppliedMessage,
  ds4ContextRange,
  ds4PendingNote,
  omlxContextRange,
} from "@/features/attached-engines/context-control-state";
import type {
  Ds4ContextInfo,
  OmlxContextInfo,
} from "@/features/attached-engines/types";
import { refreshAttachedStatus } from "@/features/attached-engines/use-attached-engines";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import { ParamSlider } from "../chat-settings-sheet";

type Loaded =
  | { kind: "omlx"; info: OmlxContextInfo; range: ContextRange | null }
  | { kind: "dwarfstar"; info: Ds4ContextInfo; range: ContextRange | null };

async function load(active: ActiveAttached): Promise<Loaded> {
  if (active.kind === "omlx") {
    const info = await getOmlxContext(active.modelId);
    return { kind: "omlx", info, range: omlxContextRange(info) };
  }
  const info = await getDs4Context();
  return { kind: "dwarfstar", info, range: ds4ContextRange(info) };
}

/** "Context length" for the selected oMLX or DwarfStar model. oMLX applies a per-model prompt cap
 *  at once; DwarfStar has one value for the engine and may restart. Both wait for Apply: a drag
 *  must not restart a 100 GB model. Renders nothing while the engine cannot be read. */
export function AttachedContextControl({ active }: { active: ActiveAttached }) {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [draft, setDraft] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const { kind, modelId } = active;

  useEffect(() => {
    let cancelled = false;
    setLoaded(null);
    setDraft(null);
    void load({ kind, modelId })
      .then((next) => {
        if (cancelled) return;
        setLoaded(next);
        setDraft(next.range?.current ?? null);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [kind, modelId]);

  const apply = useCallback(
    async (value: number | null) => {
      if (!loaded) return;
      setBusy(true);
      try {
        if (loaded.kind === "omlx") {
          const info = await setOmlxContext(modelId, value);
          const range = omlxContextRange(info);
          setLoaded({ kind: "omlx", info, range });
          setDraft(range?.current ?? null);
          toast.success(
            value === null
              ? "Context length reset to the default."
              : `Context length set to ${value.toLocaleString()} tokens.`,
          );
        } else if (value !== null) {
          const info = await setDs4Context(value);
          const range = ds4ContextRange(info);
          setLoaded({ kind: "dwarfstar", info, range });
          setDraft(range?.current ?? null);
          toast.info(ds4AppliedMessage(info.applied, value));
        }
        void refreshAttachedStatus();
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not set the context length",
        );
      } finally {
        setBusy(false);
      }
    },
    [loaded, modelId],
  );

  const range = loaded?.range ?? null;
  if (!loaded || !range || draft === null) return null;

  const dirty = draft !== range.current;
  const pending =
    loaded.kind === "dwarfstar" ? ds4PendingNote(loaded.info) : null;
  const info =
    loaded.kind === "omlx" ? (
      <>
        A per-model cap on the prompt, not a memory setting. It applies
        immediately with no reload. This model&apos;s native length is{" "}
        {range.max.toLocaleString()} tokens.
      </>
    ) : (
      <>
        DwarfStar has one context length for the whole engine. A change applies
        on the next start, restarts an idle engine at once, or waits for the
        current reply. Maximum {range.max.toLocaleString()} tokens.
      </>
    );

  return (
    <div className="space-y-2" data-slot="attached-context-control">
      <ParamSlider
        label="Context length"
        value={draft}
        min={range.min}
        max={range.max}
        step={CONTEXT_STEP}
        onChange={(v) => setDraft(clampContext(v, range))}
        valueSize={8}
        disabled={busy}
        info={info}
      />
      <div className="flex min-h-7 items-center justify-between gap-3">
        <span className="text-ui-11 text-muted-foreground">
          {loaded.kind === "omlx" ? "Applies immediately" : "May restart the model"}
        </span>
        <div className="flex items-center gap-2">
          {loaded.kind === "omlx" && range.hasOverride ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void apply(null)}
            >
              Reset
            </Button>
          ) : null}
          <Button
            type="button"
            size="sm"
            disabled={busy || !dirty}
            onClick={() => void apply(draft)}
          >
            {busy ? "Applying" : "Apply"}
          </Button>
        </div>
      </div>
      {pending ? (
        <p className="text-ui-11 text-amber-500">{pending}</p>
      ) : null}
    </div>
  );
}

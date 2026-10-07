// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Add01Icon,
  Delete02Icon,
  Settings01Icon,
  ShuffleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { useAttachedEnginesStore } from "@/features/attached-engines/attached-engines-store";
import { ParamSlider } from "@/features/chat";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import type { GalleryImage } from "../api";
import {
  cancelComfyGeneration,
  comfyFailureOf,
  fetchComfyLoras,
  fetchComfyProgress,
  fetchComfyTemplates,
  generateWithComfy,
} from "./api";
import { useComfyPanelStore } from "./comfyui-panel-store";
import {
  type ComfyFailure,
  type ComfyLoraRow,
  type ComfyParams,
  type ComfyProgress,
  type ComfyTemplate,
  MAX_LORAS,
  SAMPLER_OPTIONS,
  SCHEDULER_OPTIONS,
  SIZE_PRESETS,
  applyRecall,
  buildGenerateRequest,
  defaultParams,
  hasSlot,
  missingModelList,
  queueLabel,
  randomSeed,
  reconcileParams,
  snapSide,
} from "./comfyui-panel-state";
import { ComfyuiImportDialog } from "./comfyui-import-dialog";

const PROGRESS_POLL_MS = 400;

/** The run state the Images page mirrors into its own progress card and gallery strip. */
export type ComfyRunState = { running: boolean; progress: ComfyProgress | null };

function PanelField({
  label,
  hint,
  className,
  children,
}: {
  label: string;
  hint?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-center gap-1">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        {hint ? <InfoHint>{hint}</InfoHint> : null}
      </div>
      {children}
    </div>
  );
}

function OptionSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly string[];
  onChange: (value: string) => void;
}) {
  const all = value && !options.includes(value) ? [value, ...options] : options;
  return (
    <PanelField label={label}>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger aria-label={label} className="h-8 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {all.map((option) => (
            <SelectItem key={option} value={option} className="text-xs">
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </PanelField>
  );
}

/** One image side. Typing is free; the value snaps to the template's multiple on blur. */
function SideInput({
  label,
  value,
  onCommit,
  disabled,
  template,
}: {
  label: string;
  value: number;
  onCommit: (value: number) => void;
  disabled: boolean;
  template: ComfyTemplate;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const commit = () => {
    const parsed = Number(draft);
    const next = Number.isFinite(parsed) && draft.trim() !== ""
      ? snapSide(parsed, template.limits)
      : value;
    setDraft(String(next));
    if (next !== value) onCommit(next);
  };
  return (
    <PanelField
      label={label}
      hint={`${template.limits.side[0]}-${template.limits.side[1]} px, in steps of ${template.limits.multiple}.`}
      className="min-w-0 flex-1"
    >
      <Input
        type="number"
        inputMode="numeric"
        aria-label={label}
        value={draft}
        disabled={disabled}
        min={template.limits.side[0]}
        max={template.limits.side[1]}
        step={template.limits.multiple}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
        }}
      />
    </PanelField>
  );
}

function Notice({
  failure,
  onOpenSettings,
  onUnload,
}: {
  failure: ComfyFailure;
  onOpenSettings: () => void;
  onUnload?: () => void;
}) {
  const calm = failure.kind === "busy" || failure.kind === "cancelled";
  return (
    <div
      role={calm ? "status" : "alert"}
      data-comfy-notice={failure.kind}
      className={cn(
        "flex flex-col gap-2 rounded-lg border px-3 py-2.5 text-xs leading-snug",
        calm
          ? "border-border/70 bg-muted/40 text-foreground"
          : "border-destructive/40 bg-destructive/5 text-foreground",
      )}
    >
      <span>{failure.message}</span>
      {failure.openSettings || (failure.offerUnload && onUnload) ? (
        <div className="flex flex-wrap gap-2">
          {failure.openSettings ? (
            <Button type="button" size="sm" variant="secondary" onClick={onOpenSettings}>
              <HugeiconsIcon icon={Settings01Icon} className="mr-1.5 size-3.5" />
              Open Engines settings
            </Button>
          ) : null}
          {failure.offerUnload && onUnload ? (
            <Button type="button" size="sm" variant="secondary" onClick={onUnload}>
              Unload the Studio image model
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function ComfyuiCreatePanel({
  onImagesSaved,
  onRunState,
  onUnloadStudioModel,
}: {
  /** Records ComfyUI saved to the shared gallery, to be merged into the page's strip. */
  onImagesSaved: (images: GalleryImage[]) => void;
  onRunState: (state: ComfyRunState) => void;
  /** Offered when admission refuses because a Studio image model holds memory. */
  onUnloadStudioModel?: () => void;
}) {
  const templateId = useComfyPanelStore((s) => s.templateId);
  const params = useComfyPanelStore((s) => s.params);
  const pendingRecall = useComfyPanelStore((s) => s.pendingRecall);
  const comfyStatus = useAttachedEnginesStore((s) => s.status?.comfyui ?? null);
  const reachable = comfyStatus?.reachable === true;

  const [templates, setTemplates] = useState<ComfyTemplate[]>([]);
  const [loaded, setLoaded] = useState<"loading" | "ready" | "error">("loading");
  const [loras, setLoras] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [progress, setProgress] = useState<ComfyProgress | null>(null);
  const [notice, setNotice] = useState<ComfyFailure | null>(null);
  const [lastSeed, setLastSeed] = useState<number | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const mounted = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const loadTicket = useRef(0);

  const template = useMemo(
    () => templates.find((t) => t.id === templateId) ?? null,
    [templates, templateId],
  );

  const stopPolling = useCallback(() => {
    if (pollTimer.current) clearInterval(pollTimer.current);
    pollTimer.current = null;
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stopPolling();
      onRunState({ running: false, progress: null });
    };
    // Mount and unmount only; onRunState is a stable page callback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadTemplates = useCallback(async () => {
    const ticket = ++loadTicket.current;
    try {
      const result = await fetchComfyTemplates();
      if (!mounted.current || ticket !== loadTicket.current) return;
      setTemplates(result.templates);
      setLoaded("ready");
    } catch (error) {
      if (!mounted.current || ticket !== loadTicket.current) return;
      setLoaded("error");
      setNotice(comfyFailureOf(error, "Could not list the ComfyUI templates"));
    }
  }, []);

  // The list is read on mount and again whenever ComfyUI starts or stops answering: the missing
  // model check only exists while it is up.
  useEffect(() => {
    void loadTemplates();
  }, [loadTemplates, reachable]);

  // Settle the selected template and parameters against the list: a pending recall wins, then the
  // remembered selection, then the first template.
  useEffect(() => {
    if (templates.length === 0) return;
    const store = useComfyPanelStore.getState();
    const recall = store.pendingRecall;
    if (recall) {
      const applied = applyRecall(recall, templates);
      store.clearRecall();
      if (applied) {
        store.setTemplateId(applied.template.id);
        store.setParams(applied.params);
        if (!applied.found) {
          setNotice({
            kind: "error",
            message: `The template ${recall.templateId} is no longer available. Showing ${applied.template.name} with the recalled settings.`,
          });
        }
      }
      return;
    }
    const current = templates.find((t) => t.id === store.templateId);
    if (current) {
      if (store.params) store.setParams(reconcileParams(store.params, current));
      else store.setParams(defaultParams(current));
      return;
    }
    const first = templates[0];
    store.setTemplateId(first.id);
    store.setParams({
      ...defaultParams(first),
      prompt: store.params?.prompt ?? "",
      negativePrompt: store.params?.negativePrompt ?? "",
      seed: store.params?.seed ?? "",
    });
  }, [templates, pendingRecall]);

  useEffect(() => {
    if (!template?.supportsLora) return;
    let cancelled = false;
    void fetchComfyLoras().then((files) => {
      if (!cancelled && mounted.current) setLoras(files);
    });
    return () => {
      cancelled = true;
    };
  }, [template?.id, template?.supportsLora, reachable]);

  const patch = useCallback((change: Partial<ComfyParams>) => {
    useComfyPanelStore.getState().patchParams(change);
  }, []);

  const applyTemplate = (next: ComfyTemplate) => {
    const store = useComfyPanelStore.getState();
    if (!store.params) return;
    store.setTemplateId(next.id);
    store.setParams({
      ...defaultParams(next),
      prompt: store.params.prompt,
      negativePrompt: store.params.negativePrompt,
      seed: store.params.seed,
    });
    setNotice(null);
  };

  const selectTemplate = (id: string) => {
    const next = templates.find((t) => t.id === id);
    if (next) applyTemplate(next);
  };

  const openSettings = () => useSettingsDialogStore.getState().openDialog("attached-engines");

  const startPolling = () => {
    stopPolling();
    let inFlight = false;
    pollTimer.current = setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      void fetchComfyProgress()
        .then((next) => {
          if (!mounted.current || !pollTimer.current) return;
          const live = next.active ? next : null;
          setProgress(live);
          // A queued job has no steps yet: the page card keeps "Starting" and this panel shows the slot.
          const stepping = live && (live.queue_position ?? 0) <= 0 ? live : null;
          onRunState({ running: true, progress: stepping });
        })
        .catch(() => undefined)
        .finally(() => {
          inFlight = false;
        });
    }, PROGRESS_POLL_MS);
  };

  const generate = async () => {
    if (!template || !params || running) return;
    const built = buildGenerateRequest(template, params);
    if (!built.ok) {
      toast.error(built.error);
      return;
    }
    setNotice(null);
    setStopping(false);
    setProgress(null);
    setRunning(true);
    onRunState({ running: true, progress: null });
    startPolling();
    try {
      const result = await generateWithComfy(built.body);
      if (!mounted.current) return;
      setLastSeed(result.seed);
      onImagesSaved(result.images);
      for (const action of result.actions) toast.info(action);
    } catch (error) {
      if (!mounted.current) return;
      const failure = comfyFailureOf(error, "ComfyUI generation failed");
      setNotice(failure);
      if (failure.kind === "missing" || failure.reloadTemplates) void loadTemplates();
    } finally {
      stopPolling();
      if (mounted.current) {
        setRunning(false);
        setStopping(false);
        setProgress(null);
      }
      onRunState({ running: false, progress: null });
    }
  };

  const stop = async () => {
    if (stopping) return;
    setStopping(true);
    try {
      const cancelled = await cancelComfyGeneration();
      if (!cancelled && mounted.current) setStopping(false);
    } catch {
      if (mounted.current) setStopping(false);
      toast.error("Could not reach ComfyUI to stop this generation; it is still running");
    }
  };

  const updateLora = (index: number, change: Partial<ComfyLoraRow>) => {
    if (!params) return;
    patch({ loras: params.loras.map((row, i) => (i === index ? { ...row, ...change } : row)) });
  };

  const missing = template ? missingModelList(template) : [];
  const waiting = progress ? queueLabel(progress.queue_position) : null;
  const sizeLocked = running;
  const comfyDown = !reachable && comfyStatus?.helperWanted === true;

  let body: ReactNode;
  if (loaded === "loading" && templates.length === 0) {
    body = (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Spinner className="size-4" /> Loading ComfyUI templates…
      </div>
    );
  } else if (!template || !params) {
    body = (
      <div className="flex flex-col gap-3">
        {notice ? (
          <Notice failure={notice} onOpenSettings={openSettings} onUnload={onUnloadStudioModel} />
        ) : (
          <p className="text-xs text-muted-foreground">
            No ComfyUI templates are available.
          </p>
        )}
        <Button type="button" size="sm" variant="secondary" onClick={() => void loadTemplates()}>
          Reload templates
        </Button>
      </div>
    );
  } else {
    body = (
      <>
        <PanelField
          label="Template"
          hint="A ComfyUI graph with its parameters exposed. Qwen-Image 2.1 ships with Studio; import your own API-format graph."
        >
          <div className="flex items-center gap-2">
            <Select value={template.id} onValueChange={selectTemplate} disabled={running}>
              <SelectTrigger aria-label="ComfyUI template" className="h-9 min-w-0 flex-1 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {templates.map((t) => (
                  <SelectItem key={t.id} value={t.id} className="text-xs">
                    {t.name}
                    {t.source === "user" ? " (imported)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              disabled={running}
              onClick={() => setImportOpen(true)}
            >
              Import graph
            </Button>
          </div>
        </PanelField>

        {comfyDown ? (
          <div
            role="status"
            data-comfy-notice="not_running"
            className="rounded-lg border border-border/70 bg-muted/40 px-3 py-2.5 text-xs leading-snug"
          >
            {comfyStatus?.failure
              ? `ComfyUI is failing to start: ${comfyStatus.failure.reason}. See Settings > Engines.`
              : "ComfyUI is starting. Generation works once it answers; this can take a few seconds."}
          </div>
        ) : null}

        {missing.length > 0 ? (
          <div
            role="alert"
            data-comfy-notice="missing"
            className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2.5 text-xs leading-snug"
          >
            ComfyUI is missing model files this template needs:{" "}
            <span className="font-mono">{missing.join(", ")}</span>. Put them in one of its model
            folders and reload.
          </div>
        ) : null}

        {notice ? (
          <Notice failure={notice} onOpenSettings={openSettings} onUnload={onUnloadStudioModel} />
        ) : null}

        <PanelField label="Prompt">
          <Textarea
            data-type-to-activate="prompt"
            rows={4}
            className="image-prompt-box min-h-32 rounded-lg"
            placeholder="Describe the image"
            value={params.prompt}
            onChange={(e) => patch({ prompt: e.target.value })}
          />
        </PanelField>

        {hasSlot(template, "negative_prompt") ? (
          <PanelField
            label="Negative prompt"
            hint="What to steer the image away from. Models run at low CFG often ignore it."
          >
            <Textarea
              rows={2}
              className="image-prompt-box rounded-lg"
              value={params.negativePrompt}
              onChange={(e) => patch({ negativePrompt: e.target.value })}
            />
          </PanelField>
        ) : null}

        {hasSlot(template, "width") || hasSlot(template, "height") ? (
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Size presets">
              {SIZE_PRESETS.map(([label, width, height]) => {
                const w = snapSide(width, template.limits);
                const h = snapSide(height, template.limits);
                const active = params.width === w && params.height === h;
                return (
                  <Button
                    key={label}
                    type="button"
                    size="sm"
                    variant={active ? "default" : "secondary"}
                    className="h-7 px-2.5 text-xs"
                    disabled={sizeLocked}
                    aria-pressed={active}
                    onClick={() => patch({ width: w, height: h })}
                  >
                    {label}
                  </Button>
                );
              })}
            </div>
            <div className="flex gap-2">
              <SideInput
                label="Width"
                value={params.width}
                disabled={sizeLocked}
                template={template}
                onCommit={(width) => patch({ width })}
              />
              <SideInput
                label="Height"
                value={params.height}
                disabled={sizeLocked}
                template={template}
                onCommit={(height) => patch({ height })}
              />
            </div>
          </div>
        ) : null}

        {hasSlot(template, "steps") ? (
          <div className="pt-1">
            <ParamSlider
              inline={true}
              label="Steps"
              info="Denoising steps. More take longer and do not always look better."
              value={params.steps}
              min={template.limits.steps[0]}
              max={template.limits.steps[1]}
              step={1}
              onChange={(steps) => patch({ steps })}
              disabled={running}
            />
          </div>
        ) : null}
        {hasSlot(template, "cfg") ? (
          <ParamSlider
            inline={true}
            label="CFG"
            info="How strongly the model follows the prompt. Distilled models want a low value."
            value={params.cfg}
            min={template.limits.cfg[0]}
            max={template.limits.cfg[1]}
            step={0.5}
            onChange={(cfg) => patch({ cfg })}
            disabled={running}
          />
        ) : null}
        {hasSlot(template, "batch_size") ? (
          <ParamSlider
            inline={true}
            label="Batch size"
            info="Images made at once from the same seed."
            value={params.batchSize}
            min={template.limits.batchSize[0]}
            max={template.limits.batchSize[1]}
            step={1}
            onChange={(batchSize) => patch({ batchSize })}
            disabled={running}
          />
        ) : null}

        {hasSlot(template, "sampler") || hasSlot(template, "scheduler") ? (
          <div className="flex gap-2">
            {hasSlot(template, "sampler") ? (
              <div className="min-w-0 flex-1">
                <OptionSelect
                  label="Sampler"
                  value={params.sampler}
                  options={SAMPLER_OPTIONS}
                  onChange={(sampler) => patch({ sampler })}
                />
              </div>
            ) : null}
            {hasSlot(template, "scheduler") ? (
              <div className="min-w-0 flex-1">
                <OptionSelect
                  label="Scheduler"
                  value={params.scheduler}
                  options={SCHEDULER_OPTIONS}
                  onChange={(scheduler) => patch({ scheduler })}
                />
              </div>
            ) : null}
          </div>
        ) : null}

        {hasSlot(template, "seed") ? (
          <PanelField
            label="Seed"
            hint="Leave empty for a fresh random seed each run."
            className="pt-1"
          >
            <div className="flex gap-2">
              <Input
                placeholder="Random if empty"
                inputMode="numeric"
                aria-label="Seed"
                value={params.seed}
                onChange={(e) => patch({ seed: e.target.value })}
              />
              <Button
                type="button"
                size="icon"
                variant="secondary"
                aria-label="Randomize seed"
                disabled={running}
                onClick={() => patch({ seed: randomSeed() })}
              >
                <HugeiconsIcon icon={ShuffleIcon} className="size-4" />
              </Button>
            </div>
            {lastSeed !== null ? (
              <button
                type="button"
                className="self-start text-ui-11 text-muted-foreground underline-offset-2 hover:underline"
                onClick={() => patch({ seed: String(lastSeed) })}
              >
                Last seed {lastSeed}: use again
              </button>
            ) : null}
          </PanelField>
        ) : null}

        {template.supportsLora ? (
          <PanelField
            label="LoRAs"
            hint="Adapter files from ComfyUI's loras folder, applied on top of the model. Strength 1.0 is full effect."
          >
            <div className="space-y-2">
              {params.loras.map((row, index) => (
                <div key={index} className="flex items-center gap-2">
                  {loras.length > 0 ? (
                    <Select value={row.name} onValueChange={(name) => updateLora(index, { name })}>
                      <SelectTrigger aria-label={`LoRA ${index + 1}`} className="h-8 min-w-0 flex-1 text-xs">
                        <SelectValue placeholder="Choose a LoRA" />
                      </SelectTrigger>
                      <SelectContent>
                        {(loras.includes(row.name) || !row.name ? loras : [row.name, ...loras]).map(
                          (file) => (
                            <SelectItem key={file} value={file} className="text-xs">
                              {file}
                            </SelectItem>
                          ),
                        )}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Input
                      aria-label={`LoRA ${index + 1}`}
                      className="h-8 min-w-0 flex-1 text-xs"
                      placeholder="File name"
                      value={row.name}
                      onChange={(e) => updateLora(index, { name: e.target.value })}
                    />
                  )}
                  <Input
                    type="number"
                    aria-label={`LoRA ${index + 1} strength`}
                    className="h-8 w-20 text-xs"
                    min={-4}
                    max={4}
                    step={0.05}
                    value={row.strength}
                    onChange={(e) => {
                      const strength = Number(e.target.value);
                      if (Number.isFinite(strength)) updateLora(index, { strength });
                    }}
                  />
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    aria-label={`Remove LoRA ${index + 1}`}
                    className="size-8 shrink-0"
                    onClick={() => patch({ loras: params.loras.filter((_, i) => i !== index) })}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
                </div>
              ))}
              {loras.length === 0 && reachable ? (
                <p className="text-ui-11 text-muted-foreground">
                  ComfyUI lists no LoRA files. Type a file name from its loras folder.
                </p>
              ) : null}
              <Button
                type="button"
                size="sm"
                variant="secondary"
                disabled={params.loras.length >= MAX_LORAS || running}
                onClick={() =>
                  patch({ loras: [...params.loras, { name: loras[0] ?? "", strength: 1 }] })
                }
              >
                <HugeiconsIcon icon={Add01Icon} className="mr-1.5 size-3.5" />
                Add LoRA
              </Button>
            </div>
          </PanelField>
        ) : null}

      </>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col" data-comfyui-panel="">
      <div className="flex flex-1 flex-col gap-4">{body}</div>
      <div className="sticky bottom-0 z-10 -mx-1 mt-4 flex flex-col items-center gap-2 bg-gradient-to-t from-background via-background to-transparent px-1 pt-3">
        {waiting ? (
          <p role="status" className="text-xs text-muted-foreground">
            {waiting}
          </p>
        ) : null}
        {running ? (
          <Button
            className="h-11 px-8 hover:bg-muted dark:hover:bg-muted"
            variant="outline"
            onClick={() => void stop()}
          >
            <Spinner className="mr-2 size-4" />
            {stopping ? "Stopping…" : "Stop"}
          </Button>
        ) : (
          <Button
            className="h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
            onClick={() => void generate()}
            disabled={!template || !params || !params.prompt.trim() || missing.length > 0}
          >
            Generate
          </Button>
        )}
      </div>
      <ComfyuiImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        templates={templates}
        onImported={(imported) => {
          setTemplates((prev) => [...prev.filter((t) => t.id !== imported.id), imported]);
          applyTemplate(imported);
          void loadTemplates();
        }}
        onDeleted={() => void loadTemplates()}
      />
    </div>
  );
}

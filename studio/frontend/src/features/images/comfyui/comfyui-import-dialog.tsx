// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { comfyFailureOf, deleteComfyTemplate, importComfyGraph } from "./api";
import { type ComfyTemplate, parseGraphJson } from "./comfyui-panel-state";

/** Import a ComfyUI API-format graph as a template, and delete the ones imported before. */
export function ComfyuiImportDialog({
  open,
  onOpenChange,
  templates,
  onImported,
  onDeleted,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  templates: ComfyTemplate[];
  onImported: (template: ComfyTemplate) => void;
  onDeleted: (id: string) => void;
}) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const imported = templates.filter((t) => t.source === "user");

  const pickFile = async (file: File | undefined) => {
    if (!file) return;
    setText(await file.text());
    if (!name.trim()) setName(file.name.replace(/\.json$/i, "").slice(0, 80));
  };

  const submit = async () => {
    const parsed = parseGraphJson(text);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    if (!name.trim()) {
      setError("Give the template a name");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const template = await importComfyGraph(name.trim(), parsed.graph);
      if (template) onImported(template);
      toast.success(`Imported ${name.trim()}`);
      setName("");
      setText("");
      onOpenChange(false);
    } catch (failure) {
      setError(comfyFailureOf(failure, "Could not import the graph").message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (template: ComfyTemplate) => {
    try {
      await deleteComfyTemplate(template.id);
      onDeleted(template.id);
    } catch (failure) {
      toast.error(comfyFailureOf(failure, "Could not delete the template").message);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Import a ComfyUI graph</DialogTitle>
          <DialogDescription>
            Export the graph from ComfyUI with "Save (API Format)", then paste or choose the JSON.
            Studio finds the prompt, seed, steps, size and sampler inputs and offers them as controls.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <Input
            aria-label="Template name"
            placeholder="Template name"
            value={name}
            maxLength={80}
            onChange={(e) => setName(e.target.value)}
          />
          <Textarea
            aria-label="Graph JSON"
            rows={8}
            className="font-mono text-xs"
            placeholder='{"1": {"class_type": "...", "inputs": {...}}}'
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <input
            ref={fileInput}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => {
              void pickFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          {error ? (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="secondary" onClick={() => fileInput.current?.click()}>
              Choose file
            </Button>
            <Button type="button" disabled={busy || !text.trim()} onClick={() => void submit()}>
              {busy ? <Spinner className="mr-2 size-4" /> : null}
              Import
            </Button>
          </div>
          {imported.length > 0 ? (
            <div className="flex flex-col gap-1.5 border-t border-border/60 pt-3">
              <span className="text-xs font-medium text-muted-foreground">Imported templates</span>
              {imported.map((template) => (
                <div key={template.id} className="flex items-center justify-between gap-2 text-xs">
                  <span className="min-w-0 truncate">{template.name}</span>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    className="size-7 shrink-0"
                    aria-label={`Delete ${template.name}`}
                    onClick={() => void remove(template)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                  </Button>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}

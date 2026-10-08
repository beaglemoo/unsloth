// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ComfyInput, ComfyParams, ComfyRecall } from "./comfyui-panel-state";

/** A gallery image handed to the panel by "Use as input", waiting for the panel to place it. */
export type ComfyPendingInput = {
  galleryId: string;
  url: string;
  width: number;
  height: number;
};

function revokePreview(input: ComfyInput | null | undefined) {
  if (input?.kind === "gallery") URL.revokeObjectURL(input.previewUrl);
}

/** The panel's form, kept outside the component so it survives switching engines and so a gallery
 *  recall can hand over a template and its parameters before the panel has mounted. */
interface ComfyPanelState {
  templateId: string | null;
  /** Null until the first template loads and gives the defaults. */
  params: ComfyParams | null;
  /** A recall waiting for the panel to apply it against the template list. */
  pendingRecall: ComfyRecall | null;
  /** Input images by slot name, in memory only (never persisted). Both shipped image templates use
   *  `image`, so switching between them keeps the picture. */
  inputs: Record<string, ComfyInput | null>;
  /** "Use as input" waiting for the panel to place it. */
  pendingInput: ComfyPendingInput | null;
  setTemplateId: (id: string | null) => void;
  setParams: (params: ComfyParams | null) => void;
  patchParams: (patch: Partial<ComfyParams>) => void;
  requestRecall: (recall: ComfyRecall) => void;
  clearRecall: () => void;
  setInput: (slot: string, input: ComfyInput | null) => void;
  clearInputs: () => void;
  requestInput: (input: ComfyPendingInput) => void;
  clearPendingInput: () => void;
}

export const useComfyPanelStore = create<ComfyPanelState>((set, get) => ({
  templateId: null,
  params: null,
  pendingRecall: null,
  inputs: {},
  pendingInput: null,
  setTemplateId: (templateId) => set({ templateId }),
  setParams: (params) => set({ params }),
  patchParams: (patch) =>
    set((state) => (state.params ? { params: { ...state.params, ...patch } } : state)),
  requestRecall: (pendingRecall) => set({ pendingRecall }),
  clearRecall: () => set({ pendingRecall: null }),
  setInput: (slot, input) => {
    const previous = get().inputs[slot];
    if (previous !== input) revokePreview(previous);
    set((state) => ({ inputs: { ...state.inputs, [slot]: input } }));
  },
  clearInputs: () => {
    for (const input of Object.values(get().inputs)) revokePreview(input);
    set({ inputs: {} });
  },
  requestInput: (pendingInput) => set({ pendingInput }),
  clearPendingInput: () => set({ pendingInput: null }),
}));

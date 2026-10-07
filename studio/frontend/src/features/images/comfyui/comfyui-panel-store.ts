// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ComfyParams, ComfyRecall } from "./comfyui-panel-state";

/** The panel's form, kept outside the component so it survives switching engines and so a gallery
 *  recall can hand over a template and its parameters before the panel has mounted. */
interface ComfyPanelState {
  templateId: string | null;
  /** Null until the first template loads and gives the defaults. */
  params: ComfyParams | null;
  /** A recall waiting for the panel to apply it against the template list. */
  pendingRecall: ComfyRecall | null;
  setTemplateId: (id: string | null) => void;
  setParams: (params: ComfyParams | null) => void;
  patchParams: (patch: Partial<ComfyParams>) => void;
  requestRecall: (recall: ComfyRecall) => void;
  clearRecall: () => void;
}

export const useComfyPanelStore = create<ComfyPanelState>((set) => ({
  templateId: null,
  params: null,
  pendingRecall: null,
  setTemplateId: (templateId) => set({ templateId }),
  setParams: (params) => set({ params }),
  patchParams: (patch) =>
    set((state) => (state.params ? { params: { ...state.params, ...patch } } : state)),
  requestRecall: (pendingRecall) => set({ pendingRecall }),
  clearRecall: () => set({ pendingRecall: null }),
}));

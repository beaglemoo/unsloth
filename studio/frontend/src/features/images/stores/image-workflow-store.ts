// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type ImageEngine,
  persistEngine,
  readStoredEngine,
} from "../comfyui/engine-choice";
import type { WorkflowId } from "../workflows";

export interface AdapterDeployRequest {
  baseRepo: string;
  family: string;
  catalogPath: string;
  trigger: string;
}

/** `supported` null means nothing is loaded, so every workflow stays selectable. */
interface ImageWorkflowState {
  workflow: WorkflowId;
  pageMode: "create" | "train";
  supported: WorkflowId[] | null;
  /** Which engine Create generates with; remembered. The page only honours "comfyui" while ComfyUI is available. */
  engine: ImageEngine;
  setEngine: (engine: ImageEngine) => void;
  navExpanded: boolean;
  setNavExpanded: (expanded: boolean) => void;
  setWorkflow: (id: WorkflowId) => void;
  setPageMode: (mode: "create" | "train") => void;
  setSupported: (ids: WorkflowId[] | null) => void;
  /** Video's Train tab hands Images its adapters: Video loads none. */
  pendingDeploy: AdapterDeployRequest | null;
  requestDeploy: (request: AdapterDeployRequest) => void;
  takeDeploy: () => AdapterDeployRequest | null;
  trainingCompletions: number;
  noteTrainingComplete: () => void;
}

export const useImageWorkflowStore = create<ImageWorkflowState>((set, get) => ({
  workflow: "create",
  pageMode: "create",
  supported: null,
  navExpanded: false,
  engine: readStoredEngine(),
  setEngine: (engine) => {
    persistEngine(engine);
    set({ engine });
  },
  setNavExpanded: (navExpanded) => set({ navExpanded }),
  setWorkflow: (workflow) => set({ workflow, pageMode: "create" }),
  setPageMode: (pageMode) => set({ pageMode }),
  setSupported: (supported) => set({ supported }),
  pendingDeploy: null,
  requestDeploy: (pendingDeploy) => set({ pendingDeploy }),
  takeDeploy: () => {
    const request = get().pendingDeploy;
    if (request) set({ pendingDeploy: null });
    return request;
  },
  trainingCompletions: 0,
  noteTrainingComplete: () =>
    set((s) => ({ trainingCompletions: s.trainingCompletions + 1 })),
}));

export function isWorkflowEnabled(
  id: WorkflowId,
  supported: WorkflowId[] | null,
): boolean {
  return supported === null || supported.includes(id);
}

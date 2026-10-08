// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useComfyPanelStore } from "./comfyui-panel-store";

/** The slice of a gallery record "Use as input" reads (api.ts GalleryImage satisfies it). */
export type UseAsInputSource = { id: string; url: string; width: number; height: number };

/** Hand a gallery image to the ComfyUI panel, which places it on a template that takes an input. */
export function requestComfyInput(image: UseAsInputSource): void {
  useComfyPanelStore.getState().requestInput({
    galleryId: image.id,
    url: image.url,
    width: image.width,
    height: image.height,
  });
}

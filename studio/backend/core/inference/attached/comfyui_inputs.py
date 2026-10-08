# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Input images for ComfyUI image jobs: validation, preparation and the temp-file lifecycle.

A request names an image per template slot, either inline (a data URL or base64) or by gallery id.
``plan_inputs`` checks the slots against the template without any I/O; ``load_png`` turns one
request input into PNG bytes ready to upload (EXIF-upright, alpha flattened over white, long side
capped); ``remove_inputs`` and ``sweep_stale`` delete the uploads Studio put in ComfyUI's temp
folder (``<data_dir>/temp/unsloth-inputs``). Nothing outside that folder is ever touched, and only
names Studio generated (``<32 hex>-image[_N].png``).

Deliberately independent of ``core.inference.diffusion``: PIL and the gallery are imported lazily.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.inference.attached.comfyui_graphs import ComfyParamError, Template
from loggers import get_logger

logger = get_logger(__name__)

SUBFOLDER = "unsloth-inputs"
NAME_RE = re.compile(r"^[0-9a-f]{32}-image(_[2-4])?\.png$")
MAX_SIDE_IN = 4096
MAX_SIDE_OUT = 2048
DEFAULT_DATA_DIR = "~/.unsloth/engines/comfyui-data"


@dataclass(frozen = True)
class InputPlan:
    slot: str
    # The name Studio uploads under, inside ``SUBFOLDER``, and the value the graph's LoadImage gets.
    filename: str
    annotated: str
    data: Optional[str] = None
    gallery_id: Optional[str] = None


def plan_inputs(template: Template, input_images: Any) -> dict[str, InputPlan]:
    """One ``InputPlan`` per image slot of ``template``; raises ``ComfyParamError`` for a missing or unknown slot.

    Synchronous and free of I/O, so a job can fail on it before it touches the queue.
    """
    if input_images is None:
        input_images = {}
    if not isinstance(input_images, dict):
        raise ComfyParamError("input_images must be an object.")
    for slot in input_images:
        if slot not in template.image_slots:
            raise ComfyParamError(f"This template has no input image {slot!r}.")
    plans: dict[str, InputPlan] = {}
    for slot, spec in template.image_slots.items():
        entry = input_images.get(slot)
        if not isinstance(entry, dict):
            raise ComfyParamError(f"This template needs an input image ({spec['label']}).")
        data, gallery_id = entry.get("data") or None, entry.get("gallery_id") or None
        if (data is None) == (gallery_id is None) or not isinstance(data or gallery_id, str):
            raise ComfyParamError(f"Input image {slot!r} needs either data or a gallery_id.")
        filename = f"{uuid.uuid4().hex}-{slot}.png"
        plans[slot] = InputPlan(
            slot = slot,
            filename = filename,
            annotated = f"{SUBFOLDER}/{filename} [temp]",
            data = data,
            gallery_id = gallery_id,
        )
    return plans


# ------------------------------------------------------------------ preparation


def _raw_bytes(plan: InputPlan) -> bytes:
    if plan.gallery_id is not None:
        from core.inference import image_gallery

        path = image_gallery.owned_image_path(plan.gallery_id)
        if path is None:
            raise ComfyParamError(f"Gallery image {plan.gallery_id} no longer exists.")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ComfyParamError(f"Gallery image {plan.gallery_id} could not be read: {exc.strerror or exc}") from exc
    text = (plan.data or "").strip()
    if text.startswith("data:"):
        header, comma, text = text.partition(",")
        if not comma or ";base64" not in header:
            raise ComfyParamError("The input image is not a base64 data URL.")
    try:
        return base64.b64decode("".join(text.split()), validate = True)
    except (binascii.Error, ValueError) as exc:
        raise ComfyParamError("The input image is not valid base64.") from exc


def _flatten(image: Any) -> Any:
    """RGB with any transparency composited over white (LoadImage drops alpha anyway)."""
    from PIL import Image

    if image.mode in ("RGBA", "LA", "PA", "La", "RGBa", "P") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask = rgba.getchannel("A"))
        return flat
    return image.convert("RGB")


def load_png(plan: InputPlan) -> bytes:
    """The prepared input as PNG bytes. Blocking: run it in a thread. Raises ``ComfyParamError``."""
    raw = _raw_bytes(plan)
    try:
        from PIL import Image

        from core.inference.image_orientation import exif_upright

        with Image.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            if max(width, height) > MAX_SIDE_IN:
                raise ComfyParamError(
                    f"The input image is {width} x {height}; the longest side can be at most {MAX_SIDE_IN} px."
                )
            opened.load()
            image = _flatten(exif_upright(opened))
        longest = max(image.size)
        if longest > MAX_SIDE_OUT:
            scale = MAX_SIDE_OUT / longest
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS
            )
        out = io.BytesIO()
        image.save(out, format = "PNG", compress_level = 1)
        return out.getvalue()
    except ComfyParamError:
        raise
    except Exception as exc:  # noqa: BLE001 - PIL raises many types for a bad file
        raise ComfyParamError(f"Could not read the input image: {exc}") from exc


# -------------------------------------------------------------------- cleanup


def temp_inputs_dir() -> Optional[Path]:
    """``<ComfyUI data dir>/temp/unsloth-inputs``; the data dir is ``[comfyui] data_dir`` from engines.toml
    (found like ``comfyui_jobs.configured_model_dirs`` does), default ``~/.unsloth/engines/comfyui-data``."""
    try:
        import tomllib

        from core.inference.attached.failures import engines_home

        data_dir: Any = None
        try:
            path = Path(os.environ.get("UNSLOTH_ENGINES_CONFIG") or engines_home() / "engines.toml")
            data_dir = tomllib.loads(path.read_text(encoding = "utf-8")).get("comfyui", {}).get("data_dir")
        except Exception:  # noqa: BLE001 - no config means the default
            data_dir = None
        if not (isinstance(data_dir, str) and data_dir.strip()):
            data_dir = DEFAULT_DATA_DIR
        return Path(os.path.expanduser(data_dir)) / "temp" / SUBFOLDER
    except Exception as exc:  # noqa: BLE001 - e.g. no home directory
        logger.warning("Cannot locate ComfyUI's temp folder: %s", exc)
        return None


def remove_inputs(filenames: Any) -> None:
    """Delete uploads by name. Only names Studio generates, only inside the temp folder; never raises."""
    directory = temp_inputs_dir()
    if directory is None:
        return
    for name in list(filenames or []):
        if not isinstance(name, str) or not NAME_RE.match(name):
            logger.warning("Refusing to delete a ComfyUI input with an unexpected name: %r", name)
            continue
        try:
            (directory / name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not delete the ComfyUI input %s: %s", name, exc)


def sweep_stale() -> int:
    """Delete every Studio-named upload left in the temp folder (a crashed job, a stopped Studio). Returns the count."""
    directory = temp_inputs_dir()
    if directory is None:
        return 0
    removed = 0
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0
    for path in entries:
        if not NAME_RE.match(path.name):
            continue
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not delete the stale ComfyUI input %s: %s", path.name, exc)
    return removed

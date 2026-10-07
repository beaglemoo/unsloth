# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI API-format graph templates: loading, parameter filling, import and slot detection.

Pure: no network, and no file I/O except reading and writing template files. A template is a graph
in ComfyUI's API format plus ``slots`` (which node inputs carry each Studio parameter), an optional
``lora`` block (where to splice ``LoraLoaderModelOnly`` nodes), ``defaults``, ``limits`` and
``required_models``. Shipped templates live next to this module in ``comfyui_templates/``; the
user's imported graphs live in ``user_templates_dir()`` and their ids carry the ``user:`` prefix.
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

SHIPPED_DIR = Path(__file__).resolve().parent / "comfyui_templates"
USER_PREFIX = "user:"
TEMPLATES_DIR_ENV = "UNSLOTH_COMFYUI_TEMPLATES_DIR"
SCHEMA = 1
MAX_SEED = 2**53 - 1
MAX_GRAPH_NODES = 500

PLACEHOLDER_RE = re.compile(r"^\s*\{\{\s*([a-z_]+)\s*\}\}\s*$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")

# Parameter names a template can bind. ``prompt`` is the only mandatory slot.
SLOT_NAMES = (
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "width",
    "height",
    "batch_size",
)
_PLACEHOLDER_DEFAULTS: dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "seed": 0,
    "width": 1024,
    "height": 1024,
    "steps": 25,
    "cfg": 1.0,
    "batch_size": 1,
}
DEFAULT_LIMITS: dict[str, Any] = {
    "steps": [1, 100],
    "cfg": [0, 20],
    "side": [256, 2048],
    "multiple": 16,
    "batch_size": [1, 4],
}
_SAMPLERS = ("KSampler", "KSamplerAdvanced")
# Loader node -> (input holding the file name, ComfyUI model folder).
_LOADERS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "DualCLIPLoader": ("clip_name1", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
}
_OUTPUT_NODES = ("SaveImage", "PreviewImage")


class ComfyTemplateError(ValueError):
    """A template or imported graph is malformed; the message is shown to the user."""


class ComfyParamError(ValueError):
    """The generation parameters cannot be applied to the template."""


@dataclass(frozen = True)
class Template:
    id: str
    name: str
    kind: str
    graph: dict
    slots: dict
    lora: Optional[dict]
    defaults: dict
    limits: dict
    required_models: dict
    source: str = "shipped"
    path: Optional[Path] = field(default = None, compare = False)

    def to_file(self) -> dict:
        return {
            "schema": SCHEMA,
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "graph": self.graph,
            "slots": self.slots,
            "lora": self.lora,
            "defaults": self.defaults,
            "limits": self.limits,
            "required_models": self.required_models,
        }

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "defaults": self.defaults,
            "limits": self.limits,
            "slots": [name for name in SLOT_NAMES if self.slots.get(name)],
            "supports_lora": bool(self.lora),
            "required_models": self.required_models,
        }


# ---------------------------------------------------------------- validation


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def validate_api_graph(obj: Any) -> dict:
    """The graph itself when it is a valid API-format graph; ``ComfyTemplateError`` otherwise."""
    if not isinstance(obj, dict) or not obj:
        raise ComfyTemplateError("The graph must be a non-empty JSON object in ComfyUI's API format.")
    if isinstance(obj.get("nodes"), list) and ("links" in obj or "version" in obj):
        raise ComfyTemplateError(
            "This is a UI-format workflow. Export the workflow with Export (API) in ComfyUI."
        )
    if len(obj) > MAX_GRAPH_NODES:
        raise ComfyTemplateError(f"The graph has more than {MAX_GRAPH_NODES} nodes.")
    for node_id, node in obj.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            raise ComfyTemplateError(f"Node {node_id!r} is not an object.")
        if not isinstance(node.get("class_type"), str) or not node["class_type"]:
            raise ComfyTemplateError(f"Node {node_id!r} has no class_type.")
        if not isinstance(node.get("inputs"), dict):
            raise ComfyTemplateError(f"Node {node_id!r} has no inputs object.")
    for node_id, node in obj.items():
        for name, value in node["inputs"].items():
            if _is_link(value) and value[0] not in obj:
                raise ComfyTemplateError(
                    f"Node {node_id!r} input {name!r} points at the missing node {value[0]!r}."
                )
    return obj


def _check_slots(graph: dict, slots: Any) -> dict:
    if not isinstance(slots, dict):
        raise ComfyTemplateError("slots must be an object.")
    clean: dict[str, list[dict]] = {}
    for name, targets in slots.items():
        if name not in SLOT_NAMES:
            raise ComfyTemplateError(f"Unknown slot {name!r}.")
        if not isinstance(targets, list):
            raise ComfyTemplateError(f"Slot {name!r} must be a list.")
        clean[name] = []
        for target in targets:
            node, key = (target.get("node"), target.get("input")) if isinstance(target, dict) else (None, None)
            if node not in graph or key not in graph[node]["inputs"]:
                raise ComfyTemplateError(f"Slot {name!r} points at {node!r}.{key!r}, which is not in the graph.")
            if _is_link(graph[node]["inputs"][key]):
                raise ComfyTemplateError(f"Slot {name!r} points at a linked input ({node!r}.{key!r}).")
            clean[name].append({"node": node, "input": key})
    if not clean.get("prompt"):
        raise ComfyTemplateError("The graph has no prompt input Studio can fill.")
    return clean


def _check_lora(graph: dict, lora: Any) -> Optional[dict]:
    if lora in (None, {}):
        return None
    source = lora.get("model_source") if isinstance(lora, dict) else None
    consumers = lora.get("consumers") if isinstance(lora, dict) else None
    if not (isinstance(source, list) and len(source) == 2 and source[0] in graph and isinstance(consumers, list) and consumers):
        raise ComfyTemplateError("lora needs a model_source and consumers that exist in the graph.")
    for consumer in consumers:
        node, key = (consumer.get("node"), consumer.get("input")) if isinstance(consumer, dict) else (None, None)
        if node not in graph or not _is_link(graph[node]["inputs"].get(key)):
            raise ComfyTemplateError(f"lora consumer {node!r}.{key!r} is not a linked input.")
    return {
        "model_source": [source[0], int(source[1])],
        "consumers": [{"node": c["node"], "input": c["input"]} for c in consumers],
    }


def _template_from_dict(raw: Any, *, source: str, path: Optional[Path] = None, template_id: Optional[str] = None) -> Template:
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ComfyTemplateError("Unsupported template schema.")
    graph = validate_api_graph(raw.get("graph"))
    tid = template_id or raw.get("id")
    if not isinstance(tid, str) or not tid:
        raise ComfyTemplateError("Template has no id.")
    limits = {**DEFAULT_LIMITS, **(raw.get("limits") if isinstance(raw.get("limits"), dict) else {})}
    defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
    required = raw.get("required_models") if isinstance(raw.get("required_models"), dict) else {}
    return Template(
        id = tid,
        name = str(raw.get("name") or tid),
        kind = str(raw.get("kind") or "t2i"),
        graph = graph,
        slots = _check_slots(graph, raw.get("slots")),
        lora = _check_lora(graph, raw.get("lora")),
        defaults = dict(defaults),
        limits = limits,
        required_models = {str(k): [str(f) for f in v] for k, v in required.items() if isinstance(v, list)},
        source = source,
        path = path,
    )


# ------------------------------------------------------------------ storage


def user_templates_dir() -> Path:
    override = (os.environ.get(TEMPLATES_DIR_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    from utils.paths import studio_root

    return studio_root() / "comfyui-templates"


def _load_dir(directory: Path, *, source: str) -> list[Template]:
    out: list[Template] = []
    try:
        files = sorted(directory.glob("*.json"))
    except OSError:
        return out
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding = "utf-8"))
            tid = f"{USER_PREFIX}{path.stem}" if source == "user" else None
            out.append(_template_from_dict(raw, source = source, path = path, template_id = tid))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping ComfyUI template %s: %s", path.name, exc)
    return out


def load_templates() -> list[Template]:
    """Shipped templates first, then the user's. A bad file is skipped and logged."""
    return _load_dir(SHIPPED_DIR, source = "shipped") + _load_dir(user_templates_dir(), source = "user")


def get_template(template_id: str) -> Optional[Template]:
    for template in load_templates():
        if template.id == template_id:
            return template
    return None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48].strip("-")
    return slug or "graph"


def _unique_slug(directory: Path, name: str) -> str:
    base = _slugify(name)
    slug, n = base, 1
    while (directory / f"{slug}.json").exists():
        n += 1
        slug = f"{base[: 48 - len(str(n)) - 1]}-{n}"
    return slug


def save_user_template(name: str, graph: Any, slots: Optional[dict] = None) -> Template:
    """Validate an API-format graph, detect its parameter inputs and store it. Raises ``ComfyTemplateError``."""
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise ComfyTemplateError("Give the graph a name of 1 to 80 characters.")
    graph = copy.deepcopy(validate_api_graph(graph))
    detected = detect_slots(graph)
    chosen = _check_slots(graph, slots) if slots is not None else detected["slots"]
    directory = user_templates_dir()
    slug = _unique_slug(directory, name)
    template = Template(
        id = f"{USER_PREFIX}{slug}",
        name = name,
        kind = "t2i",
        graph = graph,
        slots = chosen,
        lora = detected["lora"],
        defaults = detected["defaults"],
        limits = dict(DEFAULT_LIMITS),
        required_models = detected["required_models"],
        source = "user",
    )
    try:
        directory.mkdir(parents = True, exist_ok = True)
        path = directory / f"{slug}.json"
        tmp = directory / f".{slug}.json.tmp"
        tmp.write_text(json.dumps(template.to_file(), indent = 2) + "\n", encoding = "utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        raise ComfyTemplateError(f"Could not store the graph: {exc.strerror or exc}") from exc
    return Template(**{**template.__dict__, "path": path})


def delete_user_template(template_id: str) -> bool:
    """Delete a user template; shipped ones cannot be deleted. False when it does not exist."""
    if not template_id.startswith(USER_PREFIX):
        return False
    slug = template_id[len(USER_PREFIX):]
    if not _SLUG_RE.match(slug):
        return False
    path = user_templates_dir() / f"{slug}.json"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


# ------------------------------------------------------------------ building


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComfyParamError(f"{name} must be a number.")
    return float(value)


def _clamp(value: float, bounds: Any) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


def build_graph(template: Template, params: dict) -> tuple[dict, dict]:
    """A submit-ready copy of the template graph and the effective parameters.

    Values outside ``limits`` are clamped, sizes snap down to ``limits.multiple``, a seed of ``None`` or
    below zero becomes a random one, and every ``SaveImage`` becomes a ``PreviewImage`` so ComfyUI keeps
    no copy of the output. ``params`` keys: the slot names plus ``loras`` (a list of ``{name, strength}``).
    """
    graph = copy.deepcopy(template.graph)
    limits, defaults, slots = template.limits, template.defaults, template.slots

    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ComfyParamError("A prompt is required.")
    resolved: dict[str, Any] = {"prompt": prompt}
    values: dict[str, Any] = {"prompt": prompt}

    negative = params.get("negative_prompt")
    values["negative_prompt"] = resolved["negative_prompt"] = negative if isinstance(negative, str) else ""

    seed = params.get("seed")
    if isinstance(seed, bool) or not (seed is None or isinstance(seed, int)):
        raise ComfyParamError(f"seed must be a whole number from 0 to {MAX_SEED}.")
    if seed is None or seed < 0:
        seed = random.randint(0, MAX_SEED)
    if seed > MAX_SEED:
        raise ComfyParamError(f"seed must be a whole number from 0 to {MAX_SEED}.")
    values["seed"] = resolved["seed"] = seed

    steps = params.get("steps")
    steps = defaults.get("steps", 25) if steps is None else steps
    values["steps"] = resolved["steps"] = int(_clamp(_num(steps, "steps"), limits["steps"]))

    cfg = params.get("cfg")
    cfg = defaults.get("cfg", 1.0) if cfg is None else cfg
    values["cfg"] = resolved["cfg"] = round(float(_clamp(_num(cfg, "cfg"), limits["cfg"])), 4)

    multiple = int(limits["multiple"])
    lo, hi = limits["side"]
    for side in ("width", "height"):
        value = params.get(side)
        value = defaults.get(side, 1024) if value is None else value
        snapped = int(_clamp(_num(value, side), (lo, hi))) // multiple * multiple
        values[side] = resolved[side] = max(snapped, -(-lo // multiple) * multiple)

    batch = params.get("batch_size")
    batch = 1 if batch is None else batch
    values["batch_size"] = resolved["batch_size"] = int(_clamp(_num(batch, "batch_size"), limits["batch_size"]))

    for name in ("sampler", "scheduler"):
        value = params.get(name)
        value = defaults.get(name) if value is None else value
        if value is not None:
            if not isinstance(value, str) or not value.strip() or len(value) > 64:
                raise ComfyParamError(f"{name} must be a short text value.")
            values[name] = resolved[name] = value.strip()

    for name, value in values.items():
        for target in slots.get(name, []):
            graph[target["node"]]["inputs"][target["input"]] = value

    applied = _splice_loras(graph, template, params.get("loras"))
    resolved["loras"] = [f"{name}:{strength:g}" for name, strength in applied]

    for node in graph.values():
        if node["class_type"] == "SaveImage":
            node["class_type"] = "PreviewImage"
            node["inputs"].pop("filename_prefix", None)
    return graph, resolved


def _splice_loras(graph: dict, template: Template, loras: Any) -> list[tuple[str, float]]:
    wanted: list[tuple[str, float]] = []
    for entry in loras or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        strength = entry.get("strength", 1.0) if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ComfyParamError("Each LoRA needs a file name.")
        strength = _num(strength, "LoRA strength")
        if strength != 0:
            wanted.append((name, strength))
    if not wanted:
        return []
    if not template.lora:
        raise ComfyParamError("This template does not support LoRAs.")
    source = list(template.lora["model_source"])
    for index, (name, strength) in enumerate(wanted):
        node_id = f"unsloth_lora_{index}"
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": source, "lora_name": name, "strength_model": strength},
        }
        source = [node_id, 0]
    for consumer in template.lora["consumers"]:
        graph[consumer["node"]]["inputs"][consumer["input"]] = source
    return wanted


# ----------------------------------------------------------------- detection


def _node(graph: dict, link: Any) -> Optional[dict]:
    return graph.get(link[0]) if _is_link(link) else None


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _text_node(graph: dict, link: Any) -> tuple[Optional[str], Optional[str]]:
    """The node (id, text input name) reached by following a conditioning link back to a text input."""
    for _ in range(8):
        if not _is_link(link):
            return None, None
        node_id = link[0]
        inputs = graph[node_id]["inputs"]
        for key in ("text", "prompt"):
            if _is_text(inputs.get(key)):
                return node_id, key
        link = inputs.get("conditioning") or inputs.get("conditioning_1")
    return None, None


def _pick_sampler(graph: dict) -> Optional[str]:
    candidates = sorted((i for i, n in graph.items() if n["class_type"] in _SAMPLERS), key = _natural)
    base = [i for i in candidates if graph[i]["inputs"].get("denoise", 1) in (1, 1.0)]
    return (base or candidates or [None])[0]


def _natural(value: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", value))


def detect_slots(graph: dict) -> dict:
    """Find the parameter-bearing inputs of an API-format graph.

    Returns ``{"slots", "lora", "defaults", "required_models"}`` and raises ``ComfyTemplateError`` when
    the graph has no prompt input or no image output. ``{{prompt}}``-style placeholders (the StoryPress
    convention) win; they are replaced by a default value in ``graph`` (in place) and become slots.
    Otherwise a ``KSampler`` / ``KSamplerAdvanced`` and the nodes it links to are inspected.
    """
    validate_api_graph(graph)
    slots: dict[str, list[dict]] = {}
    defaults: dict[str, Any] = {}

    for node_id, node in graph.items():
        for key, value in list(node["inputs"].items()):
            match = PLACEHOLDER_RE.match(value) if isinstance(value, str) else None
            if not match:
                continue
            name = match.group(1)
            if name not in _PLACEHOLDER_DEFAULTS:
                raise ComfyTemplateError(f"Unknown placeholder {{{{{name}}}}} at node {node_id!r}.")
            node["inputs"][key] = _PLACEHOLDER_DEFAULTS[name]
            slots.setdefault(name, []).append({"node": node_id, "input": key})
            if name not in ("prompt", "negative_prompt", "seed"):
                defaults[name] = _PLACEHOLDER_DEFAULTS[name]

    sampler_id = _pick_sampler(graph)
    lora = None
    if sampler_id is not None:
        sampler = graph[sampler_id]["inputs"]

        def bind(name: str, node_id: str, key: str, *, record_default: bool = True) -> None:
            if name in slots or key not in graph[node_id]["inputs"] or _is_link(graph[node_id]["inputs"][key]):
                return
            slots[name] = [{"node": node_id, "input": key}]
            if record_default and name not in ("prompt", "negative_prompt", "seed"):
                defaults[name] = graph[node_id]["inputs"][key]

        for name, key in (
            ("seed", "seed"), ("seed", "noise_seed"), ("steps", "steps"), ("cfg", "cfg"),
            ("sampler", "sampler_name"), ("scheduler", "scheduler"),
        ):
            bind(name, sampler_id, key)

        latent = _node(graph, sampler.get("latent_image"))
        if latent is not None and re.match(r"^Empty.*Latent", graph[sampler["latent_image"][0]]["class_type"]):
            latent_id = sampler["latent_image"][0]
            for key in ("width", "height", "batch_size"):
                bind(key, latent_id, key, record_default = key != "batch_size")

        positive_id, positive_key = _text_node(graph, sampler.get("positive"))
        if positive_id is not None:
            bind("prompt", positive_id, positive_key)
            if _is_text(graph[positive_id]["inputs"].get("negative_prompt")):
                bind("negative_prompt", positive_id, "negative_prompt")
        negative_id, negative_key = _text_node(graph, sampler.get("negative"))
        if negative_id is not None and negative_id != positive_id:
            bind("negative_prompt", negative_id, negative_key)

        source = sampler.get("model")
        if _is_link(source):
            lora = {
                "model_source": [source[0], int(source[1])],
                "consumers": [{"node": sampler_id, "input": "model"}],
            }

    if "seed" not in slots:
        for node_id, node in graph.items():
            if node["class_type"] == "RandomNoise" and not _is_link(node["inputs"].get("noise_seed")):
                slots["seed"] = [{"node": node_id, "input": "noise_seed"}]
                break

    if not slots.get("prompt"):
        raise ComfyTemplateError(
            "No prompt input was found. Use a KSampler whose positive conditioning comes from a text "
            "encode node, or put {{prompt}} in the text input and import again."
        )
    if not any(n["class_type"] in _OUTPUT_NODES for n in graph.values()):
        raise ComfyTemplateError("The graph needs a SaveImage or PreviewImage node.")
    if any(n["class_type"] in ("LoadImage", "LoadImageMask") for n in graph.values()):
        raise ComfyTemplateError("Graphs that load an input image are not supported yet.")

    required: dict[str, list[str]] = {}
    for node in graph.values():
        loader = _LOADERS.get(node["class_type"])
        if loader and _is_text(node["inputs"].get(loader[0])):
            required.setdefault(loader[1], [])
            if node["inputs"][loader[0]] not in required[loader[1]]:
                required[loader[1]].append(node["inputs"][loader[0]])
    return {"slots": slots, "lora": lora, "defaults": defaults, "required_models": required}


def graph_class_types(graph: dict) -> list[str]:
    return sorted({node["class_type"] for node in graph.values()})

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import comfyui_graphs as g


@pytest.fixture(autouse = True)
def _user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(g.TEMPLATES_DIR_ENV, str(tmp_path / "user-templates"))
    return tmp_path / "user-templates"


SHIPPED_IDS = [
    "qwen-image-2.1-t2i",
    "qwen-image-2.1-t2i-uncensored",
    "qwen-image-2.1-img2img",
    "qwen-image-2.1-edit",
]
T2I_SLOT_NAMES = set(g.SLOT_NAMES) - {"denoise", "reference_resolution"}
UNCENSORED_LORA = "qwen-image-2.1-uncensored-lora.safetensors"


def shipped() -> g.Template:
    template = g.get_template("qwen-image-2.1-t2i")
    assert template is not None
    return template


# StoryPress's qwen-image-2.1-api.json convention: placeholders in the prompt and seed inputs.
PLACEHOLDER_GRAPH = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "m.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors", "type": "qwen_image", "device": "cpu"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
    "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": "{{prompt}}", "negative_prompt": "", "resolution": 1024}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "seed": "{{seed}}", "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
    "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "StoryPress"}},
}

# A plain SD graph with no placeholders: every slot comes from the heuristics.
SD_GRAPH = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}},
}


# A VAEEncode img2img graph with no placeholders; the sampler runs at denoise 0.55.
IMG2IMG_GRAPH = {
    **SD_GRAPH,
    "10": {"class_type": "LoadImage", "inputs": {"image": "photo.png"}, "_meta": {"title": "Photo"}},
    "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}},
    "3": {**SD_GRAPH["3"], "inputs": {**SD_GRAPH["3"]["inputs"], "denoise": 0.55, "latent_image": ["13", 0]}},
}
del IMG2IMG_GRAPH["5"]

_REAL_GRAPH_NODES = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image", "device": "cpu"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
    "4": {"class_type": "LoadImage", "inputs": {"image": "{{reference_image}}"}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
}


def _edit_graph(*, encoder_inputs: dict, sampler_inputs: dict, prefix: str) -> str:
    graph = {
        **copy.deepcopy(_REAL_GRAPH_NODES),
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "images.image_1": ["4", 0], **encoder_inputs}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2],
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1, **sampler_inputs}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }
    return json.dumps(graph)


# The two real edit graphs on disk (StoryPress and PicturePress), reduced to what the tests rely on.
STORYPRESS_REFERENCE = _edit_graph(
    encoder_inputs = {"prompt": "{{prompt}}", "negative_prompt": "", "resolution": 0},
    sampler_inputs = {"seed": "{{seed}}", "steps": 25, "cfg": 1},
    prefix = "StoryPress-Reference",
)
PICTUREPRESS_EDIT = _edit_graph(
    encoder_inputs = {"prompt": "{{prompt}}", "negative_prompt": "{{negative_prompt}}", "resolution": "{{edit_resolution}}"},
    sampler_inputs = {"seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{cfg}}"},
    prefix = "PicturePress/edit",
)


# ---------------------------------------------------------------- shipped template


def test_shipped_template_loads_and_slots_point_at_real_inputs():
    template = shipped()
    assert template.source == "shipped" and template.kind == "t2i"
    assert set(template.slots) == T2I_SLOT_NAMES
    for targets in template.slots.values():
        for target in targets:
            assert target["input"] in template.graph[target["node"]]["inputs"]
    assert template.lora == {"model_source": ["1", 0], "consumers": [{"node": "6", "input": "model"}]}
    assert template.defaults["steps"] == 25 and template.defaults["cfg"] == 1.0
    assert template.graph["1"]["inputs"]["unet_name"] == "qwen_image_2.1_bf16.safetensors"
    assert template.graph["2"]["inputs"]["clip_name"] == "qwen3vl_8b_bf16.safetensors"
    assert template.graph["3"]["inputs"]["vae_name"] == "qwen_image_2.1_vae_bf16.safetensors"
    assert template.required_models["diffusion_models"] == ["qwen_image_2.1_bf16.safetensors"]
    assert template.graph["4"]["inputs"]["resolution"] == 1024
    assert shipped().summary()["supports_lora"] is True


def test_shipped_template_file_is_in_the_package_data_glob():
    assert (g.SHIPPED_DIR / "qwen-image-2.1-t2i.json").is_file()
    for name in ("qwen-image-2.1-img2img.json", "qwen-image-2.1-edit.json"):
        assert (g.SHIPPED_DIR / name).is_file()


def uncensored() -> g.Template:
    template = g.get_template("qwen-image-2.1-t2i-uncensored")
    assert template is not None
    return template


def test_uncensored_template_is_listed_after_the_standard_one_and_is_valid():
    shipped_ids = [t.id for t in g.load_templates() if t.source == "shipped"]
    assert shipped_ids == SHIPPED_IDS
    template = uncensored()
    assert template.name == "Qwen-Image 2.1 (uncensored)" and template.source == "shipped"
    g.validate_api_graph(template.graph)
    assert set(template.slots) == T2I_SLOT_NAMES
    assert template.summary()["supports_lora"] is True
    assert template.required_models["loras"] == [UNCENSORED_LORA]
    assert {k: v for k, v in template.required_models.items() if k != "loras"} == shipped().required_models


def test_uncensored_template_applies_the_lora_between_loader_and_sampler():
    graph, resolved = g.build_graph(uncensored(), {"prompt": "x", "seed": 1})
    assert graph["9"] == {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["1", 0], "lora_name": UNCENSORED_LORA, "strength_model": 1.0},
    }
    assert graph["6"]["inputs"]["model"] == ["9", 0]
    assert resolved["loras"] == []
    # Everything else is the standard graph.
    plain, _ = g.build_graph(shipped(), {"prompt": "x", "seed": 1})
    assert {k: v for k, v in graph.items() if k not in ("9",)} == {
        **plain, "6": {**plain["6"], "inputs": {**plain["6"]["inputs"], "model": ["9", 0]}}
    }


def test_user_loras_chain_after_the_uncensored_lora():
    loras = [{"name": "u0.safetensors", "strength": 0.5}, {"name": "u1.safetensors", "strength": 0.7}]
    graph, resolved = g.build_graph(uncensored(), {"prompt": "x", "seed": 1, "loras": loras})
    assert graph["unsloth_lora_0"]["inputs"]["model"] == ["9", 0]
    assert graph["unsloth_lora_1"]["inputs"]["model"] == ["unsloth_lora_0", 0]
    assert graph["6"]["inputs"]["model"] == ["unsloth_lora_1", 0]
    assert graph["9"]["inputs"]["model"] == ["1", 0]
    assert resolved["loras"] == ["u0.safetensors:0.5", "u1.safetensors:0.7"]


GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "comfyui_t2i_build_golden.json").read_text())


@pytest.mark.parametrize("template_id", ["qwen-image-2.1-t2i", "qwen-image-2.1-t2i-uncensored"])
@pytest.mark.parametrize("case", ["defaults", "full", "loras", "clamped"])
def test_t2i_graph_output_is_unchanged_by_the_image_input_work(template_id, case):
    # Recorded from the code before the image-input slots existed: the submitted graph must stay identical.
    recorded = GOLDEN[template_id][case]
    graph, resolved = g.build_graph(g.get_template(template_id), recorded["params"])
    assert graph == recorded["graph"]
    assert json.dumps(graph, sort_keys = True) == json.dumps(recorded["graph"], sort_keys = True)
    assert {k: v for k, v in resolved.items() if k != "kind"} == recorded["resolved"]
    assert resolved["kind"] == "t2i"


def template_by_id(template_id: str) -> g.Template:
    template = g.get_template(template_id)
    assert template is not None
    return template


def test_shipped_image_templates_are_valid_and_ordered():
    assert [t.id for t in g.load_templates() if t.source == "shipped"] == SHIPPED_IDS
    img2img, edit = template_by_id("qwen-image-2.1-img2img"), template_by_id("qwen-image-2.1-edit")
    assert (img2img.kind, edit.kind) == ("img2img", "edit")
    for template in (img2img, edit):
        g.validate_api_graph(template.graph)
        for targets in template.slots.values():
            for target in targets:
                value = template.graph[target["node"]]["inputs"][target["input"]]
                assert not g._is_link(value)
        assert list(template.image_slots) == ["image"]
        for target in template.image_slots["image"]["targets"]:
            node = template.graph[target["node"]]
            assert node["class_type"] == "LoadImage" and node["inputs"]["image"] == ""
        assert template.limits["batch_size"] == [1, 1] and "batch_size" not in template.slots
        assert template.required_models == shipped().required_models
        assert template.summary()["image_slots"] == [{"name": "image", "label": "Input image", "required": True}]
        assert template.summary()["supports_lora"] is True
    assert set(img2img.slots) == T2I_SLOT_NAMES - {"batch_size"} | {"denoise"}
    assert set(edit.slots) == {"prompt", "negative_prompt", "seed", "steps", "cfg", "sampler", "scheduler", "reference_resolution"}
    assert img2img.defaults["denoise"] == 0.6 and edit.defaults["reference_resolution"] == 1024
    assert shipped().summary()["image_slots"] == [] and shipped().image_slots == {}


def test_img2img_build_fills_the_image_denoise_and_size():
    template = template_by_id("qwen-image-2.1-img2img")
    before = copy.deepcopy(template.graph)
    graph, resolved = g.build_graph(
        template, {"prompt": "x", "seed": 1, "width": 1248, "height": 832, "images": {"image": "unsloth-inputs/a-image.png [temp]"}}
    )
    assert template.graph == before
    assert graph["5"]["inputs"]["image"] == "unsloth-inputs/a-image.png [temp]"
    assert (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]) == (1248, 832)
    assert graph["8"]["inputs"]["denoise"] == 0.6 and resolved["denoise"] == 0.6 and resolved["kind"] == "img2img"
    assert graph["8"]["inputs"]["latent_image"] == ["7", 0] and graph["10"]["class_type"] == "PreviewImage"
    assert "reference_resolution" not in resolved


@pytest.mark.parametrize("asked,expected", [(0.0, 0.01), (0.123456, 0.1235), (2, 1.0), (None, 0.6)])
def test_img2img_denoise_is_clamped_and_rounded(asked, expected):
    params = {"prompt": "x", "seed": 1, "images": {"image": "a.png"}}
    if asked is not None:
        params["denoise"] = asked
    graph, resolved = g.build_graph(template_by_id("qwen-image-2.1-img2img"), params)
    assert resolved["denoise"] == expected and graph["8"]["inputs"]["denoise"] == expected


@pytest.mark.parametrize("asked,expected", [(1000, 992), (99999, 2048), (10, 256), (768, 768), (None, 1024)])
def test_edit_reference_resolution_snaps_and_clamps(asked, expected):
    params = {"prompt": "x", "seed": 1, "images": {"image": "a.png"}}
    if asked is not None:
        params["reference_resolution"] = asked
    graph, resolved = g.build_graph(template_by_id("qwen-image-2.1-edit"), params)
    assert resolved["reference_resolution"] == expected
    assert graph["5"]["inputs"]["resolution"] == expected and graph["4"]["inputs"]["image"] == "a.png"
    assert graph["6"]["inputs"]["denoise"] == 1 and "denoise" not in resolved


def test_reference_resolution_zero_only_when_it_is_the_template_default():
    with pytest.raises(g.ComfyParamError, match = "keep the input size"):
        g.build_graph(
            template_by_id("qwen-image-2.1-edit"), {"prompt": "x", "reference_resolution": 0, "images": {"image": "a.png"}}
        )
    storypress = g.save_user_template("storypress", json.loads(STORYPRESS_REFERENCE))
    graph, resolved = g.build_graph(storypress, {"prompt": "x", "seed": 1, "images": {"image": "a.png"}})
    assert resolved["reference_resolution"] == 0 and graph["5"]["inputs"]["resolution"] == 0
    graph, resolved = g.build_graph(storypress, {"prompt": "x", "images": {"image": "a.png"}, "reference_resolution": 1000})
    assert resolved["reference_resolution"] == 992


def test_image_params_are_checked_against_the_template():
    img2img, edit = template_by_id("qwen-image-2.1-img2img"), template_by_id("qwen-image-2.1-edit")
    before = copy.deepcopy(img2img.graph)
    for params in ({"prompt": "x"}, {"prompt": "x", "images": {}}, {"prompt": "x", "images": {"image": ""}}):
        with pytest.raises(g.ComfyParamError, match = "needs an input image \\(Input image\\)"):
            g.build_graph(img2img, params)
    with pytest.raises(g.ComfyParamError, match = "no input image 'image_2'"):
        g.build_graph(edit, {"prompt": "x", "images": {"image": "a.png", "image_2": "b.png"}})
    with pytest.raises(g.ComfyParamError, match = "no input image 'image'"):
        g.build_graph(shipped(), {"prompt": "x", "images": {"image": "a.png"}})
    with pytest.raises(g.ComfyParamError):
        g.build_graph(img2img, {"prompt": "x", "images": ["a.png"]})
    assert img2img.graph == before


# ------------------------------------------------------------------ build_graph


def test_build_graph_fills_every_slot_and_leaves_the_template_untouched():
    template = shipped()
    before = copy.deepcopy(template.graph)
    graph, resolved = g.build_graph(
        template,
        {"prompt": "a fox", "negative_prompt": "blur", "width": 768, "height": 512, "seed": 42,
         "steps": 30, "cfg": 2.5, "sampler": "dpmpp_2m", "scheduler": "karras", "batch_size": 2},
    )
    assert template.graph == before
    assert graph["4"]["inputs"]["prompt"] == "a fox" and graph["4"]["inputs"]["negative_prompt"] == "blur"
    assert graph["5"]["inputs"] == {"width": 768, "height": 512, "batch_size": 2}
    sampler = graph["6"]["inputs"]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (42, 30, 2.5)
    assert (sampler["sampler_name"], sampler["scheduler"]) == ("dpmpp_2m", "karras")
    assert resolved["seed"] == 42 and resolved["steps"] == 30 and resolved["loras"] == []


def test_build_graph_uses_template_defaults():
    graph, resolved = g.build_graph(shipped(), {"prompt": "x", "seed": 1})
    assert resolved["steps"] == 25 and resolved["cfg"] == 1.0
    assert (resolved["width"], resolved["height"]) == (1024, 1024)
    assert graph["6"]["inputs"]["sampler_name"] == "euler"


def test_build_graph_clamps_and_snaps():
    _, resolved = g.build_graph(
        shipped(), {"prompt": "x", "seed": 1, "steps": 500, "cfg": 99, "width": 1000, "height": 100, "batch_size": 9}
    )
    assert resolved["steps"] == 100 and resolved["cfg"] == 20.0 and resolved["batch_size"] == 4
    assert resolved["width"] == 992  # 1000 snapped down to a multiple of 16
    assert resolved["height"] == 256  # raised to the lower side limit (a multiple of 16)


@pytest.mark.parametrize("seed", [None, -1])
def test_seed_none_or_negative_is_random_and_in_range(seed):
    seeds = {g.build_graph(shipped(), {"prompt": "x", "seed": seed})[1]["seed"] for _ in range(5)}
    assert len(seeds) > 1 and all(0 <= s <= g.MAX_SEED for s in seeds)


@pytest.mark.parametrize(
    "params",
    [{"prompt": ""}, {"prompt": "   "}, {"prompt": 3}, {"prompt": "x", "seed": 2**60},
     {"prompt": "x", "seed": True}, {"prompt": "x", "seed": "7"}, {"prompt": "x", "steps": "many"},
     {"prompt": "x", "sampler": ""}],
)
def test_build_graph_rejects_bad_params(params):
    with pytest.raises(g.ComfyParamError):
        g.build_graph(shipped(), params)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_loras_chain_and_rewire_the_consumers(count):
    loras = [{"name": f"l{i}.safetensors", "strength": 0.5 + i} for i in range(count)]
    graph, resolved = g.build_graph(shipped(), {"prompt": "x", "seed": 1, "loras": loras})
    nodes = [n for n in graph if n.startswith("unsloth_lora_")]
    assert len(nodes) == count
    assert resolved["loras"] == [f"l{i}.safetensors:{0.5 + i:g}" for i in range(count)]
    if count == 0:
        assert graph["6"]["inputs"]["model"] == ["1", 0]
        return
    assert graph["unsloth_lora_0"]["inputs"] == {
        "model": ["1", 0], "lora_name": "l0.safetensors", "strength_model": 0.5,
    }
    assert graph["unsloth_lora_0"]["class_type"] == "LoraLoaderModelOnly"
    if count == 2:
        assert graph["unsloth_lora_1"]["inputs"]["model"] == ["unsloth_lora_0", 0]
    assert graph["6"]["inputs"]["model"] == [f"unsloth_lora_{count - 1}", 0]


def test_zero_strength_lora_is_skipped_and_unsupported_template_refuses():
    graph, resolved = g.build_graph(shipped(), {"prompt": "x", "seed": 1, "loras": [{"name": "a", "strength": 0}]})
    assert "unsloth_lora_0" not in graph and resolved["loras"] == []
    template = g.Template(**{**shipped().__dict__, "lora": None})
    with pytest.raises(g.ComfyParamError, match = "LoRA"):
        g.build_graph(template, {"prompt": "x", "loras": [{"name": "a", "strength": 1}]})
    with pytest.raises(g.ComfyParamError):
        g.build_graph(shipped(), {"prompt": "x", "loras": [{"strength": 1}]})


def test_save_image_becomes_preview_image():
    graph, _ = g.build_graph(shipped(), {"prompt": "x", "seed": 1})
    assert graph["8"]["class_type"] == "PreviewImage"
    assert "filename_prefix" not in graph["8"]["inputs"]
    assert shipped().graph["8"]["class_type"] == "SaveImage"


# ------------------------------------------------------------------- validation


def test_validation_rejects_ui_format_and_malformed_graphs():
    with pytest.raises(g.ComfyTemplateError, match = "Export \\(API\\)"):
        g.validate_api_graph({"nodes": [], "links": [], "version": 0.4})
    for bad in (None, [], {}, {"1": []}, {"1": {"inputs": {}}}, {"1": {"class_type": "X"}}):
        with pytest.raises(g.ComfyTemplateError):
            g.validate_api_graph(bad)
    with pytest.raises(g.ComfyTemplateError, match = "missing node"):
        g.validate_api_graph({"1": {"class_type": "X", "inputs": {"a": ["9", 0]}}})
    huge = {str(i): {"class_type": "X", "inputs": {}} for i in range(g.MAX_GRAPH_NODES + 1)}
    with pytest.raises(g.ComfyTemplateError, match = "more than"):
        g.validate_api_graph(huge)


# -------------------------------------------------------------------- detection


def test_detect_slots_on_the_storypress_placeholder_graph():
    graph = copy.deepcopy(PLACEHOLDER_GRAPH)
    found = g.detect_slots(graph)
    slots = found["slots"]
    assert slots["prompt"] == [{"node": "4", "input": "prompt"}]
    assert slots["seed"] == [{"node": "6", "input": "seed"}]
    assert slots["negative_prompt"] == [{"node": "4", "input": "negative_prompt"}]
    assert slots["width"] == [{"node": "5", "input": "width"}]
    assert slots["steps"] == [{"node": "6", "input": "steps"}]
    assert graph["4"]["inputs"]["prompt"] == "" and graph["6"]["inputs"]["seed"] == 0
    assert found["lora"] == {"model_source": ["1", 0], "consumers": [{"node": "6", "input": "model"}]}
    assert found["defaults"]["steps"] == 25 and found["defaults"]["sampler"] == "euler"
    assert found["required_models"] == {
        "diffusion_models": ["m.safetensors"], "text_encoders": ["te.safetensors"], "vae": ["vae.safetensors"],
    }


def test_detect_slots_heuristics_without_placeholders():
    found = g.detect_slots(copy.deepcopy(SD_GRAPH))
    slots = found["slots"]
    assert slots["prompt"] == [{"node": "6", "input": "text"}]
    assert slots["negative_prompt"] == [{"node": "7", "input": "text"}]
    assert slots["seed"] == [{"node": "3", "input": "seed"}]
    assert slots["width"] == [{"node": "5", "input": "width"}] and slots["height"][0]["node"] == "5"
    assert slots["cfg"] == [{"node": "3", "input": "cfg"}] and slots["scheduler"][0]["input"] == "scheduler"
    assert found["defaults"]["cfg"] == 7 and found["defaults"]["width"] == 512
    assert found["required_models"] == {"checkpoints": ["sd.safetensors"]}
    assert found["lora"]["model_source"] == ["4", 0]


def test_detect_follows_conditioning_wrappers():
    graph = copy.deepcopy(SD_GRAPH)
    graph["10"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": 3.5}}
    graph["3"]["inputs"]["positive"] = ["10", 0]
    assert g.detect_slots(graph)["slots"]["prompt"] == [{"node": "6", "input": "text"}]


def test_detect_requires_prompt_and_output():
    no_prompt = copy.deepcopy(SD_GRAPH)
    del no_prompt["3"]
    no_prompt.pop("8"), no_prompt.pop("9")
    with pytest.raises(g.ComfyTemplateError, match = "No prompt"):
        g.detect_slots(no_prompt)
    no_output = copy.deepcopy(SD_GRAPH)
    del no_output["9"]
    with pytest.raises(g.ComfyTemplateError, match = "SaveImage"):
        g.detect_slots(no_output)


def test_detect_accepts_the_storypress_reference_graph_as_an_edit():
    graph = json.loads(STORYPRESS_REFERENCE)
    found = g.detect_slots(graph)
    assert found["kind"] == "edit"
    assert found["image_slots"] == {"image": {"label": "Input image 1", "targets": [{"node": "4", "input": "image"}]}}
    assert graph["4"]["inputs"]["image"] == ""
    assert found["slots"]["reference_resolution"] == [{"node": "5", "input": "resolution"}]
    assert found["defaults"]["reference_resolution"] == 0
    assert "denoise" not in found["slots"] and "width" not in found["slots"]


def test_detect_accepts_the_picturepress_edit_graph_through_the_alias():
    graph = json.loads(PICTUREPRESS_EDIT)
    found = g.detect_slots(graph)
    assert found["kind"] == "edit" and list(found["image_slots"]) == ["image"]
    assert found["slots"]["reference_resolution"] == [{"node": "5", "input": "resolution"}]
    assert found["defaults"]["reference_resolution"] == 1024
    assert found["defaults"]["steps"] == 25 and found["slots"]["cfg"] == [{"node": "6", "input": "cfg"}]
    assert graph["5"]["inputs"]["resolution"] == 1024 and graph["4"]["inputs"]["image"] == ""


def test_detect_infers_img2img_from_a_vae_encode_graph():
    found = g.detect_slots(copy.deepcopy(IMG2IMG_GRAPH))
    assert found["kind"] == "img2img"
    assert found["image_slots"] == {"image": {"label": "Photo", "targets": [{"node": "10", "input": "image"}]}}
    assert found["slots"]["denoise"] == [{"node": "3", "input": "denoise"}]
    assert found["defaults"]["denoise"] == 0.55
    assert "reference_resolution" not in found["slots"]


def test_detect_names_several_input_images_in_node_order():
    graph = copy.deepcopy(IMG2IMG_GRAPH)
    graph["11"] = {"class_type": "LoadImage", "inputs": {"image": "b.png"}}
    graph["2"] = {"class_type": "LoadImage", "inputs": {"image": "{{reference_image}}"}}
    found = g.detect_slots(graph)
    assert found["image_slots"]["image"]["targets"] == [{"node": "2", "input": "image"}]
    assert found["image_slots"]["image_2"]["targets"] == [{"node": "10", "input": "image"}]
    assert found["image_slots"]["image_3"]["label"] == "Input image 3"
    assert graph["10"]["inputs"]["image"] == "" and graph["11"]["inputs"]["image"] == ""


def test_detect_rejects_five_input_images_and_masks():
    graph = copy.deepcopy(IMG2IMG_GRAPH)
    for i in range(20, 24):
        graph[str(i)] = {"class_type": "LoadImage", "inputs": {"image": "x.png"}}
    with pytest.raises(g.ComfyTemplateError, match = "at most 4"):
        g.detect_slots(graph)
    masked = copy.deepcopy(IMG2IMG_GRAPH)
    masked["12"] = {"class_type": "LoadImageMask", "inputs": {"image": "m.png", "channel": "alpha"}}
    with pytest.raises(g.ComfyTemplateError, match = "Mask inputs"):
        g.detect_slots(masked)


def test_detect_rejects_a_linked_load_image_mask_output():
    graph = copy.deepcopy(IMG2IMG_GRAPH)
    graph["30"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["10", 1]}}
    with pytest.raises(g.ComfyTemplateError, match = "Mask inputs"):
        g.detect_slots(graph)


def test_detect_accepts_a_graph_using_only_the_image_output():
    graph = copy.deepcopy(IMG2IMG_GRAPH)
    graph["30"] = {"class_type": "ImageScale", "inputs": {"image": ["10", 0]}}
    assert "image" in g.detect_slots(graph)["image_slots"]


def test_a_saved_user_template_linking_the_mask_output_is_skipped(_user_dir):
    graph = copy.deepcopy(IMG2IMG_GRAPH)
    graph["30"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["10", 1]}}
    raw = {"schema": g.SCHEMA, "id": "x", "name": "Masked", "graph": graph}
    with pytest.raises(g.ComfyTemplateError, match = "Mask inputs"):
        g._template_from_dict(raw, source = "user")


def test_an_image_placeholder_on_another_node_is_unknown():
    graph = copy.deepcopy(SD_GRAPH)
    graph["6"]["inputs"]["text"] = "{{reference_image}}"
    with pytest.raises(g.ComfyTemplateError, match = "Unknown placeholder"):
        g.detect_slots(graph)


def test_detect_prefers_the_base_pass_of_two_samplers():
    graph = copy.deepcopy(SD_GRAPH)
    graph["20"] = {"class_type": "KSampler", "inputs": {**graph["3"]["inputs"], "seed": 99, "denoise": 0.5, "latent_image": ["3", 0]}}
    assert g.detect_slots(graph)["slots"]["seed"] == [{"node": "3", "input": "seed"}]


# ----------------------------------------------------------------- user templates


def test_user_template_round_trip_and_delete(_user_dir):
    saved = g.save_user_template("My Graph!", copy.deepcopy(SD_GRAPH))
    assert saved.id == "user:my-graph" and saved.source == "user"
    assert (_user_dir / "my-graph.json").is_file()
    loaded = g.get_template("user:my-graph")
    assert loaded is not None and loaded.slots == saved.slots and loaded.graph == saved.graph
    assert [t.id for t in g.load_templates()] == SHIPPED_IDS + ["user:my-graph"]
    again = g.save_user_template("My Graph!", copy.deepcopy(SD_GRAPH))
    assert again.id == "user:my-graph-2"
    graph, resolved = g.build_graph(loaded, {"prompt": "dog", "seed": 3})
    assert graph["6"]["inputs"]["text"] == "dog" and graph["9"]["class_type"] == "PreviewImage"
    assert g.delete_user_template("user:my-graph") is True
    assert g.delete_user_template("user:my-graph") is False
    assert g.get_template("user:my-graph") is None


def test_user_templates_keep_kind_and_image_slots(_user_dir):
    saved = g.save_user_template("Photo to painting", copy.deepcopy(IMG2IMG_GRAPH))
    assert saved.kind == "img2img" and list(saved.image_slots) == ["image"]
    assert saved.limits["batch_size"] == [1, 1]
    stored = json.loads((_user_dir / "photo-to-painting.json").read_text())
    assert stored["schema"] == 1 and stored["kind"] == "img2img" and stored["image_slots"] == saved.image_slots
    loaded = g.get_template("user:photo-to-painting")
    assert loaded is not None and loaded.kind == "img2img" and loaded.image_slots == saved.image_slots
    assert loaded.slots["denoise"] == [{"node": "3", "input": "denoise"}]
    graph, resolved = g.build_graph(loaded, {"prompt": "x", "seed": 1, "denoise": 0.4, "images": {"image": "n.png"}})
    assert graph["10"]["inputs"]["image"] == "n.png" and graph["3"]["inputs"]["denoise"] == 0.4
    assert loaded.summary()["kind"] == "img2img" and loaded.summary()["image_slots"][0]["label"] == "Photo"
    # A manual slot mapping does not drop the detected image slots.
    manual = g.save_user_template("manual photo", copy.deepcopy(IMG2IMG_GRAPH), {"prompt": [{"node": "6", "input": "text"}]})
    assert list(manual.image_slots) == ["image"] and manual.kind == "img2img"
    assert g.save_user_template("plain", copy.deepcopy(SD_GRAPH)).kind == "t2i"
    edit = g.save_user_template("pp", json.loads(PICTUREPRESS_EDIT))
    assert edit.kind == "edit" and g.get_template(edit.id).slots["reference_resolution"]


def test_old_user_files_without_image_slots_still_load(_user_dir):
    _user_dir.mkdir(parents = True)
    old = g.Template(**{**g.save_user_template("seed", copy.deepcopy(SD_GRAPH)).__dict__}).to_file()
    del old["image_slots"]
    (_user_dir / "old.json").write_text(json.dumps(old))
    loaded = g.get_template("user:old")
    assert loaded is not None and loaded.image_slots == {} and loaded.summary()["image_slots"] == []


@pytest.mark.parametrize(
    "bad",
    [
        {"image": {"label": "x", "targets": []}},
        {"photo": {"label": "x", "targets": [{"node": "10", "input": "image"}]}},
        {"image": {"label": "x", "targets": [{"node": "3", "input": "seed"}]}},
        {"image": {"label": "x", "targets": [{"node": "99", "input": "image"}]}},
        {"image": {"label": "x", "targets": [{"node": "13", "input": "pixels"}]}},
        [],
    ],
)
def test_image_slots_must_point_at_a_load_image_input(bad):
    with pytest.raises(g.ComfyTemplateError):
        g._check_image_slots(copy.deepcopy(IMG2IMG_GRAPH), bad)
    assert g._check_image_slots(copy.deepcopy(IMG2IMG_GRAPH), None) == {}


def test_delete_refuses_shipped_and_traversal(_user_dir):
    assert g.delete_user_template("qwen-image-2.1-t2i") is False
    (_user_dir.parent / "outside.json").write_text("{}")
    assert g.delete_user_template("user:../outside") is False
    assert (_user_dir.parent / "outside.json").exists()


def test_save_rejects_bad_name_and_unusable_graph():
    with pytest.raises(g.ComfyTemplateError):
        g.save_user_template("  ", copy.deepcopy(SD_GRAPH))
    with pytest.raises(g.ComfyTemplateError):
        g.save_user_template("x", {"nodes": [], "links": []})


def test_bad_user_files_are_skipped(_user_dir):
    _user_dir.mkdir(parents = True)
    (_user_dir / "broken.json").write_text("{not json")
    (_user_dir / "wrong.json").write_text(json.dumps({"schema": 9}))
    (_user_dir / "dangling.json").write_text(json.dumps({
        "schema": 1, "id": "x", "graph": SD_GRAPH, "slots": {"prompt": [{"node": "99", "input": "text"}]},
    }))
    g.save_user_template("good", copy.deepcopy(SD_GRAPH))
    assert [t.id for t in g.load_templates()] == SHIPPED_IDS + ["user:good"]


def test_manual_slot_mapping_is_validated():
    chosen = {"prompt": [{"node": "7", "input": "text"}]}
    assert g.save_user_template("manual", copy.deepcopy(SD_GRAPH), chosen).slots == chosen
    with pytest.raises(g.ComfyTemplateError):
        g.save_user_template("bad", copy.deepcopy(SD_GRAPH), {"prompt": [{"node": "7", "input": "nope"}]})
    with pytest.raises(g.ComfyTemplateError):
        g.save_user_template("bad", copy.deepcopy(SD_GRAPH), {"prompt": [{"node": "3", "input": "model"}]})

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


# ---------------------------------------------------------------- shipped template


def test_shipped_template_loads_and_slots_point_at_real_inputs():
    template = shipped()
    assert template.source == "shipped" and template.kind == "t2i"
    assert set(template.slots) == set(g.SLOT_NAMES)
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


def test_detect_requires_prompt_and_output_and_rejects_image_inputs():
    no_prompt = copy.deepcopy(SD_GRAPH)
    del no_prompt["3"]
    no_prompt.pop("8"), no_prompt.pop("9")
    with pytest.raises(g.ComfyTemplateError, match = "No prompt"):
        g.detect_slots(no_prompt)
    no_output = copy.deepcopy(SD_GRAPH)
    del no_output["9"]
    with pytest.raises(g.ComfyTemplateError, match = "SaveImage"):
        g.detect_slots(no_output)
    with_image = copy.deepcopy(SD_GRAPH)
    with_image["11"] = {"class_type": "LoadImage", "inputs": {"image": "a.png"}}
    with pytest.raises(g.ComfyTemplateError, match = "input image"):
        g.detect_slots(with_image)
    reference = copy.deepcopy(PLACEHOLDER_GRAPH)
    reference["10"] = {"class_type": "LoadImage", "inputs": {"image": "{{reference_image}}"}}
    with pytest.raises(g.ComfyTemplateError, match = "reference_image"):
        g.detect_slots(reference)


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
    assert [t.id for t in g.load_templates()] == ["qwen-image-2.1-t2i", "user:my-graph"]
    again = g.save_user_template("My Graph!", copy.deepcopy(SD_GRAPH))
    assert again.id == "user:my-graph-2"
    graph, resolved = g.build_graph(loaded, {"prompt": "dog", "seed": 3})
    assert graph["6"]["inputs"]["text"] == "dog" and graph["9"]["class_type"] == "PreviewImage"
    assert g.delete_user_template("user:my-graph") is True
    assert g.delete_user_template("user:my-graph") is False
    assert g.get_template("user:my-graph") is None


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
    assert [t.id for t in g.load_templates()] == ["qwen-image-2.1-t2i", "user:good"]


def test_manual_slot_mapping_is_validated():
    chosen = {"prompt": [{"node": "7", "input": "text"}]}
    assert g.save_user_template("manual", copy.deepcopy(SD_GRAPH), chosen).slots == chosen
    with pytest.raises(g.ComfyTemplateError):
        g.save_user_template("bad", copy.deepcopy(SD_GRAPH), {"prompt": [{"node": "7", "input": "nope"}]})
    with pytest.raises(g.ComfyTemplateError):
        g.save_user_template("bad", copy.deepcopy(SD_GRAPH), {"prompt": [{"node": "3", "input": "model"}]})

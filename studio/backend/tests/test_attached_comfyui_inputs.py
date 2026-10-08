# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Input images for ComfyUI jobs: planning, preparation and the temp-file lifecycle."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import comfyui_graphs as g
from core.inference.attached import comfyui_inputs as inputs


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv(g.TEMPLATES_DIR_ENV, str(tmp_path / "templates"))
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path / "engines"))
    monkeypatch.delenv("UNSLOTH_ENGINES_CONFIG", raising = False)
    monkeypatch.setattr(inputs.os.path, "expanduser", lambda p: str(p).replace("~", str(tmp_path / "home"), 1))
    return tmp_path


def encode(image: Image.Image, fmt = "PNG", **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format = fmt, **kwargs)
    return buf.getvalue()


def plan_for(**kwargs) -> inputs.InputPlan:
    return inputs.InputPlan(slot = "image", filename = "0" * 32 + "-image.png", annotated = "x", **kwargs)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def open_png(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG"
    image.load()
    return image


def template(name = "qwen-image-2.1-img2img") -> g.Template:
    found = g.get_template(name)
    assert found is not None
    return found


# ---------------------------------------------------------------------- plan


def test_plan_names_the_upload_per_slot():
    plans = inputs.plan_inputs(template(), {"image": {"data": "abc", "gallery_id": None}})
    plan = plans["image"]
    assert inputs.NAME_RE.match(plan.filename) and plan.filename.endswith("-image.png")
    assert plan.annotated == f"unsloth-inputs/{plan.filename} [temp]"
    assert (plan.data, plan.gallery_id) == ("abc", None)
    other = inputs.plan_inputs(template(), {"image": {"gallery_id": "a1b2"}})["image"]
    assert other.filename != plan.filename and other.gallery_id == "a1b2" and other.data is None


@pytest.mark.parametrize(
    "given,match",
    [
        ({}, "needs an input image \\(Input image\\)"),
        (None, "needs an input image"),
        ({"image": None}, "needs an input image"),
        ({"image": {"data": None, "gallery_id": None}}, "either data or a gallery_id"),
        ({"image": {"data": "a", "gallery_id": "b"}}, "either data or a gallery_id"),
        ({"image": {"data": "a"}, "image_2": {"data": "b"}}, "no input image 'image_2'"),
        ([], "must be an object"),
    ],
)
def test_plan_refuses_missing_unknown_and_ambiguous_slots(given, match):
    with pytest.raises(g.ComfyParamError, match = match):
        inputs.plan_inputs(template(), given)


def test_a_text_to_image_template_takes_no_images():
    t2i = template("qwen-image-2.1-t2i")
    assert inputs.plan_inputs(t2i, {}) == {} and inputs.plan_inputs(t2i, None) == {}
    with pytest.raises(g.ComfyParamError, match = "no input image 'image'"):
        inputs.plan_inputs(t2i, {"image": {"data": "a"}})


# ------------------------------------------------------------------ load_png


def test_data_url_and_raw_base64_both_decode():
    data = encode(Image.new("RGB", (40, 30), (10, 200, 30)))
    for text in (f"data:image/png;base64,{b64(data)}", b64(data), f"  {b64(data)[:20]}\n{b64(data)[20:]} "):
        out = open_png(inputs.load_png(plan_for(data = text)))
        assert out.size == (40, 30) and out.mode == "RGB" and out.getpixel((1, 1)) == (10, 200, 30)


def test_an_exif_rotated_jpeg_comes_out_upright():
    base = Image.new("RGB", (60, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    out = open_png(inputs.load_png(plan_for(data = b64(encode(base, "JPEG", exif = exif)))))
    assert out.size == (20, 60)


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_transparency_is_flattened_over_white(mode):
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    if mode == "LA":
        image = Image.new("LA", (8, 8), (0, 0))
    if mode == "P":
        image = Image.new("RGBA", (8, 8), (255, 0, 0, 0)).convert("P")
        image.info["transparency"] = 0
    out = open_png(inputs.load_png(plan_for(data = b64(encode(image)))))
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)


def test_a_large_image_is_capped_at_2048_on_the_long_side():
    out = open_png(inputs.load_png(plan_for(data = b64(encode(Image.new("RGB", (3000, 1500)))))))
    assert out.size == (2048, 1024)
    small = open_png(inputs.load_png(plan_for(data = b64(encode(Image.new("RGB", (2048, 100)))))))
    assert small.size == (2048, 100)


def test_an_image_over_4096_is_refused_before_it_is_decoded(monkeypatch):
    data = encode(Image.new("L", (5000, 10)))
    loads = []
    original = Image.Image.load
    monkeypatch.setattr(Image.Image, "load", lambda self, *a, **k: loads.append(1) or original(self, *a, **k))
    with pytest.raises(g.ComfyParamError, match = "at most 4096"):
        inputs.load_png(plan_for(data = b64(data)))
    assert loads == []


@pytest.mark.parametrize("text", ["not base64 at all!!", "data:text/plain,hello", b64(b"plain bytes, not an image"), ""])
def test_garbage_is_a_param_error(text):
    with pytest.raises(g.ComfyParamError):
        inputs.load_png(plan_for(data = text))


def test_a_gallery_image_is_read_from_disk():
    from core.inference import image_gallery

    source = Image.new("RGB", (64, 32), (0, 0, 255))
    record = image_gallery.save(source, {"prompt": "x", "width": 64, "height": 32, "steps": 1, "guidance": 1.0, "seed": 1, "created_at": 1.0})
    out = open_png(inputs.load_png(plan_for(gallery_id = record["id"])))
    assert out.size == (64, 32) and out.getpixel((0, 0)) == (0, 0, 255)
    with pytest.raises(g.ComfyParamError, match = "no longer exists"):
        inputs.load_png(plan_for(gallery_id = "deadbeef"))
    with pytest.raises(g.ComfyParamError, match = "no longer exists"):
        inputs.load_png(plan_for(gallery_id = "../../etc/passwd"))


# ------------------------------------------------------------------- cleanup


def write_config(tmp_path, data_dir):
    path = tmp_path / "engines" / "engines.toml"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(f'[comfyui]\nport = 8844\ndata_dir = "{data_dir}"\n')


def test_the_temp_dir_comes_from_engines_toml(tmp_path):
    write_config(tmp_path, tmp_path / "custom-data")
    assert inputs.temp_inputs_dir() == tmp_path / "custom-data" / "temp" / "unsloth-inputs"


def test_the_temp_dir_honours_the_config_override_and_tilde(tmp_path, monkeypatch):
    other = tmp_path / "other.toml"
    other.write_text('[comfyui]\ndata_dir = "~/elsewhere"\n')
    monkeypatch.setenv("UNSLOTH_ENGINES_CONFIG", str(other))
    assert inputs.temp_inputs_dir() == tmp_path / "home" / "elsewhere" / "temp" / "unsloth-inputs"


@pytest.mark.parametrize("body", [None, "[comfyui]\nport = 1\n", "not toml [", '[comfyui]\ndata_dir = 5\n'])
def test_the_temp_dir_falls_back_to_the_default(tmp_path, body):
    if body is not None:
        path = tmp_path / "engines" / "engines.toml"
        path.parent.mkdir(parents = True)
        path.write_text(body)
    assert inputs.temp_inputs_dir() == tmp_path / "engines" / "comfyui-data" / "temp" / "unsloth-inputs"


def stage(tmp_path):
    write_config(tmp_path, tmp_path / "data")
    directory = tmp_path / "data" / "temp" / "unsloth-inputs"
    directory.mkdir(parents = True)
    return directory


def test_remove_inputs_only_deletes_studio_names_inside_the_folder(tmp_path):
    directory = stage(tmp_path)
    mine, mine2 = "a" * 32 + "-image.png", "b" * 32 + "-image_2.png"
    for name in (mine, mine2, "keep.png", "g" * 32 + "-image.png"):
        (directory / name).write_bytes(b"x")
    outside = tmp_path / "data" / "temp" / ("c" * 32 + "-image.png")
    outside.write_bytes(b"x")
    inputs.remove_inputs([mine, mine2, "../" + outside.name, "keep.png", "g" * 32 + "-image.png", "gone" , "c" * 32 + "-image.png"])
    assert sorted(p.name for p in directory.iterdir()) == ["g" * 32 + "-image.png", "keep.png"]
    assert outside.exists()
    inputs.remove_inputs([mine])  # already gone: ignored
    inputs.remove_inputs(None)


def test_remove_inputs_tolerates_a_missing_folder_and_os_errors(tmp_path, monkeypatch):
    write_config(tmp_path, tmp_path / "nowhere")
    inputs.remove_inputs(["a" * 32 + "-image.png"])
    directory = stage(tmp_path)
    name = "a" * 32 + "-image.png"
    (directory / name).write_bytes(b"x")

    def boom(self, *a, **k):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "unlink", boom)
    inputs.remove_inputs([name])
    assert inputs.sweep_stale() == 0
    assert (directory / name).exists()


def test_sweep_stale_removes_only_studio_names(tmp_path):
    directory = stage(tmp_path)
    for name in ("a" * 32 + "-image.png", "b" * 32 + "-image_3.png", "photo.png", "a" * 32 + "-image_9.png"):
        (directory / name).write_bytes(b"x")
    (directory / ("c" * 32 + "-image.png.partial")).write_bytes(b"x")
    assert inputs.sweep_stale() == 2
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        ["photo.png", "a" * 32 + "-image_9.png", "c" * 32 + "-image.png.partial"]
    )
    assert inputs.sweep_stale() == 0


def test_sweep_stale_without_a_folder_is_a_no_op(tmp_path):
    write_config(tmp_path, tmp_path / "nowhere")
    assert inputs.sweep_stale() == 0

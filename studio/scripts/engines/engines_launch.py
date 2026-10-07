#!/usr/bin/env python3
"""Resolve engines.toml and exec an engine server (oMLX or ComfyUI).

Run by omlx-launch / comfyui-launch with the user-side venv under ~/.unsloth/engines. Stdlib only
(needs Python 3.11+ for tomllib). The process is replaced via execve, so launchd supervises the real
server directly.

    engines_launch.py omlx|comfyui [--print]

--print shows the resolved argv/env additions instead of exec'ing.

A fatal precondition (missing venv, config, launcher or model, or a port already in use)
writes <home>/<engine>.fail, JSON {"reason", "ts", "count"}, then sleeps min(10 * 2^(count-1), 300)
seconds before exiting 78, so launchd's KeepAlive respawn backs off instead of spinning. The
marker is removed right before the engine is exec'd; a "healthy start" here means every
precondition passed. The tray, the engines panel and /api/engines/attached/status show it.

Environment:
    UNSLOTH_ENGINES_HOME    default ~/.unsloth/engines
    UNSLOTH_ENGINES_CONFIG  default <home>/engines.toml
    ENGINES_FAIL_NO_SLEEP   1: write the marker but skip the backoff sleep (set by the wrappers
                            on a terminal, and by tests)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(os.environ.get("UNSLOTH_ENGINES_BUNDLE") or Path(__file__).resolve().parent)
HOME = Path(os.environ.get("UNSLOTH_ENGINES_HOME") or Path.home() / ".unsloth" / "engines")
CONFIG = Path(os.environ.get("UNSLOTH_ENGINES_CONFIG") or HOME / "engines.toml")
HOMEBREW_PATHS = ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"]
DEFAULT_LOG_DIR = Path.home() / "Library" / "Logs" / "Unsloth" / "engines"


FAIL_BASE_S = 10
FAIL_CAP_S = 300
# Set once the engine is known; stays None for --print so diagnostics leave no marker.
_failing_engine: str | None = None


def fail_path(engine: str) -> Path:
    return HOME / f"{engine}.fail"


def backoff_seconds(count: int) -> int:
    return min(FAIL_BASE_S * 2 ** max(0, min(count, 16) - 1), FAIL_CAP_S)


def read_fail_count(engine: str) -> int:
    try:
        count = json.loads(fail_path(engine).read_text(encoding="utf-8")).get("count", 0)
    except (OSError, ValueError, AttributeError):
        return 0
    return count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0


def record_failure(engine: str, reason: str) -> int:
    """Write <home>/<engine>.fail atomically and return the consecutive failure count."""
    count = read_fail_count(engine) + 1
    path = fail_path(engine)
    body = json.dumps({"reason": reason, "ts": round(time.time(), 3), "count": count})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(body + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        print(f"engines_launch: could not write {path}: {exc}", file=sys.stderr)
    return count


def clear_failure(engine: str) -> None:
    try:
        fail_path(engine).unlink()
    except OSError:
        pass


def die(msg: str) -> "None":
    print(f"engines_launch: {msg}", file=sys.stderr)
    if _failing_engine:
        count = record_failure(_failing_engine, msg)
        wait = backoff_seconds(count)
        if os.environ.get("ENGINES_FAIL_NO_SLEEP") != "1":
            print(f"engines_launch: failure {count}, backing off {wait} s before exiting", file=sys.stderr)
            sys.stderr.flush()
            time.sleep(wait)
    sys.exit(78)


def port_in_use(port: int, host: str | None = None) -> bool:
    """True when something already accepts connections on the port (loopback, or the bind host)."""
    targets = ["127.0.0.1"]
    if host and host not in ("0.0.0.0", "::", "[::]", "127.0.0.1", "localhost"):
        targets.append(str(host))
    for target in targets:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((target, port)) == 0:
                return True
    return False


def port_owner(port: int) -> tuple[int, str] | None:
    """(pid, command line) of the process listening on the port, best effort (None when unknown)."""
    try:
        listing = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"], capture_output=True, text=True, timeout=2
        ).stdout
        pid = next(int(line[1:]) for line in listing.splitlines() if line[:1] == "p" and line[1:].isdigit())
        command = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, StopIteration):
        return None
    return (pid, command) if command else None


def require_free_port(port: int, label: str, host: str | None = None) -> None:
    # --print only inspects the resolved launch; it routinely runs while the engine is live.
    if _failing_engine and port_in_use(port, host):
        owner = port_owner(port)
        if owner is None:
            die(f"port {port} ({label}) is already in use; another process holds it")
        pid, command = owner
        if "StoryPressRuntime" in command:
            die(f"port {port} ({label}) is held by StoryPress's ComfyUI (pid {pid}); stop it or change [{label.lower()}] port")
        die(f"port {port} ({label}) is held by {command[:200]} (pid {pid})")


def expand(value: object) -> str:
    return os.path.expanduser(str(value))


def load_config() -> dict:
    try:
        with CONFIG.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        die(f"config not found: {CONFIG} (run studio/scripts/build-engines-mac.sh)")
    except tomllib.TOMLDecodeError as exc:
        die(f"invalid TOML in {CONFIG}: {exc}")
    return {}


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    path = env.get("PATH", "")
    for prefix in reversed(HOMEBREW_PATHS):
        if prefix not in path.split(":"):
            path = f"{prefix}:{path}" if path else prefix
    env["PATH"] = path
    return env


def omlx(cfg: dict) -> tuple[list[str], dict[str, str], str | None]:
    section = cfg.get("omlx", {})
    python = HOME / "omlx" / "bin" / "python"
    if not python.exists():
        die(f"oMLX venv missing: {python} (run studio/scripts/build-engines-mac.sh)")
    base_path = expand(section.get("base_path", "~/.omlx-tuned"))
    port = int(section.get("port", 8843))
    require_free_port(port, "oMLX", section.get("host"))
    argv = [str(python), "-m", "omlx.cli", "serve", "--base-path", base_path, "--port", str(port)]
    if section.get("host"):
        argv += ["--host", str(section["host"])]
    argv += [str(a) for a in section.get("extra_args", [])]
    env = base_env()
    env.setdefault("MallocSpaceEfficient", "1")
    for key, value in section.get("env", {}).items():
        env[str(key)] = str(value)
    return argv, env, None


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
# Folder keys written for every shared model dir; each value is a newline-separated list of
# sub-folders of base_path (ComfyUI's extra_config splits it on a newline).
MODEL_FOLDERS = {
    "checkpoints": ["checkpoints"],
    "diffusion_models": ["diffusion_models", "unet"],
    "text_encoders": ["text_encoders", "clip"],
    "vae": ["vae"],
    "loras": ["loras"],
    "clip_vision": ["clip_vision"],
    "controlnet": ["controlnet"],
    "upscale_models": ["upscale_models"],
    "embeddings": ["embeddings"],
    "latent_upscale_models": ["latent_upscale_models"],
    "model_patches": ["model_patches"],
}


def write_model_paths(path: Path, dirs: list[str]) -> list[str]:
    """Write ComfyUI's extra_model_paths YAML for read-only shared model dirs, atomically.

    Stdlib only: every scalar is a JSON-quoted string, which is valid YAML. Returns the warnings
    (a missing dir is a warning, never a failure). With no dirs the file is `{}`: an empty file
    would load as None and ComfyUI would crash iterating it.
    """
    warnings = []
    blocks = []
    for i, raw in enumerate(dirs):
        base = expand(raw)
        if not os.path.isdir(base):
            warnings.append(f"model dir not found (still listed): {base}")
        lines = [f"unsloth_shared_{i}:", f"  base_path: {json.dumps(base)}", "  is_default: false"]
        for key, subs in MODEL_FOLDERS.items():
            lines.append(f"  {key}: {json.dumps(chr(10).join(subs))}")
        blocks.append("\n".join(lines))
    body = "# Generated by engines_launch.py at every start from [comfyui] model_dirs. Do not edit.\n"
    body += "\n".join(blocks) + "\n" if blocks else "{}\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        die(f"could not write {path}: {exc}")
    return warnings


def comfyui(cfg: dict) -> tuple[list[str], dict[str, str], str | None]:
    section = cfg.get("comfyui", {})
    python = HOME / "comfyui" / "bin" / "python"
    src = HOME / "comfyui" / "src"
    if not python.exists() or not (src / "main.py").exists():
        die(f"ComfyUI venv missing: {python} and {src / 'main.py'} (run studio/scripts/build-engines-mac.sh)")
    host = str(section.get("host", "127.0.0.1"))
    if host not in LOOPBACK_HOSTS:
        die(f"[comfyui] host {host!r} is not loopback; ComfyUI has no authentication")
    port = int(section.get("port", 8844))
    require_free_port(port, "ComfyUI", host)
    data_dir = Path(expand(section["data_dir"])) if section.get("data_dir") else HOME / "comfyui-data"
    yaml_path = data_dir / "extra_model_paths.unsloth.yaml"
    if _failing_engine:  # --print never creates or writes anything
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            die(f"could not create the ComfyUI data dir {data_dir}: {exc}")
        for warning in write_model_paths(yaml_path, [str(d) for d in section.get("model_dirs", [])]):
            print(f"engines_launch: warning: {warning}", file=sys.stderr)
    argv = [
        str(python), str(src / "main.py"), "--listen", host, "--port", str(port),
        "--base-directory", str(data_dir), "--extra-model-paths-config", str(yaml_path),
        "--disable-auto-launch", *[str(a) for a in section.get("extra_args", [])],
    ]
    env = base_env()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    for key, value in section.get("env", {}).items():
        env[str(key)] = str(value)
    return argv, env, str(src)


ENGINES = {"omlx": (omlx, ("OMLX_", "Malloc")), "comfyui": (comfyui, ("PYTORCH_", "COMFY", "Malloc"))}


def main(argv: list[str]) -> None:
    args = [a for a in argv[1:] if a != "--print"]
    if len(args) != 1 or args[0] not in ENGINES:
        die("usage: engines_launch.py omlx|comfyui [--print]")
    global _failing_engine
    if "--print" not in argv:
        _failing_engine = args[0]
    cfg = load_config()
    build, prefixes = ENGINES[args[0]]
    command, env, cwd = build(cfg)
    if "--print" in argv:
        shown = {k: v for k, v in env.items() if k.startswith(prefixes)}
        print(json.dumps({"argv": command, "cwd": cwd, "env": shown}, indent=2))
        return
    clear_failure(args[0])
    if cwd:
        os.chdir(cwd)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main(sys.argv)

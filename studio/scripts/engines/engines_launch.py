#!/usr/bin/env python3
"""Resolve engines.toml and exec the oMLX server or the ds4 on-demand launcher.

Run by omlx-launch / ds4-ondemand-launch with the interpreter of the matching
user-side venv under ~/.unsloth/engines. Stdlib only (needs Python 3.11+ for
tomllib). The process is replaced via execve, so launchd supervises the real
server directly.

    engines_launch.py omlx|ds4 [--print]

--print shows the resolved argv/env additions instead of exec'ing.

A fatal precondition (missing venv, config, launcher or model, or a port already in use)
writes <home>/<engine>.fail, JSON {"reason", "ts", "count"}, then sleeps min(10 * 2^(count-1), 300)
seconds before exiting 78, so launchd's KeepAlive respawn backs off instead of spinning. The
marker is removed right before the engine is exec'd; a "healthy start" here means every
precondition passed. The tray, the engines panel and /api/engines/attached/status show it.

Environment:
    UNSLOTH_ENGINES_HOME    default ~/.unsloth/engines
    UNSLOTH_ENGINES_CONFIG  default <home>/engines.toml
    UNSLOTH_ENGINES_BUNDLE  dir holding ds4/ (default: this script's directory)
    ENGINES_FAIL_NO_SLEEP   1: write the marker but skip the backoff sleep (set by the wrappers
                            on a terminal, and by tests)
"""

from __future__ import annotations

import json
import os
import socket
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


def require_free_port(port: int, label: str, host: str | None = None) -> None:
    # --print only inspects the resolved launch; it routinely runs while the engine is live.
    if _failing_engine and port_in_use(port, host):
        die(f"port {port} ({label}) is already in use; another process holds it")


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


def ds4(cfg: dict) -> tuple[list[str], dict[str, str], str | None]:
    section = cfg.get("ds4", {})
    python = HOME / "ds4-ondemand" / "bin" / "python"
    if not python.exists():
        die(f"ds4-ondemand venv missing: {python} (run studio/scripts/build-engines-mac.sh)")
    script = HERE / "ds4" / "ds4_ondemand.py"
    if not script.exists():
        die(f"launcher not found: {script}")
    workdir = Path(expand(section.get("workdir") or HERE / "ds4"))
    binary = Path(expand(section.get("binary") or HERE / "ds4" / "ds4-server"))
    log_dir = Path(expand(section.get("log_dir") or DEFAULT_LOG_DIR))
    model = expand(section.get("model", "~/Homelab/dwarfstar/ds4flash.gguf"))
    if _failing_engine and not os.path.exists(model):
        die(f"model file missing: {model}")
    launcher_port = int(section.get("launcher_port", 8001))
    server_port = int(section.get("server_port", 8000))
    require_free_port(launcher_port, "ds4 launcher", section.get("host"))
    require_free_port(server_port, "ds4-server")
    env = base_env()
    env.update(
        {
            "DS4_ONDEMAND_HOST": str(section.get("host", "127.0.0.1")),
            "DS4_ONDEMAND_PORT": str(launcher_port),
            "DS4_SERVER_PORT": str(server_port),
            "DS4_BINARY": str(binary),
            "DS4_WORKDIR": str(workdir),
            "DS4_LOG_DIR": str(log_dir),
            "DS4_MODEL_FILE": model,
            "DS4_CTX": str(int(section.get("ctx", 131072))),
            "DS4_PREFILL_CHUNK": str(int(section.get("prefill_chunk", 1024))),
            "DS4_START_TIMEOUT": str(section.get("start_timeout", 120)),
            "DS4_IDLE_SECONDS": str(section.get("idle_seconds", 300)),
            "OMLX_BASE_URL": str(section.get("omlx_url", "http://127.0.0.1:8843")),
        }
    )
    if section.get("alias"):
        env["DS4_MODEL_ALIAS"] = str(section["alias"])
    if section.get("vision"):
        env["DS4_VISION_FILE"] = expand(section["vision"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return [str(python), str(script)], env, str(workdir)


def main(argv: list[str]) -> None:
    args = [a for a in argv[1:] if a != "--print"]
    if len(args) != 1 or args[0] not in ("omlx", "ds4"):
        die("usage: engines_launch.py omlx|ds4 [--print]")
    global _failing_engine
    if "--print" not in argv:
        _failing_engine = args[0]
    cfg = load_config()
    command, env, cwd = (omlx if args[0] == "omlx" else ds4)(cfg)
    if "--print" in argv:
        shown = {k: v for k, v in env.items() if k.startswith(("OMLX_", "DS4_", "Malloc"))}
        print(json.dumps({"argv": command, "cwd": cwd, "env": shown}, indent=2))
        return
    clear_failure(args[0])
    if cwd:
        os.chdir(cwd)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main(sys.argv)

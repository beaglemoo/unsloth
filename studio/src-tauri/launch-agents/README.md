# Engine LaunchAgents

SMAppService agent plists for the attached engines. They are only bundled by the fork
build (`../tauri.fork.conf.json`, `bundle.macOS.files`), into `Contents/Library/LaunchAgents`.

- `BundleProgram` is relative to the app bundle and points at the wrappers staged by
  `studio/scripts/build-engines-mac.sh` into `Contents/Resources/engines`.
- launchd does not expand `~`, so the wrappers redirect their own output to
  `~/Library/Logs/Unsloth/engines/` (no `StandardOutPath` here).
- The agents are registered and unregistered by `src/engine_helpers.rs` (Settings > Engines). oMLX
  (`omlx-launch`, :8843) is on by default; ComfyUI (`comfyui-launch`, :8844) is off until the user
  turns it on (`helpers` in `~/.unsloth/engines/desktop.json`). Quitting Unsloth only unregisters them
  in `with_app` mode; with `always`, pi, Claude Code and OpenCode keep using :8843.

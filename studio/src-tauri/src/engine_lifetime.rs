//! How long the bundled engines (oMLX) live, and the intent that goes with it. Behind the
//! `attached-engines` feature.
//!
//! Two values are kept in `<engines home>/desktop.json` (`~/.unsloth/engines`, or
//! `UNSLOTH_ENGINES_HOME`, resolved exactly like the wrappers and the tray do):
//!
//! - `engine_lifetime`: `with_app` (the default: the helpers run only while Unsloth runs) or
//!   `always` (they keep serving after Unsloth quits).
//! - `engines_enabled`: the master "Engines enabled" switch. It has to be remembered here
//!   because in `with_app` mode the quit path unregisters the helpers, so the SMAppService
//!   registration no longer says whether the user wants engines at the next launch.
//!
//! Why a file on the desktop side, not the backend's `attached_engines` setting: the launch
//! hook runs in Tauri `setup()` before the backend is spawned, the quit hook runs while the
//! backend is being reaped or is already gone, and the `--engine-helpers` CLI and the install
//! scripts run with no backend at all. A single file means one source of truth that all of
//! those can read. The scripts read the same file.
//!
//! At launch (`reconcile_at_launch`), with engines enabled: `with_app` brings the helpers up
//! fresh (restart when a crash left them registered, else register); `always` makes sure they
//! are registered. The quit side is `engine_quit`.

use std::path::{Path, PathBuf};

use log::{info, warn};
use serde::{Deserialize, Serialize};

use crate::engine_helpers::{self, HelperState};

pub(crate) const SETTINGS_FILE: &str = "desktop.json";

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EngineLifetime {
    /// Stop the engines when Unsloth quits.
    #[default]
    WithApp,
    /// Keep serving after Unsloth quits.
    Always,
}

impl EngineLifetime {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            EngineLifetime::WithApp => "with_app",
            EngineLifetime::Always => "always",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "with_app" => Some(EngineLifetime::WithApp),
            "always" => Some(EngineLifetime::Always),
            _ => None,
        }
    }
}

/// What the file holds. `engines_enabled` is None until the user (or the first launch after the
/// upgrade) has decided.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct DesktopEngineSettings {
    pub(crate) engines_enabled: Option<bool>,
    pub(crate) lifetime: EngineLifetime,
}

/// Anything that is not a JSON object with valid fields falls back per field: one bad value
/// must not discard the other, and a missing lifetime is `with_app`.
pub(crate) fn parse_settings(text: &str) -> DesktopEngineSettings {
    let mut settings = DesktopEngineSettings::default();
    let Ok(body) = serde_json::from_str::<serde_json::Value>(text) else {
        if !text.trim().is_empty() {
            warn!("ignoring unreadable {SETTINGS_FILE}");
        }
        return settings;
    };
    if let Some(value) = body.get("engine_lifetime") {
        match value.as_str().and_then(EngineLifetime::parse) {
            Some(lifetime) => settings.lifetime = lifetime,
            None => warn!("ignoring invalid engine_lifetime {value} in {SETTINGS_FILE}"),
        }
    }
    settings.engines_enabled = body.get("engines_enabled").and_then(|v| v.as_bool());
    settings
}

pub(crate) fn render_settings(settings: &DesktopEngineSettings) -> String {
    let mut body = serde_json::Map::new();
    if let Some(enabled) = settings.engines_enabled {
        body.insert("engines_enabled".into(), enabled.into());
    }
    body.insert("engine_lifetime".into(), settings.lifetime.as_str().into());
    let mut text = serde_json::Value::Object(body).to_string();
    text.push('\n');
    text
}

fn settings_path() -> Option<PathBuf> {
    Some(crate::engine_tray::engines_home()?.join(SETTINGS_FILE))
}

pub(crate) fn load() -> DesktopEngineSettings {
    settings_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| parse_settings(&text))
        .unwrap_or_default()
}

/// Write through a temp file and a rename, so a reader (a script, the quit hook) never sees half.
fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let dir = path.parent().ok_or("no parent directory")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let tmp = dir.join(format!("{SETTINGS_FILE}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("rename to {}: {e}", path.display())
    })
}

fn update(
    change: impl FnOnce(&mut DesktopEngineSettings),
) -> Result<DesktopEngineSettings, String> {
    let path = settings_path().ok_or("no engines home")?;
    let mut settings = std::fs::read_to_string(&path)
        .map(|text| parse_settings(&text))
        .unwrap_or_default();
    change(&mut settings);
    write_atomic(&path, &render_settings(&settings))?;
    Ok(settings)
}

pub(crate) fn save_lifetime(lifetime: EngineLifetime) -> Result<(), String> {
    update(|s| s.lifetime = lifetime).map(|_| ())
}

pub(crate) fn save_enabled(enabled: bool) -> Result<(), String> {
    update(|s| s.engines_enabled = Some(enabled)).map(|_| ())
}

/// The master switch as the app should treat it: the remembered choice, else (first run after the
/// upgrade, when only the SMAppService registration exists) whether the helpers are registered.
pub(crate) fn effective_enabled(settings: &DesktopEngineSettings, state: HelperState) -> bool {
    settings.engines_enabled.unwrap_or(matches!(
        state,
        HelperState::Enabled | HelperState::Partial | HelperState::RequiresApproval
    ))
}

/// `effective_enabled`, and when it came from the registration, remembered in the file: the next
/// unregister (the quit path) would otherwise erase the only evidence of the user's choice.
pub(crate) fn resolve_enabled(settings: &DesktopEngineSettings, state: HelperState) -> bool {
    let enabled = effective_enabled(settings, state);
    if settings.engines_enabled.is_none() && enabled {
        match save_enabled(true) {
            Ok(()) => info!(
                "engine lifetime: remembered engines_enabled=true from the helper registration"
            ),
            Err(error) => warn!("engine lifetime: could not remember engines_enabled: {error}"),
        }
    }
    enabled
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaunchPlan {
    Nothing,
    Register,
    /// Unregister, wait for launchd, register: what `--engine-helpers restart` does.
    Restart,
}

/// What launch does to the helpers.
///
/// `with_app`: registered helpers can only be a crash leftover (a clean quit unregisters), so
/// they are restarted to start from a known state; unregistered ones are registered. `always`:
/// registered stays as it is. `RequiresApproval` is left alone in both: an unregister and a
/// register would lose the pending approval, and there is nothing running to reconcile.
pub(crate) fn launch_plan(
    enabled: bool,
    lifetime: EngineLifetime,
    state: HelperState,
) -> LaunchPlan {
    if !enabled {
        return LaunchPlan::Nothing;
    }
    match (lifetime, state) {
        (_, HelperState::Unsupported | HelperState::NotFound | HelperState::RequiresApproval) => {
            LaunchPlan::Nothing
        }
        (EngineLifetime::WithApp, HelperState::Enabled | HelperState::Partial) => {
            LaunchPlan::Restart
        }
        (EngineLifetime::WithApp, HelperState::NotRegistered) => LaunchPlan::Register,
        (EngineLifetime::Always, HelperState::Enabled) => LaunchPlan::Nothing,
        (EngineLifetime::Always, HelperState::Partial | HelperState::NotRegistered) => {
            LaunchPlan::Register
        }
    }
}

/// Run the plan with the actions injected, so the decision and the call are tested together.
pub(crate) fn apply_launch_plan(plan: LaunchPlan, register: impl FnOnce(), restart: impl FnOnce()) {
    match plan {
        LaunchPlan::Nothing => {}
        LaunchPlan::Register => register(),
        LaunchPlan::Restart => restart(),
    }
}

/// The launch hook, from Tauri `setup()`. Runs on its own thread: a restart waits up to 30 s for
/// launchd and must never hold the window back.
pub(crate) fn reconcile_at_launch() {
    let spawned = std::thread::Builder::new()
        .name("engine-launch-reconcile".to_string())
        .spawn(|| {
            engine_helpers::unregister_legacy_helper_once();
            let settings = load();
            let state = engine_helpers::current_state();
            let enabled = resolve_enabled(&settings, state);
            let plan = launch_plan(enabled, settings.lifetime, state);
            info!(
                "engine lifetime at launch: {} enabled={enabled} helpers={state:?} -> {plan:?}",
                settings.lifetime.as_str()
            );
            apply_launch_plan(
                plan,
                || log_result("register", engine_helpers::enable_helpers()),
                || log_result("restart", engine_helpers::restart_helpers()),
            );
        });
    if let Err(error) = spawned {
        warn!("could not spawn the engine launch reconcile thread: {error}");
    }
}

fn log_result(what: &str, status: engine_helpers::EngineHelpersStatus) {
    match serde_json::to_string(&status) {
        Ok(json) => info!("engine helpers {what} at launch: {json}"),
        Err(error) => {
            warn!("engine helpers {what} at launch: could not encode the status: {error}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use HelperState::*;

    #[test]
    fn a_missing_or_broken_file_means_with_app_and_no_decision() {
        for text in ["", "not json", "[]", "{}"] {
            let settings = parse_settings(text);
            assert_eq!(settings.lifetime, EngineLifetime::WithApp, "{text:?}");
            assert_eq!(settings.engines_enabled, None, "{text:?}");
        }
    }

    #[test]
    fn each_field_falls_back_on_its_own() {
        let settings = parse_settings(r#"{"engines_enabled": true, "engine_lifetime": "forever"}"#);
        assert_eq!(settings.engines_enabled, Some(true));
        assert_eq!(settings.lifetime, EngineLifetime::WithApp);
        let settings = parse_settings(r#"{"engines_enabled": "yes", "engine_lifetime": "always"}"#);
        assert_eq!(settings.engines_enabled, None);
        assert_eq!(settings.lifetime, EngineLifetime::Always);
    }

    #[test]
    fn settings_round_trip_through_the_file_text() {
        for settings in [
            DesktopEngineSettings {
                engines_enabled: Some(true),
                lifetime: EngineLifetime::Always,
            },
            DesktopEngineSettings {
                engines_enabled: Some(false),
                lifetime: EngineLifetime::WithApp,
            },
            DesktopEngineSettings {
                engines_enabled: None,
                lifetime: EngineLifetime::Always,
            },
        ] {
            assert_eq!(parse_settings(&render_settings(&settings)), settings);
        }
        // the scripts grep this shape
        let text = render_settings(&DesktopEngineSettings {
            engines_enabled: Some(true),
            lifetime: EngineLifetime::Always,
        });
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["engine_lifetime"], "always");
        assert_eq!(body["engines_enabled"], true);
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn lifetime_words_are_the_ones_the_ui_and_scripts_use() {
        assert_eq!(
            EngineLifetime::parse("with_app"),
            Some(EngineLifetime::WithApp)
        );
        assert_eq!(
            EngineLifetime::parse(" always "),
            Some(EngineLifetime::Always)
        );
        assert_eq!(EngineLifetime::parse("WithApp"), None);
        assert_eq!(
            serde_json::to_string(&EngineLifetime::WithApp).unwrap(),
            "\"with_app\""
        );
        assert_eq!(EngineLifetime::default(), EngineLifetime::WithApp);
    }

    #[test]
    fn the_remembered_choice_wins_over_the_registration() {
        let said = |enabled| DesktopEngineSettings {
            engines_enabled: Some(enabled),
            ..Default::default()
        };
        assert!(effective_enabled(&said(true), NotRegistered));
        assert!(!effective_enabled(&said(false), Enabled));
    }

    #[test]
    fn without_a_choice_registered_helpers_mean_enabled() {
        let unset = DesktopEngineSettings::default();
        for (state, expected) in [
            (Enabled, true),
            (Partial, true),
            (RequiresApproval, true),
            (NotRegistered, false),
            (NotFound, false),
            (Unsupported, false),
        ] {
            assert_eq!(effective_enabled(&unset, state), expected, "{state:?}");
        }
    }

    #[test]
    fn disabled_engines_are_never_touched_at_launch() {
        for lifetime in [EngineLifetime::WithApp, EngineLifetime::Always] {
            for state in [
                Enabled,
                NotRegistered,
                Partial,
                RequiresApproval,
                NotFound,
                Unsupported,
            ] {
                assert_eq!(launch_plan(false, lifetime, state), LaunchPlan::Nothing);
            }
        }
    }

    #[test]
    fn with_app_restarts_a_crash_leftover_and_registers_otherwise() {
        let plan = |state| launch_plan(true, EngineLifetime::WithApp, state);
        assert_eq!(plan(Enabled), LaunchPlan::Restart);
        assert_eq!(plan(Partial), LaunchPlan::Restart);
        assert_eq!(plan(NotRegistered), LaunchPlan::Register);
    }

    #[test]
    fn always_only_makes_sure_the_helpers_are_registered() {
        let plan = |state| launch_plan(true, EngineLifetime::Always, state);
        assert_eq!(plan(Enabled), LaunchPlan::Nothing);
        assert_eq!(plan(NotRegistered), LaunchPlan::Register);
        assert_eq!(plan(Partial), LaunchPlan::Register);
    }

    #[test]
    fn a_pending_approval_or_a_build_without_helpers_is_left_alone() {
        for lifetime in [EngineLifetime::WithApp, EngineLifetime::Always] {
            for state in [RequiresApproval, NotFound, Unsupported] {
                assert_eq!(launch_plan(true, lifetime, state), LaunchPlan::Nothing);
            }
        }
    }

    #[test]
    fn the_plan_runs_exactly_one_action() {
        use std::cell::RefCell;
        for (plan, expected) in [
            (LaunchPlan::Nothing, vec![]),
            (LaunchPlan::Register, vec!["register"]),
            (LaunchPlan::Restart, vec!["restart"]),
        ] {
            let calls = RefCell::new(Vec::new());
            apply_launch_plan(
                plan,
                || calls.borrow_mut().push("register"),
                || calls.borrow_mut().push("restart"),
            );
            assert_eq!(*calls.borrow(), expected);
        }
    }

    #[test]
    fn the_file_is_written_atomically_into_a_new_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join(SETTINGS_FILE);
        write_atomic(&path, &render_settings(&DesktopEngineSettings::default())).unwrap();
        assert_eq!(
            parse_settings(&std::fs::read_to_string(&path).unwrap()).lifetime,
            EngineLifetime::WithApp
        );
        let entries: Vec<_> = std::fs::read_dir(path.parent().unwrap()).unwrap().collect();
        assert_eq!(entries.len(), 1, "no temp file is left behind");
    }
}

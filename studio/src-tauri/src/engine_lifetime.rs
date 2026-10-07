//! How long the bundled engines (oMLX, ComfyUI) live, and the intent that goes with it. Behind
//! the `attached-engines` feature.
//!
//! Three values are kept in `<engines home>/desktop.json` (`~/.unsloth/engines`, or
//! `UNSLOTH_ENGINES_HOME`, resolved exactly like the wrappers and the tray do):
//!
//! - `engine_lifetime`: `with_app` (the default: the helpers run only while Unsloth runs) or
//!   `always` (they keep serving after Unsloth quits). It applies to every enabled helper.
//! - `engines_enabled`: the master "Engines enabled" switch. It has to be remembered here
//!   because in `with_app` mode the quit path unregisters the helpers, so the SMAppService
//!   registration no longer says whether the user wants engines at the next launch.
//! - `helpers`: `{"omlx": bool, "comfyui": bool}`, the per-engine choice. A helper is wanted when
//!   the master switch is on and its own choice is on; a missing choice means oMLX on and ComfyUI
//!   off, so a file from before ComfyUI behaves exactly as it did.
//!
//! Why a file on the desktop side, not the backend's `attached_engines` setting: the launch
//! hook runs in Tauri `setup()` before the backend is spawned, the quit hook runs while the
//! backend is being reaped or is already gone, and the `--engine-helpers` CLI and the install
//! scripts run with no backend at all. A single file means one source of truth that all of
//! those can read. The scripts read the same file.
//!
//! At launch (`reconcile_at_launch`), with engines enabled, each helper is reconciled on its own:
//! a wanted one in `with_app` mode comes up fresh (restart when a crash left it registered, else
//! register); in `always` mode it is registered when it is not; a helper that is registered but not
//! wanted (switched off while the app was closed) is unregistered. The quit side is `engine_quit`.

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

/// The per-engine choices. None means "never chosen": `wanted` applies the default.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct HelperChoices {
    pub(crate) omlx: Option<bool>,
    pub(crate) comfyui: Option<bool>,
}

/// What the file holds. `engines_enabled` is None until the user (or the first launch after the
/// upgrade) has decided.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct DesktopEngineSettings {
    pub(crate) engines_enabled: Option<bool>,
    pub(crate) lifetime: EngineLifetime,
    pub(crate) helpers: HelperChoices,
}

/// The default for a helper nobody has chosen for: oMLX on (it was the only helper), ComfyUI off.
fn default_choice(name: &str) -> bool {
    name == "omlx"
}

impl HelperChoices {
    fn get(&self, name: &str) -> Option<bool> {
        match name {
            "omlx" => self.omlx,
            "comfyui" => self.comfyui,
            _ => None,
        }
    }

    fn set(&mut self, name: &str, enabled: bool) -> bool {
        match name {
            "omlx" => self.omlx = Some(enabled),
            "comfyui" => self.comfyui = Some(enabled),
            _ => return false,
        }
        true
    }
}

/// The helper's own choice, with the default for one never chosen. Ignores the master switch.
pub(crate) fn chosen(settings: &DesktopEngineSettings, name: &str) -> bool {
    settings
        .helpers
        .get(name)
        .unwrap_or_else(|| default_choice(name))
}

/// Whether `name` should be running: the master switch on and the helper's own choice on.
pub(crate) fn wanted(settings: &DesktopEngineSettings, name: &str, master_enabled: bool) -> bool {
    master_enabled && chosen(settings, name)
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
    if let Some(helpers) = body.get("helpers") {
        match helpers.as_object() {
            Some(map) => {
                for (name, value) in map {
                    match value.as_bool() {
                        Some(enabled) => {
                            if !settings.helpers.set(name, enabled) {
                                warn!("ignoring unknown helper {name:?} in {SETTINGS_FILE}");
                            }
                        }
                        None => warn!("ignoring invalid helpers.{name} {value} in {SETTINGS_FILE}"),
                    }
                }
            }
            None => warn!("ignoring invalid helpers {helpers} in {SETTINGS_FILE}"),
        }
    }
    settings
}

pub(crate) fn render_settings(settings: &DesktopEngineSettings) -> String {
    let mut body = serde_json::Map::new();
    if let Some(enabled) = settings.engines_enabled {
        body.insert("engines_enabled".into(), enabled.into());
    }
    body.insert("engine_lifetime".into(), settings.lifetime.as_str().into());
    let mut helpers = serde_json::Map::new();
    for name in ["omlx", "comfyui"] {
        if let Some(enabled) = settings.helpers.get(name) {
            helpers.insert(name.into(), enabled.into());
        }
    }
    if !helpers.is_empty() {
        body.insert("helpers".into(), helpers.into());
    }
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
    update_at(&path, change)
}

fn update_at(
    path: &Path,
    change: impl FnOnce(&mut DesktopEngineSettings),
) -> Result<DesktopEngineSettings, String> {
    let mut settings = std::fs::read_to_string(path)
        .map(|text| parse_settings(&text))
        .unwrap_or_default();
    change(&mut settings);
    write_atomic(path, &render_settings(&settings))?;
    Ok(settings)
}

pub(crate) fn save_lifetime(lifetime: EngineLifetime) -> Result<(), String> {
    update(|s| s.lifetime = lifetime).map(|_| ())
}

pub(crate) fn save_enabled(enabled: bool) -> Result<(), String> {
    update(|s| s.engines_enabled = Some(enabled)).map(|_| ())
}

/// Remember the per-engine choice. An unknown helper name is an error, nothing is written.
pub(crate) fn save_helper(name: &str, enabled: bool) -> Result<(), String> {
    check_helper_name(name)?;
    update(|s| {
        s.helpers.set(name, enabled);
    })
    .map(|_| ())
}

fn check_helper_name(name: &str) -> Result<(), String> {
    if matches!(name, "omlx" | "comfyui") {
        Ok(())
    } else {
        Err(format!(
            "unknown engine helper {name:?}; use omlx or comfyui"
        ))
    }
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
    /// Unregister, wait for launchd, register: what `--engine-helpers restart <name>` does.
    Restart,
    /// A helper that is registered but no longer wanted (switched off while the app was closed).
    Unregister,
}

/// What launch does to one helper (the master switch is checked by the caller: with it off every
/// helper is left alone).
///
/// `with_app`: a registered, wanted helper can only be a crash leftover (a clean quit unregisters),
/// so it is restarted to start from a known state; an unregistered one is registered. `always`:
/// registered stays as it is. An unwanted helper that is registered is unregistered, whatever the
/// lifetime. `RequiresApproval` is left alone in both: an unregister and a register would lose
/// the pending approval, and there is nothing running to reconcile.
pub(crate) fn helper_plan(
    wanted: bool,
    lifetime: EngineLifetime,
    state: HelperState,
) -> LaunchPlan {
    if !wanted {
        return match state {
            HelperState::Enabled | HelperState::Partial => LaunchPlan::Unregister,
            _ => LaunchPlan::Nothing,
        };
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

/// The plan for every helper, in `engine_helpers::HELPER_NAMES` order. Master off: nothing is touched.
pub(crate) fn launch_plans(
    settings: &DesktopEngineSettings,
    master_enabled: bool,
    states: &[(&'static str, HelperState)],
) -> Vec<(&'static str, LaunchPlan)> {
    states
        .iter()
        .map(|(name, state)| {
            let plan = if master_enabled {
                helper_plan(wanted(settings, name, true), settings.lifetime, *state)
            } else {
                LaunchPlan::Nothing
            };
            (*name, plan)
        })
        .collect()
}

/// Run the plan with the actions injected, so the decision and the call are tested together.
pub(crate) fn apply_launch_plan(
    plan: LaunchPlan,
    register: impl FnOnce(),
    restart: impl FnOnce(),
    unregister: impl FnOnce(),
) {
    match plan {
        LaunchPlan::Nothing => {}
        LaunchPlan::Register => register(),
        LaunchPlan::Restart => restart(),
        LaunchPlan::Unregister => unregister(),
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
            let states = engine_helpers::helper_states();
            let master = resolve_enabled(&settings, engine_helpers::master_state(&states));
            let plans = launch_plans(&settings, master, &states);
            info!(
                "engine lifetime at launch: {} enabled={master} helpers={}",
                settings.lifetime.as_str(),
                plans
                    .iter()
                    .zip(&states)
                    .map(|((name, plan), (_, state))| format!("{name}={state:?}->{plan:?}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            for (name, plan) in plans {
                apply_launch_plan(
                    plan,
                    || log_result(name, "register", engine_helpers::register_one(name)),
                    || log_result(name, "restart", engine_helpers::restart_one(name)),
                    || log_result(name, "unregister", engine_helpers::unregister_one(name)),
                );
            }
        });
    if let Err(error) = spawned {
        warn!("could not spawn the engine launch reconcile thread: {error}");
    }
}

fn log_result(name: &str, what: &str, result: Result<(), String>) {
    match result {
        Ok(()) => info!("engine helper {name} {what} at launch: ok"),
        Err(error) => warn!("engine helper {name} {what} at launch: {error}"),
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
                ..Default::default()
            },
            DesktopEngineSettings {
                engines_enabled: Some(false),
                lifetime: EngineLifetime::WithApp,
                ..Default::default()
            },
            DesktopEngineSettings {
                engines_enabled: None,
                lifetime: EngineLifetime::Always,
                ..Default::default()
            },
            DesktopEngineSettings {
                engines_enabled: Some(true),
                lifetime: EngineLifetime::WithApp,
                helpers: HelperChoices {
                    omlx: Some(false),
                    comfyui: Some(true),
                },
            },
            DesktopEngineSettings {
                helpers: HelperChoices {
                    omlx: None,
                    comfyui: Some(false),
                },
                ..Default::default()
            },
        ] {
            assert_eq!(parse_settings(&render_settings(&settings)), settings);
        }
        // the scripts grep this shape
        let text = render_settings(&DesktopEngineSettings {
            engines_enabled: Some(true),
            lifetime: EngineLifetime::Always,
            ..Default::default()
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
    fn a_file_from_before_comfyui_keeps_omlx_on_and_comfyui_off() {
        for text in [
            "",
            "{}",
            r#"{"engines_enabled": true, "engine_lifetime": "always"}"#,
        ] {
            let settings = parse_settings(text);
            assert_eq!(settings.helpers, HelperChoices::default(), "{text:?}");
            assert!(wanted(&settings, "omlx", true), "{text:?}");
            assert!(!wanted(&settings, "comfyui", true), "{text:?}");
        }
    }

    #[test]
    fn a_helper_is_wanted_only_with_the_master_switch_and_its_own_choice() {
        let settings = parse_settings(r#"{"helpers": {"omlx": false, "comfyui": true}}"#);
        assert!(!wanted(&settings, "omlx", true));
        assert!(wanted(&settings, "comfyui", true));
        assert!(!wanted(&settings, "comfyui", false));
        assert!(!wanted(&settings, "omlx", false));
        assert!(!wanted(&settings, "unknown", true));
        assert!(chosen(&settings, "comfyui"));
    }

    #[test]
    fn helpers_parse_field_by_field_and_bad_values_fall_back() {
        let settings = parse_settings(r#"{"helpers": {"omlx": "yes", "comfyui": true}}"#);
        assert_eq!(settings.helpers.omlx, None);
        assert_eq!(settings.helpers.comfyui, Some(true));
        let settings = parse_settings(r#"{"helpers": {"omlx": false, "extra": true}}"#);
        assert_eq!(settings.helpers.omlx, Some(false));
        assert_eq!(settings.helpers.comfyui, None);
        for text in [
            r#"{"helpers": []}"#,
            r#"{"helpers": "on"}"#,
            r#"{"helpers": null}"#,
        ] {
            assert_eq!(
                parse_settings(text).helpers,
                HelperChoices::default(),
                "{text}"
            );
        }
        // a broken helpers value does not discard the other fields
        let settings = parse_settings(
            r#"{"engines_enabled": true, "engine_lifetime": "always", "helpers": 3}"#,
        );
        assert_eq!(settings.engines_enabled, Some(true));
        assert_eq!(settings.lifetime, EngineLifetime::Always);
    }

    #[test]
    fn the_helpers_key_is_written_only_once_something_is_set() {
        let text = render_settings(&DesktopEngineSettings::default());
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(body.get("helpers").is_none());
        let text = render_settings(&DesktopEngineSettings {
            helpers: HelperChoices {
                omlx: None,
                comfyui: Some(true),
            },
            ..Default::default()
        });
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["helpers"], serde_json::json!({"comfyui": true}));
    }

    #[test]
    fn saving_a_helper_choice_keeps_the_other_fields_and_rejects_unknown_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SETTINGS_FILE);
        std::fs::write(
            &path,
            r#"{"engines_enabled": true, "engine_lifetime": "always"}"#,
        )
        .unwrap();
        update_at(&path, |s| {
            s.helpers.set("comfyui", true);
        })
        .unwrap();
        let settings = update_at(&path, |s| {
            s.helpers.set("omlx", false);
        })
        .unwrap();
        assert!(check_helper_name("nope").is_err());
        assert!(check_helper_name("comfyui").is_ok());
        let reread = parse_settings(&std::fs::read_to_string(&path).unwrap());
        assert_eq!(reread, settings);
        assert_eq!(reread.engines_enabled, Some(true));
        assert_eq!(reread.lifetime, EngineLifetime::Always);
        assert_eq!(reread.helpers.comfyui, Some(true));
        assert_eq!(reread.helpers.omlx, Some(false));
    }

    #[test]
    fn a_wanted_helper_in_with_app_restarts_a_crash_leftover_and_registers_otherwise() {
        let plan = |state| helper_plan(true, EngineLifetime::WithApp, state);
        assert_eq!(plan(Enabled), LaunchPlan::Restart);
        assert_eq!(plan(Partial), LaunchPlan::Restart);
        assert_eq!(plan(NotRegistered), LaunchPlan::Register);
    }

    #[test]
    fn a_wanted_helper_in_always_is_only_registered_when_it_is_not() {
        let plan = |state| helper_plan(true, EngineLifetime::Always, state);
        assert_eq!(plan(Enabled), LaunchPlan::Nothing);
        assert_eq!(plan(NotRegistered), LaunchPlan::Register);
        assert_eq!(plan(Partial), LaunchPlan::Register);
    }

    #[test]
    fn an_unwanted_helper_that_is_registered_is_unregistered_in_any_lifetime() {
        for lifetime in [EngineLifetime::WithApp, EngineLifetime::Always] {
            assert_eq!(
                helper_plan(false, lifetime, Enabled),
                LaunchPlan::Unregister
            );
            assert_eq!(
                helper_plan(false, lifetime, Partial),
                LaunchPlan::Unregister
            );
            for state in [NotRegistered, RequiresApproval, NotFound, Unsupported] {
                assert_eq!(
                    helper_plan(false, lifetime, state),
                    LaunchPlan::Nothing,
                    "{state:?}"
                );
            }
        }
    }

    #[test]
    fn a_pending_approval_or_a_build_without_helpers_is_left_alone() {
        for lifetime in [EngineLifetime::WithApp, EngineLifetime::Always] {
            for state in [RequiresApproval, NotFound, Unsupported] {
                assert_eq!(helper_plan(true, lifetime, state), LaunchPlan::Nothing);
            }
        }
    }

    #[test]
    fn with_the_master_switch_off_no_helper_is_touched_at_launch() {
        let settings = parse_settings(r#"{"helpers": {"comfyui": true}}"#);
        for lifetime in [EngineLifetime::WithApp, EngineLifetime::Always] {
            let settings = DesktopEngineSettings {
                lifetime,
                ..settings
            };
            for state in [
                Enabled,
                NotRegistered,
                Partial,
                RequiresApproval,
                NotFound,
                Unsupported,
            ] {
                let plans = launch_plans(&settings, false, &[("omlx", state), ("comfyui", state)]);
                assert!(plans.iter().all(|(_, plan)| *plan == LaunchPlan::Nothing));
            }
        }
    }

    #[test]
    fn launch_plans_follow_each_helper_on_its_own() {
        // omlx wanted and registered, comfyui off by default and registered: a leftover to remove
        let settings = parse_settings(r#"{"engines_enabled": true}"#);
        let plans = launch_plans(&settings, true, &[("omlx", Enabled), ("comfyui", Enabled)]);
        assert_eq!(
            plans,
            vec![
                ("omlx", LaunchPlan::Restart),
                ("comfyui", LaunchPlan::Unregister)
            ]
        );
        // both wanted, in always mode, only comfyui missing
        let settings =
            parse_settings(r#"{"engine_lifetime": "always", "helpers": {"comfyui": true}}"#);
        let plans = launch_plans(
            &settings,
            true,
            &[("omlx", Enabled), ("comfyui", NotRegistered)],
        );
        assert_eq!(
            plans,
            vec![
                ("omlx", LaunchPlan::Nothing),
                ("comfyui", LaunchPlan::Register)
            ]
        );
        // oMLX switched off, ComfyUI on, nothing registered
        let settings = parse_settings(r#"{"helpers": {"omlx": false, "comfyui": true}}"#);
        let plans = launch_plans(
            &settings,
            true,
            &[("omlx", NotRegistered), ("comfyui", NotRegistered)],
        );
        assert_eq!(
            plans,
            vec![
                ("omlx", LaunchPlan::Nothing),
                ("comfyui", LaunchPlan::Register)
            ]
        );
    }

    #[test]
    fn the_plan_runs_exactly_one_action() {
        use std::cell::RefCell;
        for (plan, expected) in [
            (LaunchPlan::Nothing, vec![]),
            (LaunchPlan::Register, vec!["register"]),
            (LaunchPlan::Restart, vec!["restart"]),
            (LaunchPlan::Unregister, vec!["unregister"]),
        ] {
            let calls = RefCell::new(Vec::new());
            apply_launch_plan(
                plan,
                || calls.borrow_mut().push("register"),
                || calls.borrow_mut().push("restart"),
                || calls.borrow_mut().push("unregister"),
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

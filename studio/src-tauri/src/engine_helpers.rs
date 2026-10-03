//! Registration of the bundled oMLX and ds4 helper agents with SMAppService.
//!
//! The agents ship as `Contents/Library/LaunchAgents/ai.unsloth.studio.{omlx,ds4}.plist` and
//! point (BundleProgram) at the wrappers in `Contents/Resources/engines`. They are
//! registered when the user turns on "Background engines" and they KEEP RUNNING after Unsloth
//! quits, because other clients (pi, Claude Code, OpenCode) use the engines. Nothing in the
//! quit, update or repair paths may call `disable`; only the explicit user action does.

use serde::Serialize;

pub(crate) const OMLX_PLIST: &str = "ai.unsloth.studio.omlx.plist";
pub(crate) const DS4_PLIST: &str = "ai.unsloth.studio.ds4.plist";
const HELPERS: [(&str, &str); 2] = [("omlx", OMLX_PLIST), ("ds4", DS4_PLIST)];

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HelperState {
    NotRegistered,
    Enabled,
    RequiresApproval,
    NotFound,
    Unsupported,
    /// Aggregate only: some helpers enabled, some not.
    Partial,
}

#[derive(Serialize, Clone, Debug)]
pub(crate) struct HelperStatus {
    name: &'static str,
    plist: &'static str,
    state: HelperState,
}

#[derive(Serialize, Clone, Debug)]
pub(crate) struct EngineHelpersStatus {
    /// False when SMAppService is unavailable (not macOS).
    supported: bool,
    state: HelperState,
    helpers: Vec<HelperStatus>,
    error: Option<String>,
}

/// One state for the toggle. Approval and missing-plist problems outrank the rest because
/// they need a user action; a mix of enabled and not registered is `Partial`.
fn aggregate(states: &[HelperState]) -> HelperState {
    if states.is_empty() {
        return HelperState::NotRegistered;
    }
    if states.contains(&HelperState::Unsupported) {
        return HelperState::Unsupported;
    }
    if states.contains(&HelperState::NotFound) {
        return HelperState::NotFound;
    }
    if states.contains(&HelperState::RequiresApproval) {
        return HelperState::RequiresApproval;
    }
    if states.iter().all(|s| *s == HelperState::Enabled) {
        return HelperState::Enabled;
    }
    if states.iter().all(|s| *s == HelperState::NotRegistered) {
        return HelperState::NotRegistered;
    }
    HelperState::Partial
}

fn snapshot(error: Option<String>) -> EngineHelpersStatus {
    let helpers: Vec<HelperStatus> = HELPERS
        .iter()
        .map(|(name, plist)| HelperStatus {
            name,
            plist,
            state: sm::status(plist),
        })
        .collect();
    let states: Vec<HelperState> = helpers.iter().map(|h| h.state).collect();
    EngineHelpersStatus {
        supported: sm::SUPPORTED,
        state: aggregate(&states),
        helpers,
        error,
    }
}

/// Register every helper. A failure on one is reported but does not stop the others, so the
/// toggle never leaves the pair half-registered because of a single error.
fn enable() -> EngineHelpersStatus {
    let mut errors = Vec::new();
    for (name, plist) in HELPERS {
        if let Err(error) = sm::register(plist) {
            errors.push(format!("{name}: {error}"));
        }
    }
    snapshot((!errors.is_empty()).then(|| errors.join("; ")))
}

fn disable() -> EngineHelpersStatus {
    let mut errors = Vec::new();
    for (name, plist) in HELPERS {
        if let Err(error) = sm::unregister(plist) {
            errors.push(format!("{name}: {error}"));
        }
    }
    snapshot((!errors.is_empty()).then(|| errors.join("; ")))
}

pub(crate) fn current_state() -> HelperState {
    snapshot(None).state
}

/// Open System Settings > Login Items, where the user approves or removes the helpers.
pub(crate) fn open_login_items_settings() {
    sm::open_login_items_settings();
}

#[tauri::command]
pub(crate) async fn engine_helpers_status() -> EngineHelpersStatus {
    snapshot(None)
}

#[tauri::command]
pub(crate) async fn engine_helpers_enable() -> EngineHelpersStatus {
    enable()
}

#[tauri::command]
pub(crate) async fn engine_helpers_disable() -> EngineHelpersStatus {
    disable()
}

#[cfg(target_os = "macos")]
mod sm {
    use super::HelperState;
    use objc2_foundation::{NSError, NSString};
    use objc2_service_management::{SMAppService, SMAppServiceStatus};

    pub(super) const SUPPORTED: bool = true;

    fn describe(error: &NSError) -> String {
        error.localizedDescription().to_string()
    }

    pub(super) fn status(plist: &str) -> HelperState {
        let name = NSString::from_str(plist);
        // SAFETY: plain Objective-C message sends on a retained SMAppService.
        let status = unsafe { SMAppService::agentServiceWithPlistName(&name).status() };
        if status == SMAppServiceStatus::Enabled {
            HelperState::Enabled
        } else if status == SMAppServiceStatus::RequiresApproval {
            HelperState::RequiresApproval
        } else if status == SMAppServiceStatus::NotFound {
            HelperState::NotFound
        } else {
            HelperState::NotRegistered
        }
    }

    pub(super) fn register(plist: &str) -> Result<(), String> {
        let name = NSString::from_str(plist);
        // SAFETY: as above.
        unsafe { SMAppService::agentServiceWithPlistName(&name).registerAndReturnError() }
            .map_err(|e| describe(&e))
    }

    pub(super) fn unregister(plist: &str) -> Result<(), String> {
        let name = NSString::from_str(plist);
        // SAFETY: as above.
        unsafe { SMAppService::agentServiceWithPlistName(&name).unregisterAndReturnError() }
            .map_err(|e| describe(&e))
    }

    pub(super) fn open_login_items_settings() {
        // SAFETY: class method with no arguments.
        unsafe { SMAppService::openSystemSettingsLoginItems() };
    }
}

#[cfg(not(target_os = "macos"))]
mod sm {
    use super::HelperState;

    pub(super) const SUPPORTED: bool = false;

    pub(super) fn status(_plist: &str) -> HelperState {
        HelperState::Unsupported
    }

    pub(super) fn register(_plist: &str) -> Result<(), String> {
        Err("background engines need macOS".into())
    }

    pub(super) fn unregister(_plist: &str) -> Result<(), String> {
        Err("background engines need macOS".into())
    }

    pub(super) fn open_login_items_settings() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMLX: &str = include_str!("../launch-agents/ai.unsloth.studio.omlx.plist");
    const DS4: &str = include_str!("../launch-agents/ai.unsloth.studio.ds4.plist");
    const FORK_CONF: &str = include_str!("../tauri.fork.conf.json");

    #[test]
    fn aggregate_orders_states() {
        use HelperState::*;
        assert_eq!(aggregate(&[]), NotRegistered);
        assert_eq!(aggregate(&[Enabled, Enabled]), Enabled);
        assert_eq!(aggregate(&[NotRegistered, NotRegistered]), NotRegistered);
        assert_eq!(aggregate(&[Enabled, NotRegistered]), Partial);
        assert_eq!(aggregate(&[Enabled, RequiresApproval]), RequiresApproval);
        assert_eq!(aggregate(&[RequiresApproval, NotFound]), NotFound);
        assert_eq!(aggregate(&[Unsupported, Unsupported]), Unsupported);
    }

    #[test]
    fn state_serializes_as_snake_case() {
        let json = serde_json::to_string(&HelperState::RequiresApproval).unwrap();
        assert_eq!(json, "\"requires_approval\"");
    }

    #[test]
    fn plists_match_their_names_and_wrappers() {
        for (plist_name, text, wrapper) in [
            (OMLX_PLIST, OMLX, "omlx-launch"),
            (DS4_PLIST, DS4, "ds4-ondemand-launch"),
        ] {
            let label = plist_name.trim_end_matches(".plist");
            assert!(
                text.contains(&format!("<string>{label}</string>")),
                "{label} label"
            );
            assert!(
                text.contains(&format!(
                    "<key>BundleProgram</key>\n\t<string>Contents/Resources/engines/{wrapper}</string>"
                )),
                "{label} BundleProgram"
            );
            assert!(
                text.contains("<key>KeepAlive</key>\n\t<true/>"),
                "{label} KeepAlive"
            );
            assert!(
                text.contains("<key>RunAtLoad</key>\n\t<true/>"),
                "{label} RunAtLoad"
            );
            assert!(
                text.contains("<key>ProcessType</key>\n\t<string>Interactive</string>"),
                "{label} ProcessType"
            );
            assert!(
                text.contains("<key>ExitTimeOut</key>\n\t<integer>120</integer>"),
                "{label} ExitTimeOut"
            );
            assert!(
                !text.contains("StandardOutPath"),
                "{label} must not use ~ paths"
            );
        }
    }

    #[test]
    fn fork_config_bundles_both_plists_and_the_engines_dir() {
        let conf: serde_json::Value = serde_json::from_str(FORK_CONF).unwrap();
        let files = &conf["bundle"]["macOS"]["files"];
        for plist in [OMLX_PLIST, DS4_PLIST] {
            let key = format!("Library/LaunchAgents/{plist}");
            assert_eq!(files[key.as_str()], format!("launch-agents/{plist}"));
        }
        assert_eq!(conf["bundle"]["resources"]["engines-staging/"], "engines/");
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unsupported_platforms_report_it() {
        let status = snapshot(None);
        assert!(!status.supported);
        assert_eq!(status.state, HelperState::Unsupported);
    }
}

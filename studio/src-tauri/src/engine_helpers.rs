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

/// Where a bundled launch agent plist lives, resolved from the running executable
/// (`<bundle>/Contents/MacOS/<exe>` -> `<bundle>/Contents/Library/LaunchAgents/<plist>`).
fn bundled_plist_path(exe: &std::path::Path, plist: &str) -> Option<std::path::PathBuf> {
    let contents = exe.parent()?.parent()?;
    Some(contents.join("Library").join("LaunchAgents").join(plist))
}

/// The tray polls the state every few seconds; say once per plist what NotFound was mapped to.
fn log_not_found_once(plist: &str, bundled: bool, resolved: HelperState) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let Ok(mut seen) = SEEN.lock() else { return };
    if seen.iter().any(|p| p == plist) {
        return;
    }
    seen.push(plist.to_string());
    log::info!(
        "engine helper {plist}: SMAppService status NotFound (3), plist bundled={bundled}, reported as {resolved:?}"
    );
}

fn plist_is_bundled(plist: &str) -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|exe| bundled_plist_path(&exe, plist))
        .is_some_and(|path| path.is_file())
}

/// macOS answers `NotFound` both for a plist that is not in the bundle and for a bundled
/// agent that was never registered (BTM parses the plist but has no record). Only the
/// first means "this build has no helpers", so the bundle decides.
fn resolve_not_found(bundled: bool) -> HelperState {
    if bundled {
        HelperState::NotRegistered
    } else {
        HelperState::NotFound
    }
}

/// One state for the toggle. Approval and unsupported platforms outrank the rest because
/// they need a user action; a mix of enabled and not registered is `Partial`. `NotFound`
/// (not bundled) ranks lowest, so one missing plist never hides what the others report.
fn aggregate(states: &[HelperState]) -> HelperState {
    if states.is_empty() {
        return HelperState::NotRegistered;
    }
    if states.contains(&HelperState::Unsupported) {
        return HelperState::Unsupported;
    }
    if states.contains(&HelperState::RequiresApproval) {
        return HelperState::RequiresApproval;
    }
    if states.iter().all(|s| *s == HelperState::NotFound) {
        return HelperState::NotFound;
    }
    let present: Vec<HelperState> = states
        .iter()
        .copied()
        .filter(|s| *s != HelperState::NotFound)
        .collect();
    if present.iter().all(|s| *s == HelperState::Enabled) && present.len() == states.len() {
        return HelperState::Enabled;
    }
    if present.iter().all(|s| *s == HelperState::NotRegistered) {
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
            let bundled = super::plist_is_bundled(plist);
            let resolved = super::resolve_not_found(bundled);
            super::log_not_found_once(plist, bundled, resolved);
            resolved
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
        assert_eq!(aggregate(&[RequiresApproval, NotFound]), RequiresApproval);
        assert_eq!(aggregate(&[Unsupported, Unsupported]), Unsupported);
    }

    #[test]
    fn one_missing_plist_does_not_mask_the_others() {
        use HelperState::*;
        assert_eq!(aggregate(&[NotFound, NotFound]), NotFound);
        assert_eq!(aggregate(&[NotFound, NotRegistered]), NotRegistered);
        assert_eq!(aggregate(&[NotRegistered, NotFound]), NotRegistered);
        assert_eq!(aggregate(&[NotFound, Enabled]), Partial);
        assert_eq!(aggregate(&[Enabled, NotFound]), Partial);
        assert_eq!(aggregate(&[Unsupported, NotFound]), Unsupported);
    }

    #[test]
    fn unregistered_bundled_helpers_are_not_registered_rather_than_not_found() {
        assert_eq!(resolve_not_found(true), HelperState::NotRegistered);
        assert_eq!(resolve_not_found(false), HelperState::NotFound);
        // What macOS reports for the fork build before Background engines is turned on.
        assert_eq!(
            aggregate(&[resolve_not_found(true), resolve_not_found(true)]),
            HelperState::NotRegistered
        );
    }

    #[test]
    fn bundled_plist_path_is_resolved_from_the_executable() {
        let exe = std::path::Path::new("/Applications/Unsloth.app/Contents/MacOS/unsloth-studio");
        assert_eq!(
            bundled_plist_path(exe, OMLX_PLIST).unwrap(),
            std::path::Path::new(
                "/Applications/Unsloth.app/Contents/Library/LaunchAgents/ai.unsloth.studio.omlx.plist"
            )
        );
        assert!(bundled_plist_path(std::path::Path::new("unsloth-studio"), OMLX_PLIST).is_none());
    }

    #[test]
    fn plist_presence_follows_the_bundle_layout() {
        let root = std::env::temp_dir().join(format!("unsloth-helpers-{}", std::process::id()));
        let agents = root.join("Contents/Library/LaunchAgents");
        std::fs::create_dir_all(&agents).unwrap();
        std::fs::create_dir_all(root.join("Contents/MacOS")).unwrap();
        std::fs::write(agents.join(OMLX_PLIST), "x").unwrap();
        let exe = root.join("Contents/MacOS/unsloth-studio");
        let exists = |plist| bundled_plist_path(&exe, plist).is_some_and(|p| p.is_file());
        assert!(exists(OMLX_PLIST));
        assert!(!exists(DS4_PLIST));
        std::fs::remove_dir_all(&root).unwrap();
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
                text.contains("<key>ThrottleInterval</key>\n\t<integer>10</integer>"),
                "{label} ThrottleInterval"
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

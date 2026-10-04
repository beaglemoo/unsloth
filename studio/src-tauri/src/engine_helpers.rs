//! Registration of the bundled oMLX and ds4 helper agents with SMAppService.
//!
//! The agents ship as `Contents/Library/LaunchAgents/ai.unsloth.studio.{omlx,ds4}.plist` and
//! point (BundleProgram) at the wrappers in `Contents/Resources/engines`. They are
//! registered when the user turns on "Engines enabled" (the old "Background engines" switch).
//! What happens at quit follows the engine lifetime (`engine_lifetime`): `always` leaves them
//! running because other clients (pi, Claude Code, OpenCode) use the engines, so nothing in the
//! quit, update or repair paths may call `disable` then; `with_app` (the default) is the one
//! exception, where the real quit path (`engine_quit`) unloads the models, stops ds4 and calls
//! `disable` once the engines are idle.

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
    /// The master "Engines enabled" choice as `engine_lifetime` resolves it.
    engines_enabled: bool,
    /// `with_app` or `always`.
    engine_lifetime: crate::engine_lifetime::EngineLifetime,
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
    let state = aggregate(&states);
    let settings = crate::engine_lifetime::load();
    EngineHelpersStatus {
        supported: sm::SUPPORTED,
        state,
        helpers,
        error,
        engines_enabled: crate::engine_lifetime::effective_enabled(&settings, state),
        engine_lifetime: settings.lifetime,
    }
}

/// Register every helper. A failure on one is reported but does not stop the others, so the
/// toggle never leaves the pair half-registered because of a single error.
pub(crate) fn enable() -> EngineHelpersStatus {
    let mut errors = Vec::new();
    for (name, plist) in HELPERS {
        if let Err(error) = sm::register(plist) {
            errors.push(format!("{name}: {error}"));
        }
    }
    snapshot((!errors.is_empty()).then(|| errors.join("; ")))
}

pub(crate) fn disable() -> EngineHelpersStatus {
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

/// Register the helpers (the launch hook and the toggle share it). Does not touch the saved choice.
pub(crate) fn enable_helpers() -> EngineHelpersStatus {
    enable()
}

/// `restart` with the real launchd poll, for the launch hook.
pub(crate) fn restart_helpers() -> EngineHelpersStatus {
    restart(
        &launchd_has_label,
        &|| std::thread::sleep(RESTART_POLL_INTERVAL),
        RESTART_MAX_POLLS,
    )
}

fn remember_enabled(enabled: bool) {
    if let Err(error) = crate::engine_lifetime::save_enabled(enabled) {
        log::warn!("could not save engines_enabled={enabled}: {error}");
    }
}

/// The toggle: remember the choice, then register. The CLI's `register` does not remember
/// anything, so a script's transient stop and start never changes what the user chose.
#[tauri::command]
pub(crate) async fn engine_helpers_enable() -> EngineHelpersStatus {
    remember_enabled(true);
    enable()
}

#[tauri::command]
pub(crate) async fn engine_helpers_disable() -> EngineHelpersStatus {
    remember_enabled(false);
    disable()
}

/// Save the engine lifetime (`with_app` or `always`) and report the status with it applied.
#[tauri::command]
pub(crate) async fn engine_lifetime_set(lifetime: String) -> Result<EngineHelpersStatus, String> {
    let lifetime = crate::engine_lifetime::EngineLifetime::parse(&lifetime)
        .ok_or_else(|| format!("unknown engine lifetime {lifetime:?}; use with_app or always"))?;
    crate::engine_lifetime::save_lifetime(lifetime)?;
    Ok(snapshot(None))
}

// --- Headless CLI ------------------------------------------------------------------------------
//
// `unsloth-studio --engine-helpers <status|register|unregister|restart>` runs the same SMAppService
// calls as the Settings toggle, with no window. macOS refuses `launchctl bootstrap` of a bundled
// plist ("Bootstrap failed: 5"), so a restart outside the UI has to go through SMAppService, and
// SMAppService only answers a process that is the app's own executable. main() calls `cli_main`
// as its first statement, before logging, the PATH fix, the single-instance plugin, the backend
// or the tray, so it runs beside a live GUI instance without touching it.

pub(crate) const CLI_FLAG: &str = "--engine-helpers";

/// How long `restart` waits for launchd to drop both labels: 60 polls, 500 ms apart.
const RESTART_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);
const RESTART_MAX_POLLS: u32 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CliCommand {
    Status,
    Register,
    Unregister,
    Restart,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CliParse {
    /// Not an `--engine-helpers` invocation: start the app normally.
    NotCli,
    Run(CliCommand),
    /// The flag with a missing or unknown command.
    Usage(String),
}

/// `args` excludes the program name. The flag must come first, so an ordinary launch is never
/// mistaken for the CLI.
pub(crate) fn parse_cli_args(args: &[String]) -> CliParse {
    if args.first().map(String::as_str) != Some(CLI_FLAG) {
        return CliParse::NotCli;
    }
    match args.get(1).map(String::as_str) {
        Some("status") => CliParse::Run(CliCommand::Status),
        Some("register") => CliParse::Run(CliCommand::Register),
        Some("unregister") => CliParse::Run(CliCommand::Unregister),
        Some("restart") => CliParse::Run(CliCommand::Restart),
        Some(other) => CliParse::Usage(format!(
            "unknown command {other:?}; use status, register, unregister or restart"
        )),
        None => CliParse::Usage("missing command; use status, register, unregister or restart".into()),
    }
}

fn helper_labels() -> Vec<String> {
    HELPERS
        .iter()
        .map(|(_, plist)| plist.trim_end_matches(".plist").to_string())
        .collect()
}

/// Whether launchd still has `label` in this user's GUI domain.
#[cfg(unix)]
fn launchd_has_label(label: &str) -> bool {
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    std::process::Command::new("/bin/launchctl")
        .arg("print")
        .arg(format!("gui/{uid}/{label}"))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(not(unix))]
fn launchd_has_label(_label: &str) -> bool {
    false
}

/// Poll until none of `labels` is loaded. False when `max_polls` checks all still saw one; the
/// clock is the caller's `sleep`, so a test passes a no-op.
fn wait_labels_gone(
    labels: &[String],
    is_loaded: &dyn Fn(&str) -> bool,
    sleep: &dyn Fn(),
    max_polls: u32,
) -> bool {
    for poll in 0..max_polls {
        if !labels.iter().any(|label| is_loaded(label)) {
            return true;
        }
        if poll + 1 < max_polls {
            sleep();
        }
    }
    !labels.iter().any(|label| is_loaded(label))
}

/// Unregister, wait for launchd to drop the labels, register. A wait that times out still
/// registers, so the helpers are never left unregistered, and reports the timeout as the error.
fn restart(
    is_loaded: &dyn Fn(&str) -> bool,
    sleep: &dyn Fn(),
    max_polls: u32,
) -> EngineHelpersStatus {
    let unregistered = disable();
    let labels = helper_labels();
    let gone = wait_labels_gone(&labels, is_loaded, sleep, max_polls);
    let mut status = enable();
    let mut errors = Vec::new();
    if !gone {
        errors.push(format!(
            "launchd still had {} after {} s; registered anyway",
            labels.join(" and "),
            (RESTART_POLL_INTERVAL * max_polls).as_secs()
        ));
    }
    if let Some(error) = status.error.take() {
        errors.push(error);
        // The unregister failure explains a register failure; alone it is moot, since the
        // helpers ended up registered.
        if let Some(earlier) = unregistered.error {
            errors.push(format!("unregister: {earlier}"));
        }
    }
    status.error = (!errors.is_empty()).then(|| errors.join("; "));
    status
}

/// The exit code for a finished command: 0 only when it worked on a platform that has helpers.
fn exit_code(status: &EngineHelpersStatus) -> i32 {
    if status.supported && status.error.is_none() {
        0
    } else {
        1
    }
}

/// Run the CLI when the arguments ask for it. `None` means "not the CLI: start the app".
/// Prints the status as one line of JSON on stdout and returns the exit code.
pub(crate) fn cli_main(args: &[String]) -> Option<i32> {
    let command = match parse_cli_args(args) {
        CliParse::NotCli => return None,
        CliParse::Usage(message) => {
            eprintln!("unsloth-studio {CLI_FLAG}: {message}");
            return Some(2);
        }
        CliParse::Run(command) => command,
    };
    let status = if !sm::SUPPORTED {
        snapshot(Some("background engines need macOS".into()))
    } else {
        match command {
            CliCommand::Status => snapshot(None),
            CliCommand::Register => enable(),
            CliCommand::Unregister => disable(),
            CliCommand::Restart => restart_helpers(),
        }
    };
    match serde_json::to_string(&status) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("unsloth-studio {CLI_FLAG}: could not encode the status: {error}");
            return Some(1);
        }
    }
    Some(exit_code(&status))
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


    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn cli_args_select_a_command() {
        assert_eq!(parse_cli_args(&args(&[])), CliParse::NotCli);
        assert_eq!(parse_cli_args(&args(&["--hidden"])), CliParse::NotCli);
        // an ordinary launch that merely mentions the flag later is not the CLI
        assert_eq!(parse_cli_args(&args(&["--hidden", CLI_FLAG, "status"])), CliParse::NotCli);
        for (word, command) in [
            ("status", CliCommand::Status),
            ("register", CliCommand::Register),
            ("unregister", CliCommand::Unregister),
            ("restart", CliCommand::Restart),
        ] {
            assert_eq!(
                parse_cli_args(&args(&[CLI_FLAG, word])),
                CliParse::Run(command)
            );
        }
    }

    #[test]
    fn cli_args_reject_a_missing_or_unknown_command() {
        assert!(matches!(parse_cli_args(&args(&[CLI_FLAG])), CliParse::Usage(m) if m.contains("missing")));
        assert!(matches!(
            parse_cli_args(&args(&[CLI_FLAG, "bootstrap"])),
            CliParse::Usage(m) if m.contains("bootstrap")
        ));
        // a usage error is not a start of the GUI
        assert_eq!(cli_main(&args(&[CLI_FLAG, "bogus"])), Some(2));
        assert_eq!(cli_main(&args(&["--hidden"])), None);
    }

    #[test]
    fn helper_labels_follow_the_plist_names() {
        assert_eq!(
            helper_labels(),
            vec!["ai.unsloth.studio.omlx".to_string(), "ai.unsloth.studio.ds4".to_string()]
        );
    }

    #[test]
    fn the_wait_returns_as_soon_as_launchd_drops_both_labels() {
        use std::cell::Cell;
        let labels = helper_labels();
        let checks = Cell::new(0u32);
        let sleeps = Cell::new(0u32);
        // a loaded label ends a round at once, so three loaded checks are three rounds, then gone
        let is_loaded = |_: &str| {
            checks.set(checks.get() + 1);
            checks.get() <= 3
        };
        let sleep = || sleeps.set(sleeps.get() + 1);
        assert!(wait_labels_gone(&labels, &is_loaded, &sleep, 60));
        assert_eq!(sleeps.get(), 3);
    }

    #[test]
    fn the_wait_is_immediate_when_nothing_is_loaded() {
        use std::cell::Cell;
        let sleeps = Cell::new(0u32);
        assert!(wait_labels_gone(&helper_labels(), &|_| false, &|| sleeps.set(sleeps.get() + 1), 60));
        assert_eq!(sleeps.get(), 0);
    }

    #[test]
    fn the_wait_gives_up_after_the_poll_budget() {
        use std::cell::Cell;
        let sleeps = Cell::new(0u32);
        // one label never leaves
        let stuck = |label: &str| label.ends_with(".ds4");
        assert!(!wait_labels_gone(&helper_labels(), &stuck, &|| sleeps.set(sleeps.get() + 1), 5));
        assert_eq!(sleeps.get(), 4);
    }

    #[test]
    fn the_restart_budget_is_thirty_seconds() {
        assert_eq!((RESTART_POLL_INTERVAL * RESTART_MAX_POLLS).as_secs(), 30);
    }

    #[test]
    fn exit_codes_follow_support_and_errors() {
        let ok = EngineHelpersStatus {
            supported: true,
            state: HelperState::Enabled,
            helpers: Vec::new(),
            error: None,
            engines_enabled: true,
            engine_lifetime: crate::engine_lifetime::EngineLifetime::WithApp,
        };
        assert_eq!(exit_code(&ok), 0);
        let failed = EngineHelpersStatus { error: Some("boom".into()), ..ok.clone() };
        assert_eq!(exit_code(&failed), 1);
        let unsupported = EngineHelpersStatus { supported: false, ..ok };
        assert_eq!(exit_code(&unsupported), 1);
    }

    #[test]
    fn the_status_json_is_one_object_with_the_toggle_fields() {
        let json = serde_json::to_value(snapshot(None)).unwrap();
        assert!(json["supported"].is_boolean());
        assert!(json["state"].is_string());
        assert_eq!(json["helpers"].as_array().unwrap().len(), 2);
        assert!(json["error"].is_null());
        assert!(json["engines_enabled"].is_boolean());
        assert!(matches!(json["engine_lifetime"].as_str(), Some("with_app" | "always")));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn the_cli_fails_where_there_are_no_helpers() {
        assert_eq!(cli_main(&args(&[CLI_FLAG, "status"])), Some(1));
        assert_eq!(cli_main(&args(&[CLI_FLAG, "restart"])), Some(1));
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unsupported_platforms_report_it() {
        let status = snapshot(None);
        assert!(!status.supported);
        assert_eq!(status.state, HelperState::Unsupported);
    }
}

//! Stopping the bundled engines when Unsloth really quits, behind the `attached-engines`
//! feature. Only in `with_app` mode (see `engine_lifetime`), and only while the helpers are
//! registered and the user has the engines enabled.
//!
//! The sequence, all of it bounded by `QUIT_BUDGET` (45 s):
//!
//! 1. unload every loaded oMLX model: `POST :8843/v1/models/{id}/unload?force=1`, for each loaded
//!    id of `/v1/models/status`. `force=1` aborts in-flight requests instead of waiting up to 20 s
//!    for them (a plain unload answers 409 `model_busy` while a client streams), so a quit stays
//!    fast;
//! 2. ComfyUI, when its helper is registered and it answers: `POST /interrupt` while a job runs,
//!    `POST /queue {"clear": true}` while jobs are pending (a with_app quit takes down any queued
//!    job, that is the intended meaning), then `POST /free {"unload_models": true, "free_memory":
//!    true}` so the weights are released before the helper gets its SIGTERM;
//! 3. unregister every helper, last and always, even if a step ran out of budget or failed. Every
//!    failure is logged and never stops the next step.
//!
//! It shows no UI, and logs to `tauri.log`. The hook is `cleanup_child_processes` in `main.rs`,
//! which every real exit path reaches once (tray Quit, Cmd+Q and the app menu's Quit through
//! `request_quit`; an external terminate through `RunEvent::Exit`; a Unix signal; and
//! `RunEvent::Exit` as the safety net). A window close only hides the window and never gets here.

use std::time::Duration;

use log::{info, warn};

use crate::engine_helpers::{self, HelperState};
use crate::engine_lifetime::{self, EngineLifetime};
use crate::engine_tray::{self, EngineUrls};
use crate::loopback_http;

/// The unload budget. Unregister follows regardless.
pub(crate) const QUIT_BUDGET: Duration = Duration::from_secs(45);
/// Per request; the budget bounds the sum.
const STATUS_TIMEOUT: Duration = Duration::from_secs(3);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(30);
/// `/interrupt`, `/queue` and `/free` only set a flag for ComfyUI's main loop, so they answer fast.
const COMFYUI_TIMEOUT: Duration = Duration::from_secs(5);

/// Which engines the quit talks to: those whose helper is registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Targets {
    pub(crate) omlx: bool,
    pub(crate) comfyui: bool,
}

impl Targets {
    /// A helper that is registered may be running; one that is not, or awaits approval, is not ours.
    pub(crate) fn from_states(states: &[(&'static str, HelperState)]) -> Self {
        let registered = |name: &str| {
            states.iter().any(|(n, state)| {
                *n == name && matches!(state, HelperState::Enabled | HelperState::Partial)
            })
        };
        Self {
            omlx: registered("omlx"),
            comfyui: registered("comfyui"),
        }
    }

    fn any(self) -> bool {
        self.omlx || self.comfyui
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuitPlan {
    pub(crate) budget: Duration,
    pub(crate) targets: Targets,
}

impl Default for QuitPlan {
    fn default() -> Self {
        Self {
            budget: QUIT_BUDGET,
            targets: Targets {
                omlx: true,
                comfyui: true,
            },
        }
    }
}

/// Whether a real quit stops the engines: `with_app`, engines enabled, and at least one helper
/// registered (an unregistered helper is not ours to stop; one awaiting approval is not running).
pub(crate) fn stops_on_quit(lifetime: EngineLifetime, enabled: bool, any_registered: bool) -> bool {
    lifetime == EngineLifetime::WithApp && enabled && any_registered
}

/// The engines' HTTP surface the sequence needs, so the order and the timeouts are tested
/// against a fake. Status codes are returned as they are: 409 is an answer, not a failure.
pub(crate) trait EngineHttp {
    /// Ids of the loaded oMLX models; Err when oMLX does not answer.
    async fn loaded_omlx_ids(&self) -> Result<Vec<String>, String>;
    async fn unload_omlx(&self, id: &str) -> Result<u16, String>;
    /// ComfyUI's `(running, pending)` job counts from `GET /queue`; Err when it does not answer.
    async fn comfyui_queue(&self) -> Result<(usize, usize), String>;
    /// `POST <path>` to ComfyUI with a JSON body; the status code is returned as it is.
    async fn comfyui_post(&self, path: &str, body: serde_json::Value) -> Result<u16, String>;
}

fn is_ok(code: u16) -> bool {
    (200..300).contains(&code)
}

/// Unload every loaded oMLX model. Failures are logged and never stop the next step.
async fn stop_omlx<E: EngineHttp>(engine: &E) {
    match engine.loaded_omlx_ids().await {
        Err(error) => info!("engine quit: oMLX not reachable ({error}), nothing to unload"),
        Ok(ids) if ids.is_empty() => info!("engine quit: oMLX has no models loaded"),
        Ok(ids) => {
            for id in ids {
                match engine.unload_omlx(&id).await {
                    Ok(code) if is_ok(code) => info!("engine quit: oMLX unloaded {id}"),
                    Ok(code) => warn!("engine quit: oMLX unload of {id} answered HTTP {code}"),
                    Err(error) => warn!("engine quit: oMLX unload of {id} failed: {error}"),
                }
            }
        }
    }
}

async fn comfyui_step<E: EngineHttp>(engine: &E, what: &str, path: &str, body: serde_json::Value) {
    match engine.comfyui_post(path, body).await {
        Ok(code) if is_ok(code) => info!("engine quit: ComfyUI {what}"),
        Ok(code) => warn!("engine quit: ComfyUI {what} answered HTTP {code}"),
        Err(error) => warn!("engine quit: ComfyUI {what} failed: {error}"),
    }
}

/// Take the running and queued jobs out of ComfyUI and free its models and memory. Every failure is
/// logged and never stops the next step. Shared by the quit and by switching ComfyUI off.
pub(crate) async fn stop_comfyui<E: EngineHttp>(engine: &E) {
    match engine.comfyui_queue().await {
        Err(error) => {
            info!("engine quit: ComfyUI not reachable ({error}), nothing to free");
            return;
        }
        Ok((running, pending)) => {
            if running > 0 {
                comfyui_step(
                    engine,
                    "interrupted the running job",
                    "/interrupt",
                    serde_json::json!({}),
                )
                .await;
            }
            if pending > 0 {
                comfyui_step(
                    engine,
                    "cleared the queue",
                    "/queue",
                    serde_json::json!({"clear": true}),
                )
                .await;
            }
        }
    }
    comfyui_step(
        engine,
        "freed its models and memory",
        "/free",
        serde_json::json!({"unload_models": true, "free_memory": true}),
    )
    .await;
}

/// Stop the targeted engines: oMLX models first, then ComfyUI. Failures never skip the next step:
/// the helpers are unregistered afterwards either way, and SIGTERM releases whatever stayed loaded.
pub(crate) async fn stop_engines<E: EngineHttp>(engine: &E, targets: Targets) {
    if targets.omlx {
        stop_omlx(engine).await;
    }
    if targets.comfyui {
        stop_comfyui(engine).await;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct QuitOutcome {
    pub(crate) timed_out: bool,
}

/// The whole sequence: `stop_engines` inside the budget, then `unregister` (always, last, for every helper).
pub(crate) async fn quit_sequence<E: EngineHttp>(
    engine: &E,
    plan: &QuitPlan,
    unregister: impl FnOnce(),
) -> QuitOutcome {
    let timed_out = tokio::time::timeout(plan.budget, stop_engines(engine, plan.targets))
        .await
        .is_err();
    if timed_out {
        warn!(
            "engine quit: the engines did not stop within {} s; unregistering the helpers anyway",
            plan.budget.as_secs()
        );
    }
    unregister();
    QuitOutcome { timed_out }
}

// ---------------------------------------------------------------------------------------------
// The real thing

struct HttpEngines {
    urls: EngineUrls,
}

impl HttpEngines {
    async fn post(&self, base: &str, path: &str, timeout: Duration) -> Result<u16, String> {
        let client = loopback_http::client(timeout).map_err(|e| e.to_string())?;
        client
            .post(format!("{base}{path}"))
            .send()
            .await
            .map(|response| response.status().as_u16())
            .map_err(|e| e.to_string())
    }

    async fn post_json(
        &self,
        base: &str,
        path: &str,
        body: &serde_json::Value,
        timeout: Duration,
    ) -> Result<u16, String> {
        let client = loopback_http::client(timeout).map_err(|e| e.to_string())?;
        client
            .post(format!("{base}{path}"))
            .json(body)
            .send()
            .await
            .map(|response| response.status().as_u16())
            .map_err(|e| e.to_string())
    }

    async fn get_json(&self, base: &str, path: &str) -> Result<serde_json::Value, String> {
        let client = loopback_http::client(STATUS_TIMEOUT).map_err(|e| e.to_string())?;
        let response = client
            .get(format!("{base}{path}"))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status().as_u16()));
        }
        response.json().await.map_err(|e| e.to_string())
    }
}

impl EngineHttp for HttpEngines {
    async fn loaded_omlx_ids(&self) -> Result<Vec<String>, String> {
        let body = self.get_json(&self.urls.omlx, "/v1/models/status").await?;
        let status =
            engine_tray::parse_omlx_status(&body).ok_or("unexpected /v1/models/status body")?;
        Ok(engine_tray::all_loaded_dirs(&status))
    }

    async fn unload_omlx(&self, id: &str) -> Result<u16, String> {
        let path = format!(
            "/v1/models/{}/unload?force=1",
            engine_tray::encode_segment(id)
        );
        self.post(&self.urls.omlx, &path, UNLOAD_TIMEOUT).await
    }

    async fn comfyui_queue(&self) -> Result<(usize, usize), String> {
        let body = self.get_json(&self.urls.comfyui, "/queue").await?;
        engine_tray::parse_comfyui_queue(&body).ok_or_else(|| "unexpected /queue body".to_string())
    }

    async fn comfyui_post(&self, path: &str, body: serde_json::Value) -> Result<u16, String> {
        self.post_json(&self.urls.comfyui, path, &body, COMFYUI_TIMEOUT)
            .await
    }
}

/// Switching ComfyUI off in Settings: interrupt a running job and free the memory first, best effort.
pub(crate) async fn release_comfyui() {
    let engines = HttpEngines {
        urls: engine_tray::load_urls(),
    };
    stop_comfyui(&engines).await;
}

/// The quit hook. Blocks the calling thread for up to `QUIT_BUDGET` plus the unregister. The
/// callers are plain threads (the quit thread, the signal thread, the main thread in
/// `RunEvent::Exit`), none of them a tokio worker, so a private current-thread runtime is safe.
pub(crate) fn stop_on_quit() {
    let settings = engine_lifetime::load();
    let states = engine_helpers::helper_states();
    let enabled =
        engine_lifetime::resolve_enabled(&settings, engine_helpers::master_state(&states));
    let targets = Targets::from_states(&states);
    if !stops_on_quit(settings.lifetime, enabled, targets.any()) {
        info!(
            "engine quit: leaving the engines running ({} enabled={enabled} helpers={states:?})",
            settings.lifetime.as_str()
        );
        return;
    }
    info!(
        "engine quit: stopping the engines ({} budget {} s, oMLX={} ComfyUI={})",
        settings.lifetime.as_str(),
        QUIT_BUDGET.as_secs(),
        targets.omlx,
        targets.comfyui
    );
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            warn!("engine quit: no runtime ({error}); unregistering the helpers without the graceful stop");
            log_unregister(engine_helpers::disable());
            return;
        }
    };
    let engines = HttpEngines {
        urls: engine_tray::load_urls(),
    };
    let plan = QuitPlan {
        targets,
        ..QuitPlan::default()
    };
    let outcome = runtime.block_on(quit_sequence(&engines, &plan, || {
        log_unregister(engine_helpers::disable())
    }));
    info!("engine quit: done (timed out: {})", outcome.timed_out);
}

fn log_unregister(status: engine_helpers::EngineHelpersStatus) {
    match serde_json::to_string(&status) {
        Ok(json) => info!("engine quit: helpers unregistered: {json}"),
        Err(error) => {
            warn!("engine quit: unregistered the helpers, could not encode the status: {error}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct Fake {
        calls: RefCell<Vec<String>>,
        omlx: Option<Vec<String>>,
        unload_codes: RefCell<VecDeque<u16>>,
        hang_on_unload: bool,
        /// (running, pending); None = ComfyUI does not answer
        comfyui: Option<(usize, usize)>,
        post_codes: RefCell<VecDeque<u16>>,
        hang_on_comfyui_post: bool,
    }
    impl Fake {
        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
        fn log(&self, line: impl Into<String>) {
            self.calls.borrow_mut().push(line.into());
        }
    }
    impl EngineHttp for Fake {
        async fn loaded_omlx_ids(&self) -> Result<Vec<String>, String> {
            self.log("omlx status");
            self.omlx
                .clone()
                .ok_or_else(|| "connection refused".to_string())
        }
        async fn unload_omlx(&self, id: &str) -> Result<u16, String> {
            self.log(format!("unload {id}"));
            if self.hang_on_unload {
                std::future::pending::<()>().await;
            }
            Ok(self.unload_codes.borrow_mut().pop_front().unwrap_or(200))
        }
        async fn comfyui_queue(&self) -> Result<(usize, usize), String> {
            self.log("comfyui queue");
            self.comfyui.ok_or_else(|| "connection refused".to_string())
        }
        async fn comfyui_post(&self, path: &str, body: serde_json::Value) -> Result<u16, String> {
            self.log(format!("comfyui {path} {body}"));
            if self.hang_on_comfyui_post {
                std::future::pending::<()>().await;
            }
            Ok(self.post_codes.borrow_mut().pop_front().unwrap_or(200))
        }
    }
    fn fake(omlx: &[&str]) -> Fake {
        Fake {
            omlx: Some(omlx.iter().map(|s| s.to_string()).collect()),
            ..Fake::default()
        }
    }
    const FREE: &str = r#"comfyui /free {"free_memory":true,"unload_models":true}"#;
    fn omlx_only() -> QuitPlan {
        QuitPlan {
            targets: Targets {
                omlx: true,
                comfyui: false,
            },
            ..QuitPlan::default()
        }
    }
    fn comfyui_only() -> QuitPlan {
        QuitPlan {
            targets: Targets {
                omlx: false,
                comfyui: true,
            },
            ..QuitPlan::default()
        }
    }
    fn run<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }
    #[test]
    fn only_with_app_and_registered_enabled_helpers_stop_on_quit() {
        use EngineLifetime::*;
        assert!(stops_on_quit(WithApp, true, true));
        assert!(!stops_on_quit(Always, true, true));
        assert!(!stops_on_quit(WithApp, false, true));
        assert!(!stops_on_quit(WithApp, true, false));
        assert_eq!(QuitPlan::default().budget.as_secs(), 45);
    }
    #[test]
    fn the_targets_are_the_registered_helpers() {
        use HelperState::*;
        let of = |omlx, comfyui| Targets::from_states(&[("omlx", omlx), ("comfyui", comfyui)]);
        let t = |omlx, comfyui| Targets { omlx, comfyui };
        assert_eq!(of(Enabled, NotRegistered), t(true, false));
        assert_eq!(of(Enabled, Enabled), t(true, true));
        assert_eq!(of(NotRegistered, Enabled), t(false, true));
        assert_eq!(of(Partial, NotFound), t(true, false));
        // approval pending, unbundled and unsupported helpers are not running
        for state in [NotRegistered, RequiresApproval, NotFound, Unsupported] {
            assert_eq!(of(state, state), t(false, false), "{state:?}");
            assert!(!of(state, state).any());
        }
        assert!(of(Enabled, NotRegistered).any());
    }
    #[test]
    fn unloads_every_model_before_unregistering() {
        let engine = fake(&["swift", "embed"]);
        let at_unregister = RefCell::new(Vec::new());
        let result = run(quit_sequence(&engine, &omlx_only(), || {
            *at_unregister.borrow_mut() = engine.calls()
        }));
        assert!(!result.timed_out);
        assert_eq!(
            engine.calls(),
            ["omlx status", "unload swift", "unload embed"]
        );
        assert_eq!(engine.calls(), *at_unregister.borrow());
    }
    #[test]
    fn empty_and_unreachable_engines_still_unregister() {
        for engine in [fake(&[]), Fake::default()] {
            let unregistered = RefCell::new(0);
            let result = run(quit_sequence(&engine, &QuitPlan::default(), || {
                *unregistered.borrow_mut() += 1
            }));
            assert_eq!(*unregistered.borrow(), 1);
            // ComfyUI is asked too and, not answering, gets nothing else
            assert_eq!(engine.calls(), ["omlx status", "comfyui queue"]);
            assert!(!result.timed_out);
        }
    }
    #[test]
    fn failed_unload_does_not_skip_the_rest() {
        let engine = fake(&["a", "b"]);
        *engine.unload_codes.borrow_mut() = VecDeque::from([409, 200]);
        run(stop_engines(&engine, omlx_only().targets));
        assert_eq!(engine.calls(), ["omlx status", "unload a", "unload b"]);
    }
    #[test]
    fn budget_timeout_still_unregisters_last() {
        let engine = Fake {
            hang_on_unload: true,
            ..fake(&["m"])
        };
        let at_unregister = RefCell::new(Vec::new());
        let plan = QuitPlan {
            budget: Duration::from_millis(10),
            ..QuitPlan::default()
        };
        let result = run(quit_sequence(&engine, &plan, || {
            *at_unregister.borrow_mut() = engine.calls()
        }));
        assert!(result.timed_out);
        assert_eq!(*at_unregister.borrow(), ["omlx status", "unload m"]);
    }

    #[test]
    fn an_idle_comfyui_is_only_freed() {
        let engine = Fake {
            comfyui: Some((0, 0)),
            ..Fake::default()
        };
        run(stop_engines(&engine, comfyui_only().targets));
        assert_eq!(engine.calls(), ["comfyui queue", FREE]);
    }
    #[test]
    fn a_running_job_is_interrupted_before_the_free() {
        let engine = Fake {
            comfyui: Some((1, 0)),
            ..Fake::default()
        };
        run(stop_engines(&engine, comfyui_only().targets));
        assert_eq!(
            engine.calls(),
            ["comfyui queue", "comfyui /interrupt {}", FREE]
        );
    }
    #[test]
    fn pending_jobs_are_cleared_and_only_when_there_are_any() {
        let engine = Fake {
            comfyui: Some((0, 3)),
            ..Fake::default()
        };
        run(stop_engines(&engine, comfyui_only().targets));
        assert_eq!(
            engine.calls(),
            ["comfyui queue", r#"comfyui /queue {"clear":true}"#, FREE]
        );
    }
    #[test]
    fn a_busy_comfyui_is_interrupted_cleared_and_freed_in_that_order() {
        let engine = Fake {
            comfyui: Some((1, 2)),
            ..Fake::default()
        };
        run(stop_engines(&engine, comfyui_only().targets));
        assert_eq!(
            engine.calls(),
            [
                "comfyui queue",
                "comfyui /interrupt {}",
                r#"comfyui /queue {"clear":true}"#,
                FREE
            ]
        );
    }
    #[test]
    fn a_failing_interrupt_does_not_skip_the_clear_or_the_free() {
        let engine = Fake {
            comfyui: Some((1, 1)),
            ..Fake::default()
        };
        *engine.post_codes.borrow_mut() = VecDeque::from([500, 500]);
        run(stop_engines(&engine, comfyui_only().targets));
        assert_eq!(engine.calls().len(), 4);
        assert_eq!(engine.calls()[3], FREE);
    }
    #[test]
    fn omlx_is_unloaded_before_comfyui_is_freed_and_both_before_the_unregister() {
        let engine = Fake {
            comfyui: Some((0, 0)),
            ..fake(&["swift"])
        };
        let at_unregister = RefCell::new(Vec::new());
        run(quit_sequence(&engine, &QuitPlan::default(), || {
            *at_unregister.borrow_mut() = engine.calls()
        }));
        assert_eq!(
            *at_unregister.borrow(),
            ["omlx status", "unload swift", "comfyui queue", FREE]
        );
    }
    #[test]
    fn an_unwanted_engine_is_not_contacted() {
        let engine = Fake {
            comfyui: Some((1, 0)),
            ..fake(&["swift"])
        };
        run(stop_engines(&engine, omlx_only().targets));
        assert!(!engine.calls().iter().any(|c| c.starts_with("comfyui")));
        let engine = Fake {
            comfyui: Some((0, 0)),
            ..fake(&["swift"])
        };
        run(stop_engines(&engine, comfyui_only().targets));
        assert!(!engine
            .calls()
            .iter()
            .any(|c| c == "omlx status" || c.starts_with("unload ")));
    }
    #[test]
    fn an_unreachable_comfyui_still_unregisters() {
        let engine = fake(&["swift"]);
        let unregistered = RefCell::new(0);
        let result = run(quit_sequence(&engine, &QuitPlan::default(), || {
            *unregistered.borrow_mut() += 1
        }));
        assert!(!result.timed_out);
        assert_eq!(*unregistered.borrow(), 1);
        assert_eq!(
            engine.calls(),
            ["omlx status", "unload swift", "comfyui queue"]
        );
    }
    #[test]
    fn a_hanging_comfyui_runs_out_the_budget_and_still_unregisters_last() {
        let engine = Fake {
            comfyui: Some((1, 0)),
            hang_on_comfyui_post: true,
            ..fake(&["m"])
        };
        let at_unregister = RefCell::new(Vec::new());
        let plan = QuitPlan {
            budget: Duration::from_millis(10),
            ..QuitPlan::default()
        };
        let result = run(quit_sequence(&engine, &plan, || {
            *at_unregister.borrow_mut() = engine.calls()
        }));
        assert!(result.timed_out);
        assert_eq!(
            *at_unregister.borrow(),
            [
                "omlx status",
                "unload m",
                "comfyui queue",
                "comfyui /interrupt {}"
            ]
        );
    }

    // --- the real client against a socket ----------------------------------------------------

    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Answers each connection from `routes` ("METHOD path" -> (status, body)) and records the
    /// request lines. Stops after `connections` requests.
    fn serve(
        routes: Vec<(&'static str, u16, &'static str)>,
        connections: usize,
    ) -> (u16, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for _ in 0..connections {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0_u8; 4096];
                let n = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..n]).to_string();
                let line = request.lines().next().unwrap_or("").to_string();
                let key: Vec<&str> = line.split(' ').take(2).collect();
                let key = key.join(" ");
                seen.push(key.clone());
                let (code, body) = routes
                    .iter()
                    .find(|(route, _, _)| *route == key)
                    .map(|(_, code, body)| (*code, *body))
                    .unwrap_or((404, ""));
                let reply = format!(
                    "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
            }
            seen
        });
        (port, handle)
    }

    #[test]
    fn the_http_client_uses_the_documented_endpoints_and_returns_the_status() {
        let body = r#"{"models":[
            {"id":"swift-1.5-27b:fast","model_path":"/m/swift","loaded":true},
            {"id":"swift-1.5-27b","model_path":"/m/swift","loaded":true},
            {"id":"idle","model_path":"/m/idle","loaded":false}]}"#;
        let (omlx_port, omlx) = serve(
            vec![
                ("GET /v1/models/status", 200, body),
                ("POST /v1/models/swift-1.5-27b/unload?force=1", 202, ""),
                ("POST /v1/models/a%3Ab/unload?force=1", 409, ""),
            ],
            3,
        );
        let engine = HttpEngines {
            urls: EngineUrls {
                omlx: format!("http://127.0.0.1:{omlx_port}"),
                ..EngineUrls::default()
            },
        };
        run(async {
            // two rows share a model_path, so they collapse to one directory id
            assert_eq!(engine.loaded_omlx_ids().await.unwrap(), ["swift-1.5-27b"]);
            assert_eq!(engine.unload_omlx("swift-1.5-27b").await, Ok(202));
            assert_eq!(engine.unload_omlx("a:b").await, Ok(409));
        });
        // a quit never waits on a streaming client: every unload carries force=1
        let seen = omlx.join().unwrap();
        assert_eq!(seen.len(), 3);
        assert!(seen[1..]
            .iter()
            .all(|line| line.ends_with("/unload?force=1")));
    }

    #[test]
    fn a_closed_port_is_an_error_not_a_status() {
        let port = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let engine = HttpEngines {
            urls: EngineUrls {
                omlx: format!("http://127.0.0.1:{port}"),
                comfyui: format!("http://127.0.0.1:{port}"),
            },
        };
        run(async {
            assert!(engine.loaded_omlx_ids().await.is_err());
            assert!(engine.comfyui_queue().await.is_err());
            assert!(engine
                .comfyui_post("/free", serde_json::json!({}))
                .await
                .is_err());
        });
    }

    /// Like `serve`, but also keeps each request body (the part after the blank line).
    fn serve_with_bodies(
        routes: Vec<(&'static str, u16, &'static str)>,
        connections: usize,
    ) -> (u16, std::thread::JoinHandle<Vec<(String, String)>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for _ in 0..connections {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0_u8; 8192];
                let n = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..n]).to_string();
                let line = request.lines().next().unwrap_or("").to_string();
                let key: Vec<&str> = line.split(' ').take(2).collect();
                let key = key.join(" ");
                let body = request.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                let (code, reply_body) = routes
                    .iter()
                    .find(|(route, _, _)| *route == key)
                    .map(|(_, code, body)| (*code, *body))
                    .unwrap_or((404, ""));
                let reply = format!(
                    "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply_body}",
                    reply_body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
                seen.push((key, body));
            }
            seen
        });
        (port, handle)
    }

    #[test]
    fn the_comfyui_client_reads_the_queue_and_posts_json_bodies() {
        let (port, comfyui) = serve_with_bodies(
            vec![
                (
                    "GET /queue",
                    200,
                    r#"{"queue_running":[[0,"a",{},{},[]]],"queue_pending":[[1,"b",{},{},[]],[2,"c",{},{},[]]]}"#,
                ),
                ("POST /interrupt", 200, ""),
                ("POST /queue", 200, ""),
                ("POST /free", 200, ""),
            ],
            4,
        );
        let engine = HttpEngines {
            urls: EngineUrls {
                comfyui: format!("http://127.0.0.1:{port}"),
                ..EngineUrls::default()
            },
        };
        run(async { stop_comfyui(&engine).await });
        let seen = comfyui.join().unwrap();
        let keys: Vec<&str> = seen.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            ["GET /queue", "POST /interrupt", "POST /queue", "POST /free"]
        );
        assert_eq!(seen[2].1, r#"{"clear":true}"#);
        let free: serde_json::Value = serde_json::from_str(&seen[3].1).unwrap();
        assert_eq!(
            free,
            serde_json::json!({"unload_models": true, "free_memory": true})
        );
    }
}

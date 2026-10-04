//! Stopping the bundled engines when Unsloth really quits, behind the `attached-engines`
//! feature. Only in `with_app` mode (see `engine_lifetime`), and only while the helpers are
//! registered and the user has the engines enabled.
//!
//! The sequence, all of it bounded by `QUIT_BUDGET` (45 s):
//!
//! 1. unload every loaded oMLX model: `POST :8843/v1/models/{id}/unload`, for each loaded id of
//!    `/v1/models/status`;
//! 2. stop ds4 with `POST :8001/admin/stop?if_idle=1`. A 409 means a reply is in flight: poll
//!    `/admin/status` for up to 30 s until `in_flight` is 0, then stop again with `if_idle`. After
//!    the 30 s, stop anyway without `if_idle` and log it;
//! 3. unregister the helpers. launchd then sends SIGTERM and the engines shut down gracefully
//!    (their own 90 s). This runs even when steps 1 and 2 ran out of budget.
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

/// The whole of steps 1 and 2. Step 3 (unregister) follows regardless.
pub(crate) const QUIT_BUDGET: Duration = Duration::from_secs(45);
/// How long a ds4 reply in flight may keep the quit waiting.
const IN_FLIGHT_WAIT: Duration = Duration::from_secs(30);
const IN_FLIGHT_POLL: Duration = Duration::from_secs(1);
/// Per request; the budget bounds the sum.
const STATUS_TIMEOUT: Duration = Duration::from_secs(3);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(30);
const STOP_TIMEOUT: Duration = Duration::from_secs(40);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuitPlan {
    pub(crate) budget: Duration,
    pub(crate) in_flight_wait: Duration,
    pub(crate) in_flight_poll: Duration,
}

impl Default for QuitPlan {
    fn default() -> Self {
        Self {
            budget: QUIT_BUDGET,
            in_flight_wait: IN_FLIGHT_WAIT,
            in_flight_poll: IN_FLIGHT_POLL,
        }
    }
}

/// Whether a real quit stops the engines: `with_app`, engines enabled, and helpers registered
/// (an unregistered pair is not ours to stop; one awaiting approval is not running).
pub(crate) fn stops_on_quit(lifetime: EngineLifetime, enabled: bool, state: HelperState) -> bool {
    lifetime == EngineLifetime::WithApp
        && enabled
        && matches!(state, HelperState::Enabled | HelperState::Partial)
}

/// The engines' HTTP surface the sequence needs, so the order and the timeouts are tested
/// against a fake. Status codes are returned as they are: 409 is an answer, not a failure.
pub(crate) trait EngineHttp {
    /// Ids of the loaded oMLX models; Err when oMLX does not answer.
    async fn loaded_omlx_ids(&self) -> Result<Vec<String>, String>;
    async fn unload_omlx(&self, id: &str) -> Result<u16, String>;
    /// `POST /admin/stop`, with `?if_idle=1` when `if_idle`. Err when ds4's launcher does not answer.
    async fn ds4_stop(&self, if_idle: bool) -> Result<u16, String>;
    async fn ds4_in_flight(&self) -> Result<u64, String>;
    async fn sleep(&self, duration: Duration);
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Ds4Outcome {
    Stopped,
    /// A reply was in flight; it finished within the wait and ds4 was then stopped.
    StoppedAfterWait,
    /// The wait ran out and ds4 was stopped without `if_idle`.
    ForcedAfterTimeout,
    Unreachable,
    Failed(u16),
}

fn is_ok(code: u16) -> bool {
    (200..300).contains(&code)
}

async fn stop_ds4<E: EngineHttp>(engine: &E, plan: &QuitPlan) -> Ds4Outcome {
    match engine.ds4_stop(true).await {
        Err(error) => {
            info!("engine quit: ds4 not reachable ({error}), nothing to stop");
            return Ds4Outcome::Unreachable;
        }
        Ok(code) if is_ok(code) => return Ds4Outcome::Stopped,
        Ok(409) => {}
        Ok(code) => {
            warn!("engine quit: ds4 /admin/stop?if_idle=1 answered HTTP {code}");
            return Ds4Outcome::Failed(code);
        }
    }
    info!(
        "engine quit: ds4 has a reply in flight, waiting up to {} s",
        plan.in_flight_wait.as_secs()
    );
    let mut waited = Duration::ZERO;
    while waited < plan.in_flight_wait {
        engine.sleep(plan.in_flight_poll).await;
        waited += plan.in_flight_poll;
        match engine.ds4_in_flight().await {
            Ok(0) => match engine.ds4_stop(true).await {
                Ok(code) if is_ok(code) => return Ds4Outcome::StoppedAfterWait,
                Ok(409) => {}
                Ok(code) => {
                    warn!("engine quit: ds4 /admin/stop?if_idle=1 answered HTTP {code}");
                    return Ds4Outcome::Failed(code);
                }
                Err(error) => {
                    info!("engine quit: ds4 went away while waiting ({error})");
                    return Ds4Outcome::Unreachable;
                }
            },
            Ok(_) => {}
            Err(error) => {
                info!("engine quit: ds4 went away while waiting ({error})");
                return Ds4Outcome::Unreachable;
            }
        }
    }
    warn!(
        "engine quit: ds4 still busy after {} s, stopping it anyway (without if_idle)",
        plan.in_flight_wait.as_secs()
    );
    match engine.ds4_stop(false).await {
        Ok(code) if is_ok(code) => Ds4Outcome::ForcedAfterTimeout,
        Ok(code) => {
            warn!("engine quit: forced ds4 stop answered HTTP {code}");
            Ds4Outcome::Failed(code)
        }
        Err(error) => {
            warn!("engine quit: forced ds4 stop failed: {error}");
            Ds4Outcome::Unreachable
        }
    }
}

/// Steps 1 and 2. Failures are logged and never stop the next step: the helpers are
/// unregistered afterwards either way, and SIGTERM releases whatever stayed loaded.
pub(crate) async fn stop_engines<E: EngineHttp>(engine: &E, plan: &QuitPlan) -> Ds4Outcome {
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
    let outcome = stop_ds4(engine, plan).await;
    info!("engine quit: ds4 {outcome:?}");
    outcome
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct QuitOutcome {
    pub(crate) timed_out: bool,
}

/// The whole sequence: `stop_engines` inside the budget, then `unregister` (always, last).
pub(crate) async fn quit_sequence<E: EngineHttp>(
    engine: &E,
    plan: &QuitPlan,
    unregister: impl FnOnce(),
) -> QuitOutcome {
    let timed_out = tokio::time::timeout(plan.budget, stop_engines(engine, plan))
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
        let status = engine_tray::parse_omlx_status(&body).ok_or("unexpected /v1/models/status body")?;
        Ok(engine_tray::all_loaded_dirs(&status))
    }

    async fn unload_omlx(&self, id: &str) -> Result<u16, String> {
        let path = format!("/v1/models/{}/unload", engine_tray::encode_segment(id));
        self.post(&self.urls.omlx, &path, UNLOAD_TIMEOUT).await
    }

    async fn ds4_stop(&self, if_idle: bool) -> Result<u16, String> {
        let path = if if_idle { "/admin/stop?if_idle=1" } else { "/admin/stop" };
        self.post(&self.urls.ds4, path, STOP_TIMEOUT).await
    }

    async fn ds4_in_flight(&self) -> Result<u64, String> {
        let body = self.get_json(&self.urls.ds4, "/admin/status").await?;
        Ok(body.get("in_flight").and_then(serde_json::Value::as_u64).unwrap_or(0))
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}

/// The quit hook. Blocks the calling thread for up to `QUIT_BUDGET` plus the unregister. The
/// callers are plain threads (the quit thread, the signal thread, the main thread in
/// `RunEvent::Exit`), none of them a tokio worker, so a private current-thread runtime is safe.
pub(crate) fn stop_on_quit() {
    let settings = engine_lifetime::load();
    let state = engine_helpers::current_state();
    let enabled = engine_lifetime::resolve_enabled(&settings, state);
    if !stops_on_quit(settings.lifetime, enabled, state) {
        info!(
            "engine quit: leaving the engines running ({} enabled={enabled} helpers={state:?})",
            settings.lifetime.as_str()
        );
        return;
    }
    info!("engine quit: stopping the engines ({} budget {} s)", settings.lifetime.as_str(), QUIT_BUDGET.as_secs());
    let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(error) => {
            warn!("engine quit: no runtime ({error}); unregistering the helpers without the graceful stop");
            log_unregister(engine_helpers::disable());
            return;
        }
    };
    let engines = HttpEngines { urls: engine_tray::load_urls() };
    let outcome = runtime.block_on(quit_sequence(&engines, &QuitPlan::default(), || {
        log_unregister(engine_helpers::disable())
    }));
    info!("engine quit: done (timed out: {})", outcome.timed_out);
}

fn log_unregister(status: engine_helpers::EngineHelpersStatus) {
    match serde_json::to_string(&status) {
        Ok(json) => info!("engine quit: helpers unregistered: {json}"),
        Err(error) => warn!("engine quit: unregistered the helpers, could not encode the status: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    /// Records every call in order. `stops` answers each `ds4_stop` in turn (200 once empty),
    /// `in_flight` each status poll (0 once empty).
    #[derive(Default)]
    struct Fake {
        calls: RefCell<Vec<String>>,
        omlx: Option<Vec<String>>,
        unload_codes: RefCell<VecDeque<u16>>,
        stops: RefCell<VecDeque<Result<u16, String>>>,
        in_flight: RefCell<VecDeque<Result<u64, String>>>,
        hang_on_ds4_stop: bool,
        slept: RefCell<Duration>,
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
            self.omlx.clone().ok_or_else(|| "connection refused".to_string())
        }
        async fn unload_omlx(&self, id: &str) -> Result<u16, String> {
            self.log(format!("unload {id}"));
            Ok(self.unload_codes.borrow_mut().pop_front().unwrap_or(200))
        }
        async fn ds4_stop(&self, if_idle: bool) -> Result<u16, String> {
            self.log(if if_idle { "stop if_idle" } else { "stop force" });
            if self.hang_on_ds4_stop {
                std::future::pending::<()>().await;
            }
            self.stops.borrow_mut().pop_front().unwrap_or(Ok(200))
        }
        async fn ds4_in_flight(&self) -> Result<u64, String> {
            self.log("ds4 status");
            self.in_flight.borrow_mut().pop_front().unwrap_or(Ok(0))
        }
        async fn sleep(&self, duration: Duration) {
            self.log("sleep");
            *self.slept.borrow_mut() += duration;
        }
    }

    fn fake(omlx: &[&str]) -> Fake {
        Fake {
            omlx: Some(omlx.iter().map(|s| s.to_string()).collect()),
            ..Fake::default()
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
        use HelperState::*;
        assert!(stops_on_quit(WithApp, true, Enabled));
        assert!(stops_on_quit(WithApp, true, Partial));
        assert!(!stops_on_quit(Always, true, Enabled));
        assert!(!stops_on_quit(WithApp, false, Enabled));
        for state in [NotRegistered, RequiresApproval, NotFound, Unsupported] {
            assert!(!stops_on_quit(WithApp, true, state), "{state:?}");
        }
    }

    #[test]
    fn the_budget_is_about_forty_five_seconds_and_the_wait_thirty() {
        let plan = QuitPlan::default();
        assert_eq!(plan.budget.as_secs(), 45);
        assert_eq!(plan.in_flight_wait.as_secs(), 30);
        assert!(plan.in_flight_wait < plan.budget);
    }

    #[test]
    fn unloads_every_model_then_stops_ds4_then_unregisters() {
        let engine = fake(&["swift-1.5-27b", "embed"]);
        let order = RefCell::new(Vec::new());
        let outcome = run(quit_sequence(&engine, &QuitPlan::default(), || {
            order.borrow_mut().push(engine.calls());
        }));
        assert_eq!(outcome, QuitOutcome { timed_out: false });
        let at_unregister = order.borrow()[0].clone();
        assert_eq!(
            at_unregister,
            ["omlx status", "unload swift-1.5-27b", "unload embed", "stop if_idle"]
        );
        // the unregister saw every engine call and nothing came after it
        assert_eq!(engine.calls(), at_unregister);
    }

    #[test]
    fn nothing_loaded_still_stops_ds4_and_unregisters() {
        let engine = fake(&[]);
        let unregistered = RefCell::new(0);
        run(quit_sequence(&engine, &QuitPlan::default(), || *unregistered.borrow_mut() += 1));
        assert_eq!(engine.calls(), ["omlx status", "stop if_idle"]);
        assert_eq!(*unregistered.borrow(), 1);
    }

    #[test]
    fn unreachable_engines_are_skipped_and_the_helpers_still_unregister() {
        let engine = Fake {
            omlx: None,
            stops: RefCell::new(VecDeque::from([Err("connection refused".to_string())])),
            ..Fake::default()
        };
        let unregistered = RefCell::new(0);
        let outcome = run(quit_sequence(&engine, &QuitPlan::default(), || *unregistered.borrow_mut() += 1));
        assert_eq!(engine.calls(), ["omlx status", "stop if_idle"]);
        assert_eq!(*unregistered.borrow(), 1);
        assert!(!outcome.timed_out);
    }

    #[test]
    fn a_failed_unload_does_not_skip_the_rest() {
        let engine = fake(&["a", "b"]);
        *engine.unload_codes.borrow_mut() = VecDeque::from([409, 200]);
        run(stop_engines(&engine, &QuitPlan::default()));
        assert_eq!(
            engine.calls(),
            ["omlx status", "unload a", "unload b", "stop if_idle"]
        );
    }

    #[test]
    fn a_409_waits_for_the_reply_then_stops_with_if_idle() {
        let engine = fake(&[]);
        *engine.stops.borrow_mut() = VecDeque::from([Ok(409), Ok(200)]);
        *engine.in_flight.borrow_mut() = VecDeque::from([Ok(2), Ok(1), Ok(0)]);
        let outcome = run(stop_engines(&engine, &QuitPlan::default()));
        assert_eq!(outcome, Ds4Outcome::StoppedAfterWait);
        assert_eq!(
            engine.calls(),
            [
                "omlx status",
                "stop if_idle",
                "sleep", "ds4 status",
                "sleep", "ds4 status",
                "sleep", "ds4 status",
                "stop if_idle",
            ]
        );
        assert_eq!(*engine.slept.borrow(), Duration::from_secs(3));
    }

    #[test]
    fn a_reply_that_outlasts_thirty_seconds_ends_in_a_plain_stop() {
        let engine = fake(&[]);
        *engine.stops.borrow_mut() = VecDeque::from([Ok(409), Ok(200)]);
        // in flight for good: the default after the queue empties is 0, so fill it
        *engine.in_flight.borrow_mut() = (0..100).map(|_| Ok(1)).collect();
        let outcome = run(stop_engines(&engine, &QuitPlan::default()));
        assert_eq!(outcome, Ds4Outcome::ForcedAfterTimeout);
        assert_eq!(*engine.slept.borrow(), Duration::from_secs(30), "polled for 30 s, not more");
        let calls = engine.calls();
        assert_eq!(calls.last().map(String::as_str), Some("stop force"));
        assert_eq!(calls.iter().filter(|c| *c == "stop if_idle").count(), 1);
        assert_eq!(calls.iter().filter(|c| *c == "ds4 status").count(), 30);
    }

    #[test]
    fn a_second_409_after_the_reply_ended_keeps_waiting() {
        let engine = fake(&[]);
        // a new reply slipped in between the poll and the retry
        *engine.stops.borrow_mut() = VecDeque::from([Ok(409), Ok(409), Ok(200)]);
        *engine.in_flight.borrow_mut() = VecDeque::from([Ok(0), Ok(0)]);
        let outcome = run(stop_engines(&engine, &QuitPlan::default()));
        assert_eq!(outcome, Ds4Outcome::StoppedAfterWait);
        assert_eq!(*engine.slept.borrow(), Duration::from_secs(2));
    }

    #[test]
    fn ds4_going_away_while_waiting_is_not_an_error() {
        let engine = fake(&[]);
        *engine.stops.borrow_mut() = VecDeque::from([Ok(409)]);
        *engine.in_flight.borrow_mut() = VecDeque::from([Err("connection refused".to_string())]);
        assert_eq!(run(stop_engines(&engine, &QuitPlan::default())), Ds4Outcome::Unreachable);
    }

    #[test]
    fn an_unexpected_status_is_reported_and_not_retried() {
        let engine = fake(&[]);
        *engine.stops.borrow_mut() = VecDeque::from([Ok(500)]);
        assert_eq!(run(stop_engines(&engine, &QuitPlan::default())), Ds4Outcome::Failed(500));
        assert_eq!(engine.calls(), ["omlx status", "stop if_idle"]);
    }

    #[test]
    fn a_hung_engine_cannot_hold_the_quit_past_the_budget() {
        let engine = Fake { hang_on_ds4_stop: true, ..fake(&["m"]) };
        let plan = QuitPlan { budget: Duration::from_millis(60), ..QuitPlan::default() };
        let order = RefCell::new(Vec::new());
        let started = std::time::Instant::now();
        let outcome = run(quit_sequence(&engine, &plan, || order.borrow_mut().push(engine.calls())));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(outcome.timed_out);
        // the unload happened, ds4's stop was in progress, and the unregister still came last
        assert_eq!(order.borrow().len(), 1);
        assert_eq!(order.borrow()[0], ["omlx status", "unload m", "stop if_idle"]);
    }

    #[test]
    fn the_in_flight_wait_counts_against_the_budget() {
        // ds4 stays busy; with a budget shorter than the wait the wait is cut off
        let engine = fake(&[]);
        *engine.stops.borrow_mut() = VecDeque::from([Ok(409)]);
        let plan = QuitPlan {
            budget: Duration::from_millis(40),
            in_flight_wait: Duration::from_secs(30),
            in_flight_poll: Duration::from_millis(10),
        };
        struct Slow<'a>(&'a Fake);
        impl EngineHttp for Slow<'_> {
            async fn loaded_omlx_ids(&self) -> Result<Vec<String>, String> { self.0.loaded_omlx_ids().await }
            async fn unload_omlx(&self, id: &str) -> Result<u16, String> { self.0.unload_omlx(id).await }
            async fn ds4_stop(&self, if_idle: bool) -> Result<u16, String> { self.0.ds4_stop(if_idle).await }
            async fn ds4_in_flight(&self) -> Result<u64, String> { Ok(1) }
            async fn sleep(&self, duration: Duration) { tokio::time::sleep(duration).await }
        }
        let unregistered = RefCell::new(false);
        let outcome = run(quit_sequence(&Slow(&engine), &plan, || *unregistered.borrow_mut() = true));
        assert!(outcome.timed_out);
        assert!(*unregistered.borrow());
        assert!(!engine.calls().contains(&"stop force".to_string()));
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
                ("POST /v1/models/swift-1.5-27b/unload", 202, ""),
                ("POST /v1/models/a%3Ab/unload", 409, ""),
            ],
            3,
        );
        let (ds4_port, ds4) = serve(
            vec![
                ("POST /admin/stop?if_idle=1", 409, ""),
                ("GET /admin/status", 200, r#"{"loaded":true,"in_flight":3}"#),
                ("POST /admin/stop", 200, ""),
            ],
            3,
        );
        let engine = HttpEngines {
            urls: EngineUrls {
                omlx: format!("http://127.0.0.1:{omlx_port}"),
                ds4: format!("http://127.0.0.1:{ds4_port}"),
            },
        };
        run(async {
            // two rows share a model_path, so they collapse to one directory id
            assert_eq!(engine.loaded_omlx_ids().await.unwrap(), ["swift-1.5-27b"]);
            assert_eq!(engine.unload_omlx("swift-1.5-27b").await, Ok(202));
            assert_eq!(engine.unload_omlx("a:b").await, Ok(409));
            assert_eq!(engine.ds4_stop(true).await, Ok(409));
            assert_eq!(engine.ds4_in_flight().await, Ok(3));
            assert_eq!(engine.ds4_stop(false).await, Ok(200));
        });
        assert_eq!(omlx.join().unwrap().len(), 3);
        assert_eq!(ds4.join().unwrap().len(), 3);
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
                ds4: format!("http://127.0.0.1:{port}"),
            },
        };
        run(async {
            assert!(engine.loaded_omlx_ids().await.is_err());
            assert!(engine.ds4_stop(true).await.is_err());
        });
    }
}

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
//! 2. unregister the helper even if unloading ran out of budget.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuitPlan {
    pub(crate) budget: Duration,
}

impl Default for QuitPlan {
    fn default() -> Self {
        Self {
            budget: QUIT_BUDGET,
        }
    }
}

/// Whether a real quit stops the engines: `with_app`, engines enabled, and helpers registered
/// (an unregistered helper is not ours to stop; one awaiting approval is not running).
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
}

fn is_ok(code: u16) -> bool {
    (200..300).contains(&code)
}

/// Unload oMLX. Failures are logged and never stop the next step: the helpers are
/// unregistered afterwards either way, and SIGTERM releases whatever stayed loaded.
pub(crate) async fn stop_engines<E: EngineHttp>(engine: &E) {
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
    let timed_out = tokio::time::timeout(plan.budget, stop_engines(engine))
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
    info!(
        "engine quit: stopping the engines ({} budget {} s)",
        settings.lifetime.as_str(),
        QUIT_BUDGET.as_secs()
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
    let outcome = runtime.block_on(quit_sequence(&engines, &QuitPlan::default(), || {
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
            assert!(!stops_on_quit(WithApp, true, state));
        }
        assert_eq!(QuitPlan::default().budget.as_secs(), 45);
    }
    #[test]
    fn unloads_every_model_before_unregistering() {
        let engine = fake(&["swift", "embed"]);
        let at_unregister = RefCell::new(Vec::new());
        let result = run(quit_sequence(&engine, &QuitPlan::default(), || {
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
            assert_eq!(engine.calls(), ["omlx status"]);
            assert!(!result.timed_out);
        }
    }
    #[test]
    fn failed_unload_does_not_skip_the_rest() {
        let engine = fake(&["a", "b"]);
        *engine.unload_codes.borrow_mut() = VecDeque::from([409, 200]);
        run(stop_engines(&engine));
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
        };
        let result = run(quit_sequence(&engine, &plan, || {
            *at_unregister.borrow_mut() = engine.calls()
        }));
        assert!(result.timed_out);
        assert_eq!(*at_unregister.borrow(), ["omlx status", "unload m"]);
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
        assert!(seen[1..].iter().all(|line| line.ends_with("/unload?force=1")));
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
            },
        };
        run(async {
            assert!(engine.loaded_omlx_ids().await.is_err());
        });
    }
}

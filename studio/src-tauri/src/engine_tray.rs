//! Tray items for the attached engines (oMLX and ComfyUI), behind the
//! `attached-engines` feature.
//!
//! The tray talks to the engines directly (it must work with Studio's backend stopped): every
//! 5 s it polls oMLX status and idle timers and ComfyUI's queue, using engines.toml for the URLs.
//! The ComfyUI rows (status line and "Unload ComfyUI models") show only while the user wants
//! ComfyUI or it answers anyway. ComfyUI has no loaded-model list, and on Apple Silicon the device
//! figures in `/system_stats` are system-wide, so the status line shows the memory its process holds
//! (read from `lsof` and `ps`). The unload item is enabled only while the queue is empty and
//! refuses (with a message on the status line for one poll) if a job arrived since the last poll.
//! The idle-free countdown ("frees in 4m 12s") comes from the marker file the backend writes
//! (`comfyui-idle.json`), because the backend's own status API needs the owner's token.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use log::{info, warn};
use serde_json::Value;
use tauri::menu::{
    IsMenuItem, Menu, MenuItem, MenuItemBuilder, PredefinedMenuItem, Submenu, SubmenuBuilder,
};
use tauri::{AppHandle, Manager, Wry};

use crate::engine_helpers::{self, HelperState};
use crate::engine_lifetime::{self, EngineLifetime};
use crate::loopback_http;

const DEFAULT_OMLX_URL: &str = "http://127.0.0.1:8843";
const DEFAULT_COMFYUI_URL: &str = "http://127.0.0.1:8844";
const POLL_INTERVAL: Duration = Duration::from_secs(5);
const STATUS_TIMEOUT: Duration = Duration::from_secs(2);
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(30);

const ID_STOP_ALL: &str = "engine:stop-all";
const ID_HELPERS: &str = "engine:helpers";
const ID_COMFYUI_FREE: &str = "engine:comfyui-free";
const ID_UNLOAD_PREFIX: &str = "engine:unload:";
const ID_LOAD_PREFIX: &str = "engine:load:";

// ---------------------------------------------------------------------------------------------
// Configuration

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EngineUrls {
    pub(crate) omlx: String,
    pub(crate) comfyui: String,
}

impl Default for EngineUrls {
    fn default() -> Self {
        Self {
            omlx: DEFAULT_OMLX_URL.to_string(),
            comfyui: DEFAULT_COMFYUI_URL.to_string(),
        }
    }
}

/// A bind address is not a connect address: wildcards mean "this machine".
fn connect_host(host: Option<&str>) -> String {
    match host.map(str::trim) {
        None | Some("") | Some("0.0.0.0") | Some("::") | Some("[::]") => "127.0.0.1".to_string(),
        Some(other) => other.to_string(),
    }
}

fn port_of(table: &toml::Table, section: &str, key: &str, default: u16) -> u16 {
    table
        .get(section)
        .and_then(|s| s.get(key))
        .and_then(|p| p.as_integer())
        .and_then(|p| u16::try_from(p).ok())
        .filter(|p| *p != 0)
        .unwrap_or(default)
}

fn host_of<'a>(table: &'a toml::Table, section: &str) -> Option<&'a str> {
    table.get(section)?.get("host")?.as_str()
}

/// Unparseable or missing config falls back to the defaults, never to an error: the tray
/// should still show the engines on their standard ports.
pub(crate) fn parse_urls(text: &str) -> EngineUrls {
    let Ok(table) = text.parse::<toml::Table>() else {
        return EngineUrls::default();
    };
    EngineUrls {
        omlx: format!(
            "http://{}:{}",
            connect_host(host_of(&table, "omlx")),
            port_of(&table, "omlx", "port", 8843)
        ),
        comfyui: format!(
            "http://{}:{}",
            connect_host(host_of(&table, "comfyui")),
            port_of(&table, "comfyui", "port", 8844)
        ),
    }
}

fn config_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("UNSLOTH_ENGINES_CONFIG") {
        return Some(PathBuf::from(path));
    }
    let home = match std::env::var_os("UNSLOTH_ENGINES_HOME") {
        Some(home) => PathBuf::from(home),
        None => dirs::home_dir()?.join(".unsloth").join("engines"),
    };
    Some(home.join("engines.toml"))
}

pub(crate) fn load_urls() -> EngineUrls {
    config_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| parse_urls(&text))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// Launch failures

/// What the engine wrappers leave in `<engines home>/<name>.fail` when a precondition (missing venv,
/// config or model, port in use) stops an engine from starting: `{"reason", "ts", "count"}`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Failure {
    reason: String,
    count: u64,
}

/// Wrappers and the backend resolve the home the same way: `UNSLOTH_ENGINES_HOME`, else
/// `~/.unsloth/engines`. (`UNSLOTH_ENGINES_CONFIG` only moves the toml.)
pub(crate) fn engines_home() -> Option<PathBuf> {
    match std::env::var_os("UNSLOTH_ENGINES_HOME") {
        Some(home) => Some(PathBuf::from(home)),
        None => Some(dirs::home_dir()?.join(".unsloth").join("engines")),
    }
}

/// None for anything that is not a complete marker: a half-written or foreign file is no failure.
pub(crate) fn parse_failure(text: &str) -> Option<Failure> {
    let body: Value = serde_json::from_str(text).ok()?;
    let reason = body.get("reason")?.as_str()?.trim();
    body.get("ts")?.as_f64()?;
    let count = body.get("count")?.as_u64().filter(|c| *c >= 1)?;
    if reason.is_empty() {
        return None;
    }
    Some(Failure {
        reason: reason.to_string(),
        count,
    })
}

fn read_failure(name: &str) -> Option<Failure> {
    let path = engines_home()?.join(format!("{name}.fail"));
    parse_failure(&std::fs::read_to_string(path).ok()?)
}

/// The deadline the backend publishes in `<engines home>/comfyui-idle.json` (`{"free_at": epoch
/// seconds}`) while the "Free ComfyUI after idle" timer is pending. The tray cannot ask the backend
/// (its API needs the owner's token, and the tray must work with the backend stopped), so it reads the
/// same file the way it reads the `*.fail` markers.
pub(crate) fn parse_idle_marker(text: &str) -> Option<f64> {
    let free_at = serde_json::from_str::<Value>(text)
        .ok()?
        .get("free_at")?
        .as_f64()?;
    free_at.is_finite().then_some(free_at)
}

/// Seconds until the idle free; None once the deadline has passed (a backend that died leaves a stale
/// file behind, and a frozen "frees in 0s" would be wrong).
pub(crate) fn idle_remaining_s(free_at: f64, now_epoch_s: f64) -> Option<f64> {
    let remaining = free_at - now_epoch_s;
    (remaining > 0.0).then_some(remaining)
}

fn read_idle_free_at() -> Option<f64> {
    parse_idle_marker(&std::fs::read_to_string(engines_home()?.join("comfyui-idle.json")).ok()?)
}

const FAILURE_REASON_MAX: usize = 90;

/// "oMLX failing: <reason>", the reason cut to fit a menu row.
fn failing_label(engine: &str, failure: &Failure) -> String {
    let mut reason: String = failure.reason.chars().take(FAILURE_REASON_MAX).collect();
    if failure.reason.chars().count() > FAILURE_REASON_MAX {
        reason.push_str("...");
    }
    if failure.count > 1 {
        format!("{engine} failing: {reason} (x{})", failure.count)
    } else {
        format!("{engine} failing: {reason}")
    }
}

// ---------------------------------------------------------------------------------------------
// Status parsing

#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct OmlxModel {
    id: String,
    model_path: String,
    loaded: bool,
    is_loading: bool,
    pinned: bool,
    engine_type: Option<String>,
    is_helper: bool,
    alias: Option<String>,
    /// Epoch seconds of the last load or request, as oMLX reports it.
    last_access: Option<f64>,
    /// Seconds before the idle unload; set by `apply_idle_ttl`, None while pinned or untimed.
    idle_remaining_s: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct OmlxStatus {
    reachable: bool,
    models: Vec<OmlxModel>,
    memory_bytes: u64,
    /// The launch failure marker, read only while the engine is not answering.
    failure: Option<Failure>,
}

fn str_field(row: &Value, key: &str) -> Option<String> {
    row.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

pub(crate) fn parse_omlx_status(body: &Value) -> Option<OmlxStatus> {
    let rows = body.get("models")?.as_array()?;
    let models = rows
        .iter()
        .filter_map(|row| {
            Some(OmlxModel {
                id: str_field(row, "id")?,
                model_path: str_field(row, "model_path").unwrap_or_default(),
                loaded: row.get("loaded").and_then(Value::as_bool).unwrap_or(false),
                is_loading: row
                    .get("is_loading")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                pinned: row.get("pinned").and_then(Value::as_bool).unwrap_or(false),
                engine_type: str_field(row, "engine_type"),
                is_helper: row
                    .get("is_helper")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                alias: str_field(row, "model_alias"),
                last_access: row.get("last_access").and_then(Value::as_f64),
                idle_remaining_s: None,
            })
        })
        .collect();
    Some(OmlxStatus {
        reachable: true,
        models,
        memory_bytes: body
            .get("current_model_memory")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        failure: None,
    })
}

/// ComfyUI as `GET /queue` shows it.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct ComfyStatus {
    reachable: bool,
    running: usize,
    pending: usize,
    /// The launch failure marker, read only while ComfyUI is not answering.
    failure: Option<Failure>,
    /// Resident memory of the ComfyUI process in bytes, when it could be read.
    resident_bytes: Option<u64>,
    /// Seconds before Studio frees ComfyUI's models after idle; set only while the queue is empty.
    idle_free_remaining_s: Option<f64>,
}

/// `(running, pending)` from ComfyUI's `/queue` body; None for anything that is not that shape.
pub(crate) fn parse_comfyui_queue(body: &Value) -> Option<(usize, usize)> {
    let running = body.get("queue_running")?.as_array()?.len();
    let pending = body.get("queue_pending")?.as_array()?.len();
    Some((running, pending))
}

/// The global idle timeout and each model directory's own `ttl_seconds`, from oMLX's admin API.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct OmlxTtl {
    global: Option<u64>,
    per_model: std::collections::HashMap<String, u64>,
}

pub(crate) fn parse_omlx_ttl(global_settings: &Value, models: &Value) -> OmlxTtl {
    let positive = |v: Option<&Value>| v.and_then(Value::as_u64).filter(|t| *t > 0);
    let global = positive(
        global_settings
            .get("idle_timeout")
            .and_then(|t| t.get("idle_timeout_seconds")),
    );
    let per_model = models
        .get("models")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| {
                    let id = str_field(row, "id")?;
                    let ttl = positive(row.get("settings").and_then(|s| s.get("ttl_seconds")))?;
                    Some((id, ttl))
                })
                .collect()
        })
        .unwrap_or_default();
    OmlxTtl { global, per_model }
}

/// Mirrors `EnginePool.check_ttl_expirations`: a loaded, unpinned model unloads once idle for its
/// own TTL (else the global one), counted from `last_access`.
pub(crate) fn apply_idle_ttl(status: &mut OmlxStatus, ttl: &OmlxTtl, now_epoch_s: f64) {
    let snapshot = status.models.clone();
    for model in status.models.iter_mut() {
        if !model.loaded || model.pinned || model.is_loading {
            continue;
        }
        let dir = resolve_dir(&snapshot, model);
        let limit = ttl.per_model.get(&dir).or(ttl.global.as_ref());
        let last = model.last_access.or_else(|| {
            snapshot
                .iter()
                .find(|m| m.id == dir)
                .and_then(|m| m.last_access)
        });
        if let (Some(limit), Some(last)) = (limit, last) {
            model.idle_remaining_s = Some((*limit as f64 - (now_epoch_s - last)).max(0.0));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Model grouping (mirrors OmlxClient.resolve_dir in the backend)

fn is_chat(model: &OmlxModel) -> bool {
    match model.engine_type.as_deref() {
        Some("embedding") => false,
        Some(_) => !model.is_helper,
        None => !model.is_helper && !model.id.to_ascii_lowercase().contains("embed"),
    }
}

/// Directory id the admin load endpoint accepts for a row: the id equal to the basename of the
/// shared `model_path`, else the first sibling without a profile suffix, else the row itself.
fn resolve_dir(models: &[OmlxModel], row: &OmlxModel) -> String {
    if row.model_path.is_empty() {
        return row.id.clone();
    }
    let siblings: Vec<&OmlxModel> = models
        .iter()
        .filter(|m| m.model_path == row.model_path)
        .collect();
    let base = row
        .model_path
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("");
    if let Some(found) = siblings.iter().find(|m| m.id == base) {
        return found.id.clone();
    }
    if let Some(found) = siblings.iter().find(|m| !m.id.contains(':')) {
        return found.id.clone();
    }
    row.id.clone()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModelEntry {
    dir: String,
    label: String,
}

/// One entry per model directory (aliases and profiles collapse), chat models only,
/// split into loaded and loadable.
fn model_entries(status: &OmlxStatus) -> (Vec<ModelEntry>, Vec<ModelEntry>) {
    let mut seen: Vec<String> = Vec::new();
    let mut loaded = Vec::new();
    let mut idle = Vec::new();
    for row in status.models.iter().filter(|m| is_chat(m)) {
        let dir = resolve_dir(&status.models, row);
        if seen.contains(&dir) {
            continue;
        }
        seen.push(dir.clone());
        let group: Vec<&OmlxModel> = status
            .models
            .iter()
            .filter(|m| resolve_dir(&status.models, m) == dir)
            .collect();
        let label = group
            .iter()
            .find_map(|m| m.alias.clone())
            .unwrap_or_else(|| dir.clone());
        let entry = ModelEntry { dir, label };
        if group.iter().any(|m| m.loaded) {
            loaded.push(entry);
        } else {
            idle.push(entry);
        }
    }
    (loaded, idle)
}

/// Every loaded directory including pinned and embedding models, for "Stop all engines".
pub(crate) fn all_loaded_dirs(status: &OmlxStatus) -> Vec<String> {
    let mut dirs: Vec<String> = Vec::new();
    for row in status.models.iter().filter(|m| m.loaded) {
        let dir = resolve_dir(&status.models, row);
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    dirs
}

// ---------------------------------------------------------------------------------------------
// View model

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct View {
    omlx_label: String,
    unloads: Vec<ModelEntry>,
    loads: Vec<ModelEntry>,
    load_enabled: bool,
    stop_all_enabled: bool,
    helpers_label: String,
    comfyui_label: String,
    /// The ComfyUI rows show while the user wants ComfyUI or it answers anyway.
    comfyui_visible: bool,
    comfyui_free_enabled: bool,
}

fn gib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn fmt_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    if total >= 60 {
        format!("{}m {:02}s", total / 60, total % 60)
    } else {
        format!("{total}s")
    }
}

fn omlx_label(status: &OmlxStatus) -> String {
    if !status.reachable {
        return match &status.failure {
            Some(failure) => failing_label("oMLX", failure),
            None => "oMLX: not running".to_string(),
        };
    }
    if let Some(loading) = status.models.iter().find(|m| m.is_loading) {
        let name = loading.alias.as_deref().unwrap_or(&loading.id);
        return format!("oMLX: loading {name}");
    }
    let loaded = all_loaded_dirs(status).len();
    let mut label = format!("oMLX: {loaded} loaded ({} GB)", gib(status.memory_bytes));
    let soonest = status
        .models
        .iter()
        .filter(|m| m.loaded)
        .filter_map(|m| m.idle_remaining_s)
        .reduce(f64::min);
    if let Some(idle) = soonest {
        label.push_str(&format!(", unloads in {}", fmt_duration(idle)));
    } else if loaded > 0 && status.models.iter().filter(|m| m.loaded).all(|m| m.pinned) {
        label.push_str(", pinned");
    }
    label
}

fn comfyui_label(status: &ComfyStatus) -> String {
    if !status.reachable {
        return match &status.failure {
            Some(failure) => failing_label("ComfyUI", failure),
            None => "ComfyUI: not running".to_string(),
        };
    }
    let memory = status
        .resident_bytes
        .map(|bytes| format!(", {} GB in memory", gib(bytes)))
        .unwrap_or_default();
    if status.running == 0 && status.pending == 0 {
        let countdown = status
            .idle_free_remaining_s
            .map(|seconds| format!(", frees in {}", fmt_duration(seconds)))
            .unwrap_or_default();
        format!("ComfyUI: idle{memory}{countdown}")
    } else {
        // the running job is not "queued": N counts the jobs waiting behind it
        format!("ComfyUI: generating ({} queued){memory}", status.pending)
    }
}

/// Why "Unload ComfyUI models" is refused right now, or None when the queue is empty. The tray
/// never interrupts a job: the user cancels it in Settings > Engines.
pub(crate) fn unload_refusal(running: usize, pending: usize) -> Option<String> {
    match (running, pending) {
        (0, 0) => None,
        (_, 0) => Some("ComfyUI: a job is running, unload after it ends".to_string()),
        (_, queued) => Some(format!(
            "ComfyUI: busy ({queued} queued), unload after the queue empties"
        )),
    }
}

/// The pid `lsof -t` prints first.
pub(crate) fn parse_pid(text: &str) -> Option<u32> {
    text.lines().find_map(|line| line.trim().parse().ok())
}

/// Resident memory in bytes from `ps -o rss=` (kilobytes).
pub(crate) fn parse_rss_bytes(text: &str) -> Option<u64> {
    text.trim()
        .parse::<u64>()
        .ok()
        .filter(|kb| *kb > 0)
        .map(|kb| kb * 1024)
}

/// The status line. While the helpers run in `with_app` mode it says they stop when Unsloth quits.
fn helpers_label(state: HelperState, lifetime: EngineLifetime) -> String {
    let text = match state {
        HelperState::Enabled => "Engines: running as background helpers",
        HelperState::RequiresApproval => "Engines: approve the helpers in Login Items",
        HelperState::NotRegistered => "Engines: not running as helpers (enable in Settings)",
        HelperState::Partial => "Engines: helpers partly enabled",
        HelperState::NotFound => "Engines: helper agents not bundled",
        HelperState::Unsupported => "Engines: background helpers need macOS",
    };
    match (state, lifetime) {
        (HelperState::Enabled | HelperState::Partial, EngineLifetime::WithApp) => {
            format!("{text} (stop on quit)")
        }
        _ => text.to_string(),
    }
}

pub(crate) fn build_view(
    omlx: &OmlxStatus,
    comfyui: &ComfyStatus,
    comfyui_wanted: bool,
    helpers: HelperState,
    lifetime: EngineLifetime,
) -> View {
    let (unloads, loads) = if omlx.reachable {
        model_entries(omlx)
    } else {
        (Vec::new(), Vec::new())
    };
    View {
        omlx_label: omlx_label(omlx),
        unloads,
        stop_all_enabled: omlx.reachable,
        load_enabled: omlx.reachable,
        loads,
        helpers_label: helpers_label(helpers, lifetime),
        comfyui_label: comfyui_label(comfyui),
        comfyui_visible: comfyui_wanted || comfyui.reachable,
        comfyui_free_enabled: comfyui.reachable && comfyui.running == 0 && comfyui.pending == 0,
    }
}

// ---------------------------------------------------------------------------------------------
// Actions

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    StopAll,
    FreeComfyui,
    OpenLoginItems,
    Unload(String),
    Load(String),
}

pub(crate) fn parse_action(id: &str) -> Option<Action> {
    match id {
        ID_STOP_ALL => Some(Action::StopAll),
        ID_COMFYUI_FREE => Some(Action::FreeComfyui),
        ID_HELPERS => Some(Action::OpenLoginItems),
        _ => id
            .strip_prefix(ID_UNLOAD_PREFIX)
            .map(|dir| Action::Unload(dir.to_string()))
            .or_else(|| {
                id.strip_prefix(ID_LOAD_PREFIX)
                    .map(|dir| Action::Load(dir.to_string()))
            }),
    }
}

/// Percent-encode one path segment (model ids carry `:` and `/`).
pub(crate) fn encode_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

async fn get_json(base: &str, path: &str) -> Option<Value> {
    let client = loopback_http::client(STATUS_TIMEOUT).ok()?;
    let response = client.get(format!("{base}{path}")).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json().await.ok()
}

async fn post(base: &str, path: &str, timeout: Duration) -> Result<(), String> {
    let client = loopback_http::client(timeout).map_err(|e| e.to_string())?;
    let response = client
        .post(format!("{base}{path}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}", response.status().as_u16()))
    }
}

async fn fetch_omlx(urls: &EngineUrls) -> OmlxStatus {
    let mut status = match get_json(&urls.omlx, "/v1/models/status")
        .await
        .and_then(|body| parse_omlx_status(&body))
    {
        Some(status) => status,
        None => {
            return OmlxStatus {
                failure: read_failure("omlx"),
                ..OmlxStatus::default()
            }
        }
    };
    // The idle TTL lives in the admin API, so it is only asked for when a timer could show.
    if status.models.iter().any(|m| m.loaded && !m.pinned) {
        let (global, models) = tokio::join!(
            get_json(&urls.omlx, "/admin/api/global-settings"),
            get_json(&urls.omlx, "/admin/api/models")
        );
        if let (Some(global), Some(models)) = (global, models) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            apply_idle_ttl(&mut status, &parse_omlx_ttl(&global, &models), now);
        }
    }
    status
}

async fn fetch_comfyui(urls: &EngineUrls) -> ComfyStatus {
    match get_json(&urls.comfyui, "/queue")
        .await
        .and_then(|body| parse_comfyui_queue(&body))
    {
        Some((running, pending)) => ComfyStatus {
            reachable: true,
            running,
            pending,
            failure: None,
            resident_bytes: resident_bytes_of_port(port_of_url(&urls.comfyui)).await,
            idle_free_remaining_s: if running == 0 && pending == 0 {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                read_idle_free_at().and_then(|free_at| idle_remaining_s(free_at, now))
            } else {
                None
            },
        },
        None => ComfyStatus {
            failure: read_failure("comfyui"),
            ..ComfyStatus::default()
        },
    }
}

fn port_of_url(url: &str) -> Option<u16> {
    url.rsplit(':').next()?.trim_end_matches('/').parse().ok()
}

/// Resident memory of the process listening on `port`, best effort (None when `lsof` or `ps` fail).
async fn resident_bytes_of_port(port: Option<u16>) -> Option<u64> {
    let port = port?;
    tauri::async_runtime::spawn_blocking(move || {
        let run = |program: &str, args: &[String]| -> Option<String> {
            let output = std::process::Command::new(program)
                .args(args)
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
        };
        let pid = parse_pid(&run(
            "lsof",
            &[
                "-nP".to_string(),
                format!("-iTCP:{port}"),
                "-sTCP:LISTEN".to_string(),
                "-t".to_string(),
            ],
        )?)?;
        parse_rss_bytes(&run(
            "ps",
            &[
                "-o".to_string(),
                "rss=".to_string(),
                "-p".to_string(),
                pid.to_string(),
            ],
        )?)
    })
    .await
    .ok()
    .flatten()
}

/// Unload ComfyUI's models and free its memory. Refused, without interrupting anything, when a
/// job is running or queued (the status line already disabled the item, this covers a job that
/// arrived since the last poll). The refusal text is returned as the error.
async fn free_comfyui(urls: &EngineUrls) -> Result<(), String> {
    if let Some((running, pending)) = get_json(&urls.comfyui, "/queue")
        .await
        .and_then(|body| parse_comfyui_queue(&body))
    {
        if let Some(refusal) = unload_refusal(running, pending) {
            return Err(refusal);
        }
    }
    let client = loopback_http::client(UNLOAD_TIMEOUT).map_err(|e| e.to_string())?;
    let response = client
        .post(format!("{}/free", urls.comfyui))
        .json(&serde_json::json!({"unload_models": true, "free_memory": true}))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}", response.status().as_u16()))
    }
}

async fn load_model(urls: &EngineUrls, dir: &str) -> Result<(), String> {
    let path = format!("/admin/api/models/{}/load", encode_segment(dir));
    post(&urls.omlx, &path, LOAD_TIMEOUT).await
}

async fn unload_model(urls: &EngineUrls, dir: &str) -> Result<(), String> {
    let path = format!("/v1/models/{}/unload", encode_segment(dir));
    post(&urls.omlx, &path, UNLOAD_TIMEOUT).await
}

async fn stop_all(urls: &EngineUrls) -> Result<(), String> {
    let mut errors = Vec::new();
    for dir in all_loaded_dirs(&fetch_omlx(urls).await) {
        if let Err(error) = unload_model(urls, &dir).await {
            errors.push(format!("{dir}: {error}"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

// ---------------------------------------------------------------------------------------------
// Menu wiring

#[derive(Default)]
struct Dynamic {
    unload_items: Vec<MenuItem<Wry>>,
    unload_keys: Vec<ModelEntry>,
    load_items: Vec<MenuItem<Wry>>,
    load_keys: Option<Vec<ModelEntry>>,
    last: Option<View>,
    /// Whether the ComfyUI rows are in the menu right now.
    comfyui_shown: bool,
}

struct EngineTray {
    menu: Menu<Wry>,
    omlx_label: MenuItem<Wry>,
    comfyui_label: MenuItem<Wry>,
    comfyui_free: MenuItem<Wry>,
    load_menu: Submenu<Wry>,
    stop_all: MenuItem<Wry>,
    helpers: MenuItem<Wry>,
    dynamic: Mutex<Dynamic>,
    busy: AtomicBool,
}

impl EngineTray {
    fn apply(&self, app: &AppHandle, view: &View) {
        let Ok(mut dynamic) = self.dynamic.lock() else {
            return;
        };
        let previous = dynamic.last.clone();
        let changed = |f: fn(&View) -> bool| previous.as_ref().is_none_or(|p| f(p) != f(view));
        if previous
            .as_ref()
            .is_none_or(|p| p.omlx_label != view.omlx_label)
        {
            let _ = self.omlx_label.set_text(&view.omlx_label);
        }
        if changed(|v| v.stop_all_enabled) {
            let _ = self.stop_all.set_enabled(view.stop_all_enabled);
        }
        if changed(|v| v.load_enabled) {
            let _ = self.load_menu.set_enabled(view.load_enabled);
        }
        if previous
            .as_ref()
            .is_none_or(|p| p.helpers_label != view.helpers_label)
        {
            let _ = self.helpers.set_text(&view.helpers_label);
        }
        if previous
            .as_ref()
            .is_none_or(|p| p.comfyui_label != view.comfyui_label)
        {
            let _ = self.comfyui_label.set_text(&view.comfyui_label);
        }
        if changed(|v| v.comfyui_free_enabled) {
            let _ = self.comfyui_free.set_enabled(view.comfyui_free_enabled);
        }
        if view.comfyui_visible != dynamic.comfyui_shown {
            self.show_comfyui(&mut dynamic, view.comfyui_visible);
        }
        if dynamic.unload_keys != view.unloads || previous.is_none() {
            self.replace_unloads(app, &mut dynamic, &view.unloads);
        }
        if dynamic.load_keys.as_ref() != Some(&view.loads) {
            self.replace_loads(app, &mut dynamic, &view.loads);
        }
        dynamic.last = Some(view.clone());
    }

    /// Show a refusal on the ComfyUI status line until the next poll replaces it.
    fn comfyui_notice(&self, text: &str) {
        let Ok(mut dynamic) = self.dynamic.lock() else {
            return;
        };
        let _ = self.comfyui_label.set_text(text);
        if let Some(last) = dynamic.last.as_mut() {
            last.comfyui_label = text.to_string();
        }
    }

    /// Insert or remove the two ComfyUI rows (Tauri menu items cannot be hidden), right after
    /// "Stop all engines".
    fn show_comfyui(&self, dynamic: &mut Dynamic, show: bool) {
        if show {
            let position = self
                .menu
                .items()
                .ok()
                .and_then(|items| items.iter().position(|i| i.id() == self.stop_all.id()))
                .map_or(0, |p| p + 1);
            let ok = self.menu.insert(&self.comfyui_label, position).is_ok()
                && self.menu.insert(&self.comfyui_free, position + 1).is_ok();
            dynamic.comfyui_shown = ok;
        } else {
            let _ = self.menu.remove(&self.comfyui_free);
            let _ = self.menu.remove(&self.comfyui_label);
            dynamic.comfyui_shown = false;
        }
    }

    fn replace_unloads(&self, app: &AppHandle, dynamic: &mut Dynamic, entries: &[ModelEntry]) {
        for item in dynamic.unload_items.drain(..) {
            let _ = self.menu.remove(&item);
        }
        let position = self
            .menu
            .items()
            .ok()
            .and_then(|items| items.iter().position(|i| i.id() == self.omlx_label.id()))
            .map_or(0, |p| p + 1);
        let mut created = Vec::new();
        for (offset, entry) in entries.iter().enumerate() {
            let id = format!("{ID_UNLOAD_PREFIX}{}", entry.dir);
            let text = format!("Unload {}", entry.label);
            let Ok(item) = MenuItemBuilder::with_id(id, text).build(app) else {
                continue;
            };
            if self.menu.insert(&item, position + offset).is_ok() {
                created.push(item);
            }
        }
        dynamic.unload_items = created;
        dynamic.unload_keys = entries.to_vec();
    }

    fn replace_loads(&self, app: &AppHandle, dynamic: &mut Dynamic, entries: &[ModelEntry]) {
        for item in dynamic.load_items.drain(..) {
            let _ = self.load_menu.remove(&item);
        }
        let mut created = Vec::new();
        if entries.is_empty() {
            if let Ok(item) = MenuItemBuilder::with_id("engine:load-none", "No models to load")
                .enabled(false)
                .build(app)
            {
                if self.load_menu.append(&item).is_ok() {
                    created.push(item);
                }
            }
        }
        for entry in entries {
            let id = format!("{ID_LOAD_PREFIX}{}", entry.dir);
            let Ok(item) = MenuItemBuilder::with_id(id, &entry.label).build(app) else {
                continue;
            };
            if self.load_menu.append(&item).is_ok() {
                created.push(item);
            }
        }
        dynamic.load_items = created;
        dynamic.load_keys = Some(entries.to_vec());
    }
}

/// Add the engine items to the tray menu (before Quit) and start the 5 s poll.
pub(crate) fn install(app: &tauri::App, menu: &Menu<Wry>) -> tauri::Result<()> {
    let omlx_label = MenuItemBuilder::with_id("engine:omlx-label", "oMLX: checking")
        .enabled(false)
        .build(app)?;
    let load_menu = SubmenuBuilder::with_id(app, "engine:load-menu", "Load model")
        .enabled(false)
        .build()?;
    let stop_all = MenuItemBuilder::with_id(ID_STOP_ALL, "Stop all engines")
        .enabled(false)
        .build(app)?;
    // Not in the menu until the poll says ComfyUI is wanted or running (see `show_comfyui`).
    let comfyui_label = MenuItemBuilder::with_id("engine:comfyui-label", "ComfyUI: checking")
        .enabled(false)
        .build(app)?;
    let comfyui_free = MenuItemBuilder::with_id(ID_COMFYUI_FREE, "Unload ComfyUI models")
        .enabled(false)
        .build(app)?;
    let helpers = MenuItemBuilder::with_id(
        ID_HELPERS,
        helpers_label(HelperState::NotRegistered, EngineLifetime::default()),
    )
    .build(app)?;
    let sep_top = PredefinedMenuItem::separator(app)?;
    let sep_end = PredefinedMenuItem::separator(app)?;
    let items: [&dyn IsMenuItem<Wry>; 6] = [
        &sep_top,
        &omlx_label,
        &load_menu,
        &stop_all,
        &helpers,
        &sep_end,
    ];
    // The caller's menu is [open, toggle, quit]: these go between toggle and Quit.
    menu.insert_items(&items, 2)?;
    app.manage(EngineTray {
        menu: menu.clone(),
        omlx_label,
        comfyui_label,
        comfyui_free,
        load_menu,
        stop_all,
        helpers,
        dynamic: Mutex::new(Dynamic::default()),
        busy: AtomicBool::new(false),
    });

    let handle = app.handle().clone();
    tauri::async_runtime::spawn(async move {
        loop {
            let urls = load_urls();
            let (omlx, comfyui) = tokio::join!(fetch_omlx(&urls), fetch_comfyui(&urls));
            let (helpers, comfyui_wanted) = engine_helpers::tray_state();
            let view = build_view(
                &omlx,
                &comfyui,
                comfyui_wanted,
                helpers,
                engine_lifetime::load().lifetime,
            );
            if let Some(tray) = handle.try_state::<EngineTray>() {
                tray.apply(&handle, &view);
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    });
    Ok(())
}

/// Route a tray menu click. Ignores ids that are not the engine items.
pub(crate) fn on_menu_event(app: &AppHandle, id: &str) {
    let Some(action) = parse_action(id) else {
        return;
    };
    if action == Action::OpenLoginItems {
        engine_helpers::open_login_items_settings();
        return;
    }
    let Some(tray) = app.try_state::<EngineTray>() else {
        return;
    };
    if tray.busy.swap(true, Ordering::SeqCst) {
        info!("engine tray: ignoring {id} while another engine action is running");
        return;
    }
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let urls = load_urls();
        let result = match &action {
            Action::StopAll => stop_all(&urls).await,
            Action::FreeComfyui => free_comfyui(&urls).await,
            Action::Unload(dir) => unload_model(&urls, dir).await,
            Action::Load(dir) => load_model(&urls, dir).await,
            Action::OpenLoginItems => Ok(()),
        };
        if let Err(error) = result {
            warn!("engine tray action {action:?} failed: {error}");
            if action == Action::FreeComfyui && error.starts_with("ComfyUI:") {
                if let Some(tray) = handle.try_state::<EngineTray>() {
                    tray.comfyui_notice(&error);
                }
            }
        }
        if let Some(tray) = handle.try_state::<EngineTray>() {
            tray.busy.store(false, Ordering::SeqCst);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const OMLX_BODY: &str = r#"{
      "final_ceiling": 33115374278, "current_model_memory": 4470007249, "loaded_count": 1,
      "models": [
        {"id": "Qwen3-Embedding-8B-4bit-DWQ", "model_path": "/m/mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
         "loaded": true, "is_loading": false, "pinned": true, "engine_type": "embedding", "is_helper": false},
        {"id": "ukisai--Swift-1.5-27B-oQ4e-mtp", "model_path": "/m/ukisai--Swift-1.5-27B-oQ4e-mtp",
         "loaded": false, "is_loading": false, "engine_type": "vlm", "is_helper": false},
        {"id": "swift-1.5-27b", "model_path": "/m/ukisai--Swift-1.5-27B-oQ4e-mtp",
         "loaded": false, "engine_type": "vlm", "model_alias": "swift-1.5-27b"},
        {"id": "swift-1.5-27b:fast", "model_path": "/m/ukisai--Swift-1.5-27B-oQ4e-mtp",
         "loaded": false, "engine_type": "vlm"},
        {"id": "drafter", "model_path": "/m/drafter", "loaded": false, "engine_type": "dflash", "is_helper": true}
      ]
    }"#;

    fn omlx(body: &str) -> OmlxStatus {
        parse_omlx_status(&serde_json::from_str(body).unwrap()).unwrap()
    }

    #[test]
    fn urls_default_when_config_missing_or_invalid() {
        assert_eq!(parse_urls(""), EngineUrls::default());
        assert_eq!(parse_urls("not = [valid"), EngineUrls::default());
        assert_eq!(EngineUrls::default().omlx, "http://127.0.0.1:8843");
        assert_eq!(EngineUrls::default().comfyui, "http://127.0.0.1:8844");
    }

    #[test]
    fn the_comfyui_url_follows_its_own_section() {
        assert_eq!(
            parse_urls("[comfyui]\nport = 18844\n").comfyui,
            "http://127.0.0.1:18844"
        );
        assert_eq!(
            parse_urls("[comfyui]\nhost = \"::\"\nport = 9000\n").comfyui,
            "http://127.0.0.1:9000"
        );
        assert_eq!(
            parse_urls("[comfyui]\nhost = \"localhost\"\n").comfyui,
            "http://localhost:8844"
        );
        // the oMLX section never moves ComfyUI, and a bad port falls back
        assert_eq!(
            parse_urls("[omlx]\nport = 9000\n").comfyui,
            "http://127.0.0.1:8844"
        );
        assert_eq!(
            parse_urls("[comfyui]\nport = 0\n").comfyui,
            "http://127.0.0.1:8844"
        );
        assert_eq!(
            parse_urls("[comfyui]\nport = \"x\"\n").comfyui,
            "http://127.0.0.1:8844"
        );
    }

    #[test]
    fn urls_follow_ports_and_wildcard_hosts() {
        assert_eq!(
            parse_urls("[omlx]\nport = 18843\nhost = \"0.0.0.0\"\n").omlx,
            "http://127.0.0.1:18843"
        );
        assert_eq!(
            parse_urls("[omlx]\nhost = \"192.168.3.78\"\nport = 9000\n").omlx,
            "http://192.168.3.78:9000"
        );
        assert_eq!(parse_urls("[omlx]\nport = 0\n"), EngineUrls::default());
    }

    #[test]
    fn default_engines_toml_parses_to_the_standard_ports() {
        let text = include_str!("../../scripts/engines/engines.default.toml");
        assert_eq!(parse_urls(text), EngineUrls::default());
    }

    #[test]
    fn omlx_status_parses_live_shape() {
        let status = omlx(OMLX_BODY);
        assert!(status.reachable);
        assert_eq!(status.models.len(), 5);
        assert_eq!(status.memory_bytes, 4_470_007_249);
        assert!(parse_omlx_status(&json!({"models": "no"})).is_none());
    }

    #[test]
    fn aliases_collapse_to_one_directory_and_embedding_is_excluded() {
        let status = omlx(OMLX_BODY);
        let (loaded, loadable) = model_entries(&status);
        assert!(
            loaded.is_empty(),
            "the only loaded model is the embedding one"
        );
        assert_eq!(
            loadable,
            vec![ModelEntry {
                dir: "ukisai--Swift-1.5-27B-oQ4e-mtp".into(),
                label: "swift-1.5-27b".into()
            }]
        );
        let alias_row = status
            .models
            .iter()
            .find(|m| m.id == "swift-1.5-27b:fast")
            .unwrap();
        assert_eq!(
            resolve_dir(&status.models, alias_row),
            "ukisai--Swift-1.5-27B-oQ4e-mtp"
        );
    }

    #[test]
    fn stop_all_unloads_pinned_models_too() {
        let status = omlx(OMLX_BODY);
        assert_eq!(
            all_loaded_dirs(&status),
            vec!["Qwen3-Embedding-8B-4bit-DWQ".to_string()]
        );
    }

    #[test]
    fn loaded_chat_model_becomes_an_unload_entry() {
        let mut status = omlx(OMLX_BODY);
        for row in status
            .models
            .iter_mut()
            .filter(|m| m.model_path.contains("Swift"))
        {
            row.loaded = true;
        }
        let (loaded, loadable) = model_entries(&status);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].label, "swift-1.5-27b");
        assert!(loadable.is_empty());
        let label = omlx_label(&status);
        assert_eq!(label, "oMLX: 2 loaded (4.2 GB)");
    }

    #[test]
    fn omlx_labels() {
        assert_eq!(omlx_label(&OmlxStatus::default()), "oMLX: not running");
        let mut status = omlx(OMLX_BODY);
        status.models[1].is_loading = true;
        assert_eq!(
            omlx_label(&status),
            "oMLX: loading ukisai--Swift-1.5-27B-oQ4e-mtp"
        );
    }

    #[test]
    fn omlx_label_counts_down_to_the_idle_unload() {
        let mut status = omlx(OMLX_BODY);
        status.models[1].loaded = true;
        status.models[1].last_access = Some(1_000.0);
        let global = json!({"idle_timeout": {"idle_timeout_seconds": 600}});
        let listing = json!({"models": [
            {"id": "ukisai--Swift-1.5-27B-oQ4e-mtp", "settings": {"ttl_seconds": null}}
        ]});
        apply_idle_ttl(&mut status, &parse_omlx_ttl(&global, &listing), 1_348.0);
        assert_eq!(status.models[1].idle_remaining_s, Some(252.0));
        assert_eq!(
            omlx_label(&status),
            "oMLX: 2 loaded (4.2 GB), unloads in 4m 12s"
        );
        // pinned embedding model never gets a timer
        assert_eq!(status.models[0].idle_remaining_s, None);
    }

    #[test]
    fn per_model_ttl_wins_and_expired_clamps_to_zero() {
        let mut status = omlx(OMLX_BODY);
        status.models[1].loaded = true;
        status.models[1].last_access = Some(1_000.0);
        let global = json!({"idle_timeout": {"idle_timeout_seconds": 600}});
        let listing = json!({"models": [
            {"id": "ukisai--Swift-1.5-27B-oQ4e-mtp", "settings": {"ttl_seconds": 60}}
        ]});
        apply_idle_ttl(&mut status, &parse_omlx_ttl(&global, &listing), 1_500.0);
        assert_eq!(status.models[1].idle_remaining_s, Some(0.0));
    }

    #[test]
    fn omlx_label_without_ttl_or_with_only_pinned_models() {
        let mut status = omlx(OMLX_BODY);
        assert_eq!(omlx_label(&status), "oMLX: 1 loaded (4.2 GB), pinned");
        status.models[1].loaded = true;
        apply_idle_ttl(&mut status, &OmlxTtl::default(), 1_000.0);
        assert_eq!(omlx_label(&status), "oMLX: 2 loaded (4.2 GB)");
    }

    fn failure(reason: &str, count: u64) -> Failure {
        Failure {
            reason: reason.to_string(),
            count,
        }
    }

    #[test]
    fn failure_marker_parses_only_when_complete() {
        assert_eq!(
            parse_failure(
                r#"{"reason": "port 8843 already in use", "ts": 1760000000.5, "count": 3}"#
            ),
            Some(failure("port 8843 already in use", 3))
        );
        assert_eq!(
            parse_failure(r#"{"reason": " venv missing ", "ts": 1760000000, "count": 1}"#),
            Some(failure("venv missing", 1))
        );
        for bad in [
            "",
            "garbage",
            "[]",
            r#"{"reason": "x", "ts": 1}"#,
            r#"{"reason": "", "ts": 1, "count": 1}"#,
            r#"{"reason": "x", "ts": "now", "count": 1}"#,
            r#"{"reason": "x", "ts": 1, "count": 0}"#,
            r#"{"reason": "x", "ts": 1, "count": -2}"#,
            r#"{"reason": 7, "ts": 1, "count": 1}"#,
        ] {
            assert_eq!(parse_failure(bad), None, "{bad}");
        }
    }

    #[test]
    fn unreachable_engines_show_why_they_are_failing() {
        let omlx_down = OmlxStatus {
            failure: Some(failure("venv missing: /x/omlx/bin/python", 1)),
            ..OmlxStatus::default()
        };
        assert_eq!(
            omlx_label(&omlx_down),
            "oMLX failing: venv missing: /x/omlx/bin/python"
        );
        let long = OmlxStatus {
            failure: Some(failure(&"a".repeat(200), 1)),
            ..OmlxStatus::default()
        };
        assert_eq!(
            omlx_label(&long),
            format!("oMLX failing: {}...", "a".repeat(FAILURE_REASON_MAX))
        );
        // a stale marker never hides a healthy engine
        let mut up = omlx(OMLX_BODY);
        up.failure = Some(failure("old", 1));
        assert!(omlx_label(&up).starts_with("oMLX: 1 loaded"));
    }

    #[test]
    fn view_enables_controls_from_reachability() {
        let down = build_view(
            &OmlxStatus::default(),
            &ComfyStatus::default(),
            false,
            HelperState::NotRegistered,
            EngineLifetime::WithApp,
        );
        assert!(!down.load_enabled && !down.stop_all_enabled);
        assert_eq!(
            down.helpers_label,
            "Engines: not running as helpers (enable in Settings)"
        );

        let up = build_view(
            &omlx(OMLX_BODY),
            &ComfyStatus::default(),
            false,
            HelperState::Enabled,
            EngineLifetime::Always,
        );
        assert!(up.load_enabled && up.stop_all_enabled);
        assert_eq!(up.helpers_label, "Engines: running as background helpers");
        assert_eq!(up.loads.len(), 1);
    }

    fn comfy(running: usize, pending: usize) -> ComfyStatus {
        ComfyStatus {
            reachable: true,
            running,
            pending,
            failure: None,
            resident_bytes: None,
            idle_free_remaining_s: None,
        }
    }

    #[test]
    fn comfyui_idle_label_counts_down_to_the_idle_free() {
        let waiting = ComfyStatus {
            idle_free_remaining_s: Some(252.0),
            ..comfy(0, 0)
        };
        assert_eq!(comfyui_label(&waiting), "ComfyUI: idle, frees in 4m 12s");
        let with_memory = ComfyStatus {
            resident_bytes: Some(27 * 1024 * 1024 * 1024),
            idle_free_remaining_s: Some(45.4),
            ..comfy(0, 0)
        };
        assert_eq!(
            comfyui_label(&with_memory),
            "ComfyUI: idle, 27.0 GB in memory, frees in 45s"
        );
        // no countdown while a job runs or waits, even if a stale value is carried
        let busy = ComfyStatus {
            idle_free_remaining_s: Some(10.0),
            ..comfy(1, 0)
        };
        assert_eq!(comfyui_label(&busy), "ComfyUI: generating (0 queued)");
    }

    #[test]
    fn the_idle_marker_parses_only_a_finite_deadline() {
        assert_eq!(
            parse_idle_marker(r#"{"free_at": 1700000300.5}"#),
            Some(1_700_000_300.5)
        );
        assert_eq!(parse_idle_marker(r#"{"free_at": 1700000300}"#), Some(1_700_000_300.0));
        for bad in [
            "",
            "not json",
            "{}",
            r#"{"free_at": "soon"}"#,
            r#"{"free_at": null}"#,
            r#"{"at": 1}"#,
            "[1]",
        ] {
            assert_eq!(parse_idle_marker(bad), None, "{bad}");
        }
    }

    #[test]
    fn a_passed_deadline_shows_no_countdown() {
        assert_eq!(idle_remaining_s(1_300.0, 1_048.0), Some(252.0));
        assert_eq!(idle_remaining_s(1_300.0, 1_300.0), None);
        assert_eq!(idle_remaining_s(1_300.0, 9_999.0), None);
    }

    #[test]
    fn comfyui_labels() {
        assert_eq!(
            comfyui_label(&ComfyStatus::default()),
            "ComfyUI: not running"
        );
        assert_eq!(comfyui_label(&comfy(0, 0)), "ComfyUI: idle");
        assert_eq!(
            comfyui_label(&comfy(1, 0)),
            "ComfyUI: generating (0 queued)"
        );
        assert_eq!(
            comfyui_label(&comfy(1, 3)),
            "ComfyUI: generating (3 queued)"
        );
        let failing = ComfyStatus {
            failure: Some(failure(
                "port 8844 (ComfyUI) is held by StoryPress's ComfyUI (pid 7)",
                2,
            )),
            ..ComfyStatus::default()
        };
        assert_eq!(
            comfyui_label(&failing),
            "ComfyUI failing: port 8844 (ComfyUI) is held by StoryPress's ComfyUI (pid 7) (x2)"
        );
        let long = ComfyStatus {
            failure: Some(failure(&"b".repeat(200), 1)),
            ..ComfyStatus::default()
        };
        assert_eq!(
            comfyui_label(&long),
            format!("ComfyUI failing: {}...", "b".repeat(FAILURE_REASON_MAX))
        );
        // a stale marker never hides a healthy ComfyUI
        let up = ComfyStatus {
            failure: Some(failure("old", 1)),
            ..comfy(0, 0)
        };
        assert_eq!(comfyui_label(&up), "ComfyUI: idle");
    }

    #[test]
    fn the_comfyui_queue_parses_only_its_own_shape() {
        assert_eq!(
            parse_comfyui_queue(
                &json!({"queue_running": [[0, "a"]], "queue_pending": [[1, "b"], [2, "c"]]})
            ),
            Some((1, 2))
        );
        assert_eq!(
            parse_comfyui_queue(&json!({"queue_running": [], "queue_pending": []})),
            Some((0, 0))
        );
        for bad in [
            json!({}),
            json!({"queue_running": []}),
            json!({"queue_running": 1, "queue_pending": []}),
            json!({"models": []}),
            json!([]),
        ] {
            assert_eq!(parse_comfyui_queue(&bad), None, "{bad}");
        }
    }

    #[test]
    fn the_comfyui_rows_show_when_wanted_or_reachable() {
        let view = |status: &ComfyStatus, wanted| {
            build_view(
                &OmlxStatus::default(),
                status,
                wanted,
                HelperState::NotRegistered,
                EngineLifetime::WithApp,
            )
        };
        let hidden = view(&ComfyStatus::default(), false);
        assert!(!hidden.comfyui_visible && !hidden.comfyui_free_enabled);
        // wanted but not up (starting, or failing): shown, Free disabled
        let starting = view(&ComfyStatus::default(), true);
        assert!(starting.comfyui_visible && !starting.comfyui_free_enabled);
        assert_eq!(starting.comfyui_label, "ComfyUI: not running");
        // up without being wanted (started by hand): shown and freeable
        let running = view(&comfy(0, 0), false);
        assert!(running.comfyui_visible && running.comfyui_free_enabled);
        assert_eq!(running.comfyui_label, "ComfyUI: idle");
        // a running or queued job disables the unload item; nothing is interrupted
        assert!(!view(&comfy(1, 0), false).comfyui_free_enabled);
        assert!(!view(&comfy(0, 2), false).comfyui_free_enabled);
    }

    #[test]
    fn the_comfyui_label_shows_the_memory_the_process_holds() {
        let mut idle = comfy(0, 0);
        idle.resident_bytes = Some(27 * 1024 * 1024 * 1024);
        assert_eq!(comfyui_label(&idle), "ComfyUI: idle, 27.0 GB in memory");
        let mut busy = comfy(1, 2);
        busy.resident_bytes = Some(1_610_612_736);
        assert_eq!(
            comfyui_label(&busy),
            "ComfyUI: generating (2 queued), 1.5 GB in memory"
        );
        // an engine that is not answering has no memory figure to show
        let mut down = ComfyStatus::default();
        down.resident_bytes = Some(1);
        assert_eq!(comfyui_label(&down), "ComfyUI: not running");
    }

    #[test]
    fn unloading_comfyui_models_is_refused_while_a_job_runs_or_waits() {
        assert_eq!(unload_refusal(0, 0), None);
        assert_eq!(
            unload_refusal(1, 0).as_deref(),
            Some("ComfyUI: a job is running, unload after it ends")
        );
        assert_eq!(
            unload_refusal(1, 3).as_deref(),
            Some("ComfyUI: busy (3 queued), unload after the queue empties")
        );
        assert_eq!(
            unload_refusal(0, 1).as_deref(),
            Some("ComfyUI: busy (1 queued), unload after the queue empties")
        );
        // the refusal is shown on the status line, which only the "ComfyUI:" prefix reaches
        assert!(unload_refusal(1, 0).unwrap().starts_with("ComfyUI:"));
    }

    #[test]
    fn the_process_memory_probes_parse_lsof_and_ps_output() {
        assert_eq!(parse_pid("1234\n5678\n"), Some(1234));
        assert_eq!(parse_pid("  42 \n"), Some(42));
        assert_eq!(parse_pid(""), None);
        assert_eq!(parse_pid("not a pid\n"), None);
        assert_eq!(parse_rss_bytes("  2048\n"), Some(2048 * 1024));
        assert_eq!(parse_rss_bytes("0\n"), None);
        assert_eq!(parse_rss_bytes(""), None);
        assert_eq!(port_of_url("http://127.0.0.1:8844"), Some(8844));
        assert_eq!(port_of_url("http://127.0.0.1:8844/"), Some(8844));
        assert_eq!(port_of_url("http://127.0.0.1"), None);
    }

    #[test]
    fn the_unload_item_keeps_its_id_and_parses_to_the_free_action() {
        assert_eq!(parse_action(ID_COMFYUI_FREE), Some(Action::FreeComfyui));
    }

    #[test]
    fn the_status_line_says_stop_on_quit_only_for_running_with_app_helpers() {
        use EngineLifetime::*;
        use HelperState::*;
        assert_eq!(
            helpers_label(Enabled, WithApp),
            "Engines: running as background helpers (stop on quit)"
        );
        assert_eq!(
            helpers_label(Partial, WithApp),
            "Engines: helpers partly enabled (stop on quit)"
        );
        assert_eq!(
            helpers_label(Enabled, Always),
            "Engines: running as background helpers"
        );
        for state in [NotRegistered, RequiresApproval, NotFound, Unsupported] {
            assert!(
                !helpers_label(state, WithApp).contains("stop on quit"),
                "{state:?}"
            );
        }
    }

    #[test]
    fn menu_ids_round_trip_to_actions() {
        assert_eq!(parse_action("engine:stop-all"), Some(Action::StopAll));
        assert_eq!(parse_action("engine:helpers"), Some(Action::OpenLoginItems));
        assert_eq!(
            parse_action("engine:comfyui-free"),
            Some(Action::FreeComfyui)
        );
        assert_eq!(parse_action("engine:comfyui-label"), None);
        assert_eq!(
            parse_action("engine:unload:ukisai--Swift"),
            Some(Action::Unload("ukisai--Swift".into()))
        );
        assert_eq!(
            parse_action("engine:load:a/b:c"),
            Some(Action::Load("a/b:c".into()))
        );
        assert_eq!(parse_action("open"), None);
        assert_eq!(parse_action("engine:omlx-label"), None);
    }

    #[test]
    fn path_segments_are_percent_encoded() {
        assert_eq!(
            encode_segment("ukisai--Swift-1.5-27B-oQ4e-mtp"),
            "ukisai--Swift-1.5-27B-oQ4e-mtp"
        );
        assert_eq!(encode_segment("swift-1.5-27b:fast"), "swift-1.5-27b%3Afast");
        assert_eq!(encode_segment("a/b c"), "a%2Fb%20c");
    }

    #[test]
    fn durations_format() {
        assert_eq!(fmt_duration(45.0), "45s");
        assert_eq!(fmt_duration(252.4), "4m 12s");
        assert_eq!(fmt_duration(-3.0), "0s");
    }
}

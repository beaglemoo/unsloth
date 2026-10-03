//! Tray items for the attached engines (oMLX and DwarfStar/ds4), behind the
//! `attached-engines` feature.
//!
//! The tray talks to the engines directly (it must work with Studio's backend stopped): every
//! 5 s it polls oMLX `/v1/models/status` and the ds4 launcher `/admin/status`, with the URLs
//! taken from `~/.unsloth/engines/engines.toml` (defaults :8843 and :8001). Menu actions
//! duplicate the backend arbiter's rule: ds4 is stopped before an oMLX model is loaded, so the
//! two never hold weights at the same time on a 64 GB machine.

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
use crate::loopback_http;

const DEFAULT_OMLX_URL: &str = "http://127.0.0.1:8843";
const DEFAULT_DS4_URL: &str = "http://127.0.0.1:8001";
const POLL_INTERVAL: Duration = Duration::from_secs(5);
const STATUS_TIMEOUT: Duration = Duration::from_secs(2);
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(30);
/// The launcher's start timeout (120 s) plus its slack, as the backend client uses.
const DS4_ADMIN_TIMEOUT: Duration = Duration::from_secs(135);

const ID_DS4_TOGGLE: &str = "engine:ds4-toggle";
const ID_STOP_ALL: &str = "engine:stop-all";
const ID_HELPERS: &str = "engine:helpers";
const ID_UNLOAD_PREFIX: &str = "engine:unload:";
const ID_LOAD_PREFIX: &str = "engine:load:";

// ---------------------------------------------------------------------------------------------
// Configuration

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EngineUrls {
    pub(crate) omlx: String,
    pub(crate) ds4: String,
}

impl Default for EngineUrls {
    fn default() -> Self {
        Self {
            omlx: DEFAULT_OMLX_URL.to_string(),
            ds4: DEFAULT_DS4_URL.to_string(),
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
        ds4: format!(
            "http://{}:{}",
            connect_host(host_of(&table, "ds4")),
            port_of(&table, "ds4", "launcher_port", 8001)
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

fn load_urls() -> EngineUrls {
    config_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| parse_urls(&text))
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// Status parsing

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) struct OmlxModel {
    id: String,
    model_path: String,
    loaded: bool,
    is_loading: bool,
    engine_type: Option<String>,
    is_helper: bool,
    alias: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) struct OmlxStatus {
    reachable: bool,
    models: Vec<OmlxModel>,
    memory_bytes: u64,
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
                engine_type: str_field(row, "engine_type"),
                is_helper: row
                    .get("is_helper")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                alias: str_field(row, "model_alias"),
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
    })
}

#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct Ds4Status {
    reachable: bool,
    loaded: bool,
    starting: bool,
    in_flight: u64,
    idle_remaining_s: Option<f64>,
    live_tps: Option<f64>,
    last_tps: Option<f64>,
}

pub(crate) fn parse_ds4_status(body: &Value) -> Option<Ds4Status> {
    body.as_object()?;
    let loaded = body.get("loaded").and_then(Value::as_bool).unwrap_or(false);
    let pid_set = body.get("pid").is_some_and(|p| !p.is_null());
    let stats = body.get("stats");
    let tps = |which: &str| {
        stats
            .and_then(|s| s.get(which))
            .and_then(|s| s.get("gen_tps"))
            .and_then(Value::as_f64)
    };
    Some(Ds4Status {
        reachable: true,
        loaded,
        // The launcher has no "starting" flag: a spawned process that is not serving yet is.
        starting: pid_set && !loaded,
        in_flight: body.get("in_flight").and_then(Value::as_u64).unwrap_or(0),
        idle_remaining_s: body.get("idle_seconds_remaining").and_then(Value::as_f64),
        live_tps: tps("live"),
        last_tps: tps("last"),
    })
}

/// The rule the backend arbiter applies, duplicated here for tray-initiated loads: stop ds4
/// first whenever it is serving or still starting.
pub(crate) fn must_stop_ds4_before_load(ds4: &Ds4Status) -> bool {
    ds4.reachable && (ds4.loaded || ds4.starting)
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
fn all_loaded_dirs(status: &OmlxStatus) -> Vec<String> {
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
    ds4_label: String,
    ds4_toggle_text: &'static str,
    ds4_toggle_enabled: bool,
    stop_all_enabled: bool,
    helpers_label: &'static str,
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
        return "oMLX: not running".to_string();
    }
    if let Some(loading) = status.models.iter().find(|m| m.is_loading) {
        let name = loading.alias.as_deref().unwrap_or(&loading.id);
        return format!("oMLX: loading {name}");
    }
    let loaded = all_loaded_dirs(status).len();
    format!("oMLX: {loaded} loaded ({} GB)", gib(status.memory_bytes))
}

fn ds4_label(status: &Ds4Status) -> String {
    if !status.reachable {
        return "DwarfStar: not running".to_string();
    }
    if status.starting {
        return "DwarfStar: starting".to_string();
    }
    if !status.loaded {
        return "DwarfStar: stopped".to_string();
    }
    let mut parts = vec!["DwarfStar: loaded".to_string()];
    if status.in_flight > 0 {
        parts.push("generating".to_string());
    } else if let Some(idle) = status.idle_remaining_s {
        parts.push(format!("idle, stops in {}", fmt_duration(idle)));
    }
    if let Some(tps) = status.live_tps.or(status.last_tps) {
        parts.push(format!("{tps:.1} tok/s"));
    }
    parts.join(", ")
}

fn helpers_label(state: HelperState) -> &'static str {
    match state {
        HelperState::Enabled => "Engines: running as background helpers",
        HelperState::RequiresApproval => "Engines: approve the helpers in Login Items",
        HelperState::NotRegistered => "Engines: not running as helpers (enable in Settings)",
        HelperState::Partial => "Engines: helpers partly enabled",
        HelperState::NotFound => "Engines: helper agents not bundled",
        HelperState::Unsupported => "Engines: background helpers need macOS",
    }
}

pub(crate) fn build_view(omlx: &OmlxStatus, ds4: &Ds4Status, helpers: HelperState) -> View {
    let (unloads, loads) = if omlx.reachable {
        model_entries(omlx)
    } else {
        (Vec::new(), Vec::new())
    };
    let ds4_running = ds4.reachable && (ds4.loaded || ds4.starting);
    View {
        omlx_label: omlx_label(omlx),
        unloads,
        load_enabled: omlx.reachable,
        loads,
        ds4_label: ds4_label(ds4),
        ds4_toggle_text: if ds4_running {
            "Stop DwarfStar"
        } else {
            "Start DwarfStar"
        },
        ds4_toggle_enabled: ds4.reachable,
        stop_all_enabled: omlx.reachable || ds4.reachable,
        helpers_label: helpers_label(helpers),
    }
}

// ---------------------------------------------------------------------------------------------
// Actions

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Action {
    Ds4Toggle,
    StopAll,
    OpenLoginItems,
    Unload(String),
    Load(String),
}

pub(crate) fn parse_action(id: &str) -> Option<Action> {
    match id {
        ID_DS4_TOGGLE => Some(Action::Ds4Toggle),
        ID_STOP_ALL => Some(Action::StopAll),
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
fn encode_segment(segment: &str) -> String {
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
    get_json(&urls.omlx, "/v1/models/status")
        .await
        .and_then(|body| parse_omlx_status(&body))
        .unwrap_or_default()
}

async fn fetch_ds4(urls: &EngineUrls) -> Ds4Status {
    get_json(&urls.ds4, "/admin/status")
        .await
        .and_then(|body| parse_ds4_status(&body))
        .unwrap_or_default()
}

async fn stop_ds4(urls: &EngineUrls) -> Result<(), String> {
    post(&urls.ds4, "/admin/stop", DS4_ADMIN_TIMEOUT).await
}

async fn load_model(urls: &EngineUrls, dir: &str) -> Result<(), String> {
    // Stop ds4 first, always from a fresh read: the poll may be 5 s old.
    if must_stop_ds4_before_load(&fetch_ds4(urls).await) {
        stop_ds4(urls)
            .await
            .map_err(|e| format!("could not stop DwarfStar: {e}"))?;
    }
    let path = format!("/admin/api/models/{}/load", encode_segment(dir));
    post(&urls.omlx, &path, LOAD_TIMEOUT).await
}

async fn unload_model(urls: &EngineUrls, dir: &str) -> Result<(), String> {
    let path = format!("/v1/models/{}/unload", encode_segment(dir));
    post(&urls.omlx, &path, UNLOAD_TIMEOUT).await
}

async fn stop_all(urls: &EngineUrls) -> Result<(), String> {
    let mut errors = Vec::new();
    if fetch_ds4(urls).await.reachable {
        if let Err(error) = stop_ds4(urls).await {
            errors.push(format!("DwarfStar: {error}"));
        }
    }
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

async fn ds4_toggle(urls: &EngineUrls) -> Result<(), String> {
    if must_stop_ds4_before_load(&fetch_ds4(urls).await) {
        stop_ds4(urls).await
    } else {
        // The launcher unloads oMLX models itself before it starts ds4-server.
        post(&urls.ds4, "/admin/start", DS4_ADMIN_TIMEOUT).await
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
}

struct EngineTray {
    menu: Menu<Wry>,
    omlx_label: MenuItem<Wry>,
    load_menu: Submenu<Wry>,
    ds4_label: MenuItem<Wry>,
    ds4_toggle: MenuItem<Wry>,
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
        if previous
            .as_ref()
            .is_none_or(|p| p.ds4_label != view.ds4_label)
        {
            let _ = self.ds4_label.set_text(&view.ds4_label);
        }
        if previous
            .as_ref()
            .is_none_or(|p| p.ds4_toggle_text != view.ds4_toggle_text)
        {
            let _ = self.ds4_toggle.set_text(view.ds4_toggle_text);
        }
        if changed(|v| v.ds4_toggle_enabled) {
            let _ = self.ds4_toggle.set_enabled(view.ds4_toggle_enabled);
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
            let _ = self.helpers.set_text(view.helpers_label);
        }
        if dynamic.unload_keys != view.unloads || previous.is_none() {
            self.replace_unloads(app, &mut dynamic, &view.unloads);
        }
        if dynamic.load_keys.as_ref() != Some(&view.loads) {
            self.replace_loads(app, &mut dynamic, &view.loads);
        }
        dynamic.last = Some(view.clone());
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
    let ds4_label = MenuItemBuilder::with_id("engine:ds4-label", "DwarfStar: checking")
        .enabled(false)
        .build(app)?;
    let ds4_toggle = MenuItemBuilder::with_id(ID_DS4_TOGGLE, "Start DwarfStar")
        .enabled(false)
        .build(app)?;
    let stop_all = MenuItemBuilder::with_id(ID_STOP_ALL, "Stop all engines")
        .enabled(false)
        .build(app)?;
    let helpers = MenuItemBuilder::with_id(ID_HELPERS, helpers_label(HelperState::NotRegistered))
        .build(app)?;
    let sep_top = PredefinedMenuItem::separator(app)?;
    let sep_mid = PredefinedMenuItem::separator(app)?;
    let sep_bottom = PredefinedMenuItem::separator(app)?;
    let sep_end = PredefinedMenuItem::separator(app)?;
    let items: [&dyn IsMenuItem<Wry>; 10] = [
        &sep_top,
        &omlx_label,
        &load_menu,
        &sep_mid,
        &ds4_label,
        &ds4_toggle,
        &sep_bottom,
        &stop_all,
        &helpers,
        &sep_end,
    ];
    // The caller's menu is [open, toggle, quit]: these go between toggle and Quit.
    menu.insert_items(&items, 2)?;
    app.manage(EngineTray {
        menu: menu.clone(),
        omlx_label,
        load_menu,
        ds4_label,
        ds4_toggle,
        stop_all,
        helpers,
        dynamic: Mutex::new(Dynamic::default()),
        busy: AtomicBool::new(false),
    });

    let handle = app.handle().clone();
    tauri::async_runtime::spawn(async move {
        loop {
            let urls = load_urls();
            let (omlx, ds4) = tokio::join!(fetch_omlx(&urls), fetch_ds4(&urls));
            let view = build_view(&omlx, &ds4, engine_helpers::current_state());
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
            Action::Ds4Toggle => ds4_toggle(&urls).await,
            Action::StopAll => stop_all(&urls).await,
            Action::Unload(dir) => unload_model(&urls, dir).await,
            Action::Load(dir) => load_model(&urls, dir).await,
            Action::OpenLoginItems => Ok(()),
        };
        if let Err(error) = result {
            warn!("engine tray action {action:?} failed: {error}");
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

    fn ds4(body: Value) -> Ds4Status {
        parse_ds4_status(&body).unwrap()
    }

    #[test]
    fn urls_default_when_config_missing_or_invalid() {
        assert_eq!(parse_urls(""), EngineUrls::default());
        assert_eq!(parse_urls("not = [valid"), EngineUrls::default());
        assert_eq!(EngineUrls::default().omlx, "http://127.0.0.1:8843");
        assert_eq!(EngineUrls::default().ds4, "http://127.0.0.1:8001");
    }

    #[test]
    fn urls_follow_ports_and_map_wildcard_hosts_to_loopback() {
        let urls = parse_urls(
            "[omlx]\nport = 18843\n[ds4]\nhost = \"0.0.0.0\"\nlauncher_port = 18001\nserver_port = 18000\n",
        );
        assert_eq!(urls.omlx, "http://127.0.0.1:18843");
        assert_eq!(urls.ds4, "http://127.0.0.1:18001");
        let urls =
            parse_urls("[omlx]\nhost = \"192.168.3.78\"\nport = 9000\n[ds4]\nlauncher_port = 0\n");
        assert_eq!(urls.omlx, "http://192.168.3.78:9000");
        assert_eq!(urls.ds4, DEFAULT_DS4_URL);
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
    fn ds4_state_and_labels() {
        let stopped = ds4(json!({"loaded": false, "pid": null, "in_flight": 0,
            "idle_seconds_remaining": null, "stats": {"live": null, "last": null}}));
        assert!(!stopped.starting && !stopped.loaded);
        assert_eq!(ds4_label(&stopped), "DwarfStar: stopped");
        assert!(!must_stop_ds4_before_load(&stopped));

        let starting = ds4(json!({"loaded": false, "pid": 4242, "in_flight": 0, "stats": {}}));
        assert!(starting.starting);
        assert_eq!(ds4_label(&starting), "DwarfStar: starting");
        assert!(must_stop_ds4_before_load(&starting));

        let loaded = ds4(json!({"loaded": true, "pid": 4242, "in_flight": 0,
            "idle_seconds_remaining": 252.4,
            "stats": {"live": null, "last": {"gen_tps": 31.24, "ttft_ms": 800.0}}}));
        assert_eq!(
            ds4_label(&loaded),
            "DwarfStar: loaded, idle, stops in 4m 12s, 31.2 tok/s"
        );
        assert!(must_stop_ds4_before_load(&loaded));

        let busy = ds4(json!({"loaded": true, "pid": 1, "in_flight": 2,
            "stats": {"live": {"gen_tps": 40.0}, "last": {"gen_tps": 31.0}}}));
        assert_eq!(
            ds4_label(&busy),
            "DwarfStar: loaded, generating, 40.0 tok/s"
        );

        assert!(!must_stop_ds4_before_load(&Ds4Status::default()));
        assert_eq!(ds4_label(&Ds4Status::default()), "DwarfStar: not running");
    }

    #[test]
    fn view_enables_controls_from_reachability() {
        let down = build_view(
            &OmlxStatus::default(),
            &Ds4Status::default(),
            HelperState::NotRegistered,
        );
        assert!(!down.load_enabled && !down.ds4_toggle_enabled && !down.stop_all_enabled);
        assert_eq!(down.ds4_toggle_text, "Start DwarfStar");
        assert_eq!(
            down.helpers_label,
            "Engines: not running as helpers (enable in Settings)"
        );

        let up = build_view(
            &omlx(OMLX_BODY),
            &ds4(json!({"loaded": true, "pid": 9, "in_flight": 0})),
            HelperState::Enabled,
        );
        assert!(up.load_enabled && up.ds4_toggle_enabled && up.stop_all_enabled);
        assert_eq!(up.ds4_toggle_text, "Stop DwarfStar");
        assert_eq!(up.helpers_label, "Engines: running as background helpers");
        assert_eq!(up.loads.len(), 1);
    }

    #[test]
    fn menu_ids_round_trip_to_actions() {
        assert_eq!(parse_action("engine:ds4-toggle"), Some(Action::Ds4Toggle));
        assert_eq!(parse_action("engine:stop-all"), Some(Action::StopAll));
        assert_eq!(parse_action("engine:helpers"), Some(Action::OpenLoginItems));
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

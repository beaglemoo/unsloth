//! Settings > About > Updates for the beaglemoo fork. Everything here is inert unless the build
//! has the `attached-engines` feature: the commands answer "disabled" and no request is made.
//!
//! The fork has one update source: its own repo, github.com/beaglemoo/unsloth. A build bakes the
//! commit it was made from (`UNSLOTH_FORK_COMMIT`, set by build-fork-mac.sh) and the branch it
//! follows (`UNSLOTH_FORK_BRANCH`). The check asks GitHub, unauthenticated, how far that branch
//! is ahead of the build:
//!
//!   GET https://api.github.com/repos/beaglemoo/unsloth/compare/{commit}...{branch}   (10 s timeout)
//!
//! It never contacts upstream Unsloth, PyPI or the Tauri updater endpoint. The update itself is
//! not run in the app: the build needs cargo and npm from the user's shell, so the Update button
//! opens Terminal on studio/scripts/update-fork.sh.
//!
//! The result is cached in `<engines home>/fork-update.json` (`UNSLOTH_ENGINES_HOME`, else
//! `~/.unsloth/engines`). The launch check runs at most once a day; the button always checks.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub(crate) const ENABLED: bool = cfg!(feature = "attached-engines");

const GITHUB_API: &str = "https://api.github.com";
const FORK_SLUG: &str = "beaglemoo/unsloth";
const CHECK_TIMEOUT: Duration = Duration::from_secs(10);
/// The launch check and the About tab check again only after this long.
const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;
/// A failed check (offline, rate limited) is retried sooner.
const ERROR_RETRY_SECS: u64 = 60 * 60;
const MAX_COMMITS: usize = 30;
const CACHE_FILE: &str = "fork-update.json";
const UPDATE_SCRIPT_REL: &str = "studio/scripts/update-fork.sh";

const BUILD_COMMIT: Option<&str> = option_env!("UNSLOTH_FORK_COMMIT");
const BUILD_BRANCH: Option<&str> = option_env!("UNSLOTH_FORK_BRANCH");

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ForkCommit {
    pub(crate) sha: String,
    /// The first line of the commit message.
    pub(crate) message: String,
}

/// What the compare endpoint said: `ahead_by` commits on the branch that the build lacks.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ForkCompare {
    pub(crate) ahead_by: u64,
    pub(crate) commits: Vec<ForkCommit>,
    pub(crate) html_url: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
struct Cached {
    checked_at: u64,
    build_sha: String,
    branch: String,
    compare: Option<ForkCompare>,
    error: Option<String>,
}

/// What the About tab shows. `ahead_by` is None while the answer is unknown (never checked, or
/// the last check failed).
#[derive(Serialize, Debug, PartialEq, Eq, Default)]
pub(crate) struct ForkUpdateInfo {
    pub(crate) enabled: bool,
    pub(crate) build_sha: Option<String>,
    pub(crate) short_sha: Option<String>,
    pub(crate) branch: Option<String>,
    pub(crate) checked_at: Option<u64>,
    pub(crate) ahead_by: Option<u64>,
    pub(crate) commits: Vec<ForkCommit>,
    pub(crate) html_url: Option<String>,
    pub(crate) error: Option<String>,
    /// update-fork.sh exists in the baked checkout, so the Update button can open it.
    pub(crate) can_update: bool,
}

// ---------------------------------------------------------------------------------------------
// Pure helpers

fn is_hex_sha(value: &str) -> bool {
    (7..=40).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_safe_branch(value: &str) -> bool {
    !value.is_empty()
        && !value.contains("..")
        && !value.starts_with('/')
        && !value.ends_with('/')
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'-'))
}

/// `{base}/repos/beaglemoo/unsloth/compare/{commit}...{branch}`. The commit must be a hex sha and
/// the branch a plain ref name, so nothing the build baked can reshape the request.
pub(crate) fn compare_url(base: &str, commit: &str, branch: &str) -> Result<String, String> {
    if !is_hex_sha(commit) {
        return Err(format!("the baked fork commit {commit:?} is not a git sha"));
    }
    if !is_safe_branch(branch) {
        return Err(format!("the baked fork branch {branch:?} is not a plain branch name"));
    }
    Ok(format!(
        "{}/repos/{FORK_SLUG}/compare/{commit}...{branch}",
        base.trim_end_matches('/')
    ))
}

fn first_line(message: &str) -> String {
    message.lines().next().unwrap_or("").trim().to_string()
}

/// The fields of GitHub's compare response that the UI uses.
pub(crate) fn parse_compare(body: &str) -> Result<ForkCompare, String> {
    #[derive(Deserialize)]
    struct Raw {
        ahead_by: u64,
        #[serde(default)]
        html_url: String,
        #[serde(default)]
        commits: Vec<RawCommit>,
    }
    #[derive(Deserialize)]
    struct RawCommit {
        sha: String,
        #[serde(default)]
        commit: RawInner,
    }
    #[derive(Deserialize, Default)]
    struct RawInner {
        #[serde(default)]
        message: String,
    }
    let raw: Raw = serde_json::from_str(body)
        .map_err(|error| format!("GitHub sent a compare response this app cannot read: {error}"))?;
    Ok(ForkCompare {
        ahead_by: raw.ahead_by,
        html_url: raw.html_url,
        commits: raw
            .commits
            .into_iter()
            .take(MAX_COMMITS)
            .map(|c| ForkCommit {
                sha: c.sha,
                message: first_line(&c.commit.message),
            })
            .collect(),
    })
}

fn short(sha: &str) -> String {
    sha.chars().take(9).collect()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A cached answer is reused for a day (an hour after a failure), and only for the same build and
/// branch: a new build always asks again.
fn cache_is_fresh(cached: &Cached, now: u64, commit: &str, branch: &str) -> bool {
    if cached.build_sha != commit || cached.branch != branch {
        return false;
    }
    let age = now.saturating_sub(cached.checked_at);
    let limit = if cached.error.is_some() {
        ERROR_RETRY_SECS
    } else {
        CHECK_INTERVAL_SECS
    };
    age < limit
}

fn cache_path() -> Option<PathBuf> {
    let home = match std::env::var_os("UNSLOTH_ENGINES_HOME") {
        Some(home) => PathBuf::from(home),
        None => dirs::home_dir()?.join(".unsloth").join("engines"),
    };
    Some(home.join(CACHE_FILE))
}

fn read_cache(path: &Path) -> Option<Cached> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn write_cache(path: &Path, cached: &Cached) {
    let Ok(text) = serde_json::to_string_pretty(cached) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let temporary = path.with_extension("json.tmp");
    if std::fs::write(&temporary, text).is_ok() && std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
}

fn update_script_path(checkout: Option<&str>) -> Option<PathBuf> {
    let path = Path::new(checkout?.trim()).join(UPDATE_SCRIPT_REL);
    path.is_file().then_some(path)
}

fn info_from(
    enabled: bool,
    commit: Option<&str>,
    branch: Option<&str>,
    cached: Option<&Cached>,
    can_update: bool,
) -> ForkUpdateInfo {
    let mut info = ForkUpdateInfo {
        enabled,
        build_sha: commit.map(str::to_string),
        short_sha: commit.map(short),
        branch: branch.map(str::to_string),
        can_update,
        ..Default::default()
    };
    if !enabled {
        return ForkUpdateInfo {
            enabled: false,
            ..Default::default()
        };
    }
    if let (Some(cached), Some(commit), Some(branch)) = (cached, commit, branch) {
        // A result for another build or branch says nothing about this one.
        if cached.build_sha == commit && cached.branch == branch {
            info.checked_at = Some(cached.checked_at);
            info.error = cached.error.clone();
            if let Some(compare) = &cached.compare {
                info.ahead_by = Some(compare.ahead_by);
                info.commits = compare.commits.clone();
                info.html_url = Some(compare.html_url.clone());
            }
        }
    }
    info
}

// ---------------------------------------------------------------------------------------------
// The one network call

/// GET the compare endpoint under `base` (`https://api.github.com`; a local server in tests).
pub(crate) async fn fetch_compare(
    base: &str,
    commit: &str,
    branch: &str,
    timeout: Duration,
) -> Result<ForkCompare, String> {
    let url = compare_url(base, commit, branch)?;
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .user_agent("unsloth-fork-update-check")
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|error| {
            if error.is_timeout() {
                format!("GitHub did not answer within {} s", timeout.as_secs().max(1))
            } else {
                format!("Could not reach GitHub: {error}")
            }
        })?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(
            "This build's commit is not on github.com/beaglemoo/unsloth (an unpushed local build?)"
                .to_string(),
        );
    }
    if status == reqwest::StatusCode::FORBIDDEN || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err("GitHub is rate limiting this IP address; try again later".to_string());
    }
    if !status.is_success() {
        return Err(format!("GitHub answered HTTP {}", status.as_u16()));
    }
    let body = response.text().await.map_err(|error| error.to_string())?;
    parse_compare(&body)
}

/// Everything `run_check` needs, so a test can point it at a local server and a scratch cache.
pub(crate) struct CheckPlan<'a> {
    pub(crate) enabled: bool,
    pub(crate) base: &'a str,
    pub(crate) commit: Option<&'a str>,
    pub(crate) branch: Option<&'a str>,
    pub(crate) cache: Option<&'a Path>,
    pub(crate) checkout: Option<&'a str>,
    pub(crate) now: u64,
    pub(crate) timeout: Duration,
}

/// Answer from the cache when it is fresh (and `force` is off), else ask GitHub and remember the
/// answer. Never a request when the feature is off or the build baked no commit.
pub(crate) async fn run_check(plan: &CheckPlan<'_>, force: bool) -> ForkUpdateInfo {
    let can_update = update_script_path(plan.checkout).is_some();
    let cached = plan.cache.and_then(read_cache);
    if !plan.enabled {
        return info_from(false, None, None, None, false);
    }
    let (Some(commit), Some(branch)) = (plan.commit, plan.branch) else {
        let mut info = info_from(true, plan.commit, plan.branch, None, can_update);
        info.error = Some("This build did not record its fork commit; rebuild with build-fork-mac.sh".to_string());
        return info;
    };
    if !force {
        if let Some(cached) = &cached {
            if cache_is_fresh(cached, plan.now, commit, branch) {
                return info_from(true, Some(commit), Some(branch), Some(cached), can_update);
            }
        }
    }
    let result = fetch_compare(plan.base, commit, branch, plan.timeout).await;
    let entry = match result {
        Ok(compare) => Cached {
            checked_at: plan.now,
            build_sha: commit.to_string(),
            branch: branch.to_string(),
            compare: Some(compare),
            error: None,
        },
        Err(error) => {
            log::warn!("fork update check failed: {error}");
            Cached {
                checked_at: plan.now,
                build_sha: commit.to_string(),
                branch: branch.to_string(),
                // Keep the last good answer visible beside the error.
                compare: cached
                    .as_ref()
                    .filter(|c| c.build_sha == commit && c.branch == branch)
                    .and_then(|c| c.compare.clone()),
                error: Some(error),
            }
        }
    };
    if let Some(path) = plan.cache {
        write_cache(path, &entry);
    }
    info_from(true, Some(commit), Some(branch), Some(&entry), can_update)
}

fn production_plan(cache: Option<&Path>) -> CheckPlan<'_> {
    CheckPlan {
        enabled: ENABLED,
        base: GITHUB_API,
        commit: BUILD_COMMIT.filter(|c| !c.is_empty()),
        branch: BUILD_BRANCH.filter(|b| !b.is_empty()),
        cache,
        checkout: crate::update::FORK_REPO,
        now: now_secs(),
        timeout: CHECK_TIMEOUT,
    }
}

// ---------------------------------------------------------------------------------------------
// Terminal

/// `open -a Terminal <script>`: LaunchServices hands the script to Terminal, which runs it in a new
/// window. Unlike an AppleScript `do script` it needs no Automation permission, which a hardened
/// runtime app without the apple-events entitlement could not get.
pub(crate) fn terminal_open_args(script: &str) -> Vec<String> {
    vec!["-a".to_string(), "Terminal".to_string(), script.to_string()]
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(_path: &Path) -> bool {
    true
}

// ---------------------------------------------------------------------------------------------
// Commands

/// The last known state, from the cache: no network.
#[tauri::command]
pub(crate) fn fork_update_info() -> ForkUpdateInfo {
    let plan = production_plan(None);
    let cache = cache_path();
    let cached = cache.as_deref().and_then(read_cache);
    info_from(
        plan.enabled,
        plan.commit,
        plan.branch,
        cached.as_ref(),
        update_script_path(plan.checkout).is_some(),
    )
}

/// Check now (`force`), or only when the cached answer is older than a day.
#[tauri::command]
pub(crate) async fn fork_update_check(force: bool) -> ForkUpdateInfo {
    let cache = cache_path();
    let plan = production_plan(cache.as_deref());
    run_check(&plan, force).await
}

/// Open Terminal running update-fork.sh from the baked checkout.
#[tauri::command]
pub(crate) fn fork_update_open_terminal() -> Result<(), String> {
    if !ENABLED {
        return Err("Fork updates are not enabled in this build.".to_string());
    }
    let script = update_script_path(crate::update::FORK_REPO).ok_or_else(|| {
        "update-fork.sh was not found in this build's source checkout; run it from a terminal."
            .to_string()
    })?;
    if !is_executable(&script) {
        return Err(format!(
            "{} is not executable (chmod +x it); run it from a terminal.",
            script.display()
        ));
    }
    open_terminal(&script.to_string_lossy())
}

#[cfg(target_os = "macos")]
fn open_terminal(script: &str) -> Result<(), String> {
    let status = std::process::Command::new("/usr/bin/open")
        .args(terminal_open_args(script))
        .stdin(std::process::Stdio::null())
        .status()
        .map_err(|error| format!("could not run open: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("macOS could not open Terminal on the update script.".to_string())
    }
}

#[cfg(not(target_os = "macos"))]
fn open_terminal(_script: &str) -> Result<(), String> {
    Err("Opening Terminal is only supported on macOS.".to_string())
}

/// Once a day at launch, in the background; the About tab reads the cache.
pub(crate) fn check_at_launch() {
    if !ENABLED {
        return;
    }
    tauri::async_runtime::spawn(async {
        let cache = cache_path();
        let plan = production_plan(cache.as_deref());
        let info = run_check(&plan, false).await;
        match (info.ahead_by, &info.error) {
            (Some(0), _) => log::info!("fork update check: up to date"),
            (Some(n), _) => log::info!("fork update check: {n} update(s) available"),
            (None, Some(error)) => log::info!("fork update check: {error}"),
            (None, None) => {}
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    const SHA: &str = "5426c052d0bfab4d8ada7c6dc55bebb92ffa219e";
    const BRANCH: &str = "feat/omlx-ds4-engines";

    fn compare_json(ahead: u64) -> String {
        let commits: Vec<String> = (0..ahead)
            .map(|i| {
                format!(
                    r#"{{"sha":"{i:040x}","node_id":"x","commit":{{"message":"feat: change {i}\n\nlong body"}}}}"#
                )
            })
            .collect();
        format!(
            r#"{{"url":"u","html_url":"https://github.com/beaglemoo/unsloth/compare/{SHA}...{BRANCH}","status":"ahead","ahead_by":{ahead},"behind_by":0,"total_commits":{ahead},"commits":[{}]}}"#,
            commits.join(",")
        )
    }

    /// A one-shot-per-connection HTTP server on 127.0.0.1 that answers every request with
    /// `status` and `body`, and records the request head. Stands in for api.github.com.
    struct MockGithub {
        base: String,
        requests: Arc<Mutex<Vec<String>>>,
    }

    fn mock_github(status: u16, body: String) -> MockGithub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut head = Vec::new();
                let mut buf = [0u8; 1024];
                while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => head.extend_from_slice(&buf[..n]),
                    }
                }
                seen.lock().unwrap().push(String::from_utf8_lossy(&head).to_string());
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        MockGithub { base, requests }
    }

    /// Accepts connections and never answers.
    fn silent_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for stream in listener.incoming().flatten() {
                held.push(stream);
            }
        });
        base
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Runtime::new().unwrap().block_on(future)
    }

    fn plan<'a>(base: &'a str, cache: Option<&'a Path>, now: u64) -> CheckPlan<'a> {
        CheckPlan {
            enabled: true,
            base,
            commit: Some(SHA),
            branch: Some(BRANCH),
            cache,
            checkout: None,
            now,
            timeout: Duration::from_secs(5),
        }
    }

    #[test]
    fn the_compare_url_is_the_forks_github_endpoint() {
        assert_eq!(
            compare_url("https://api.github.com", SHA, BRANCH).unwrap(),
            format!("https://api.github.com/repos/beaglemoo/unsloth/compare/{SHA}...{BRANCH}")
        );
        assert!(compare_url("https://api.github.com/", SHA, "main").unwrap().contains("/compare/"));
        assert!(!GITHUB_API.contains("unslothai"));
        assert!(FORK_SLUG.starts_with("beaglemoo/"));
    }

    #[test]
    fn a_baked_value_cannot_reshape_the_request() {
        for bad in ["", "xyz", "abc", "5426c05 ; rm", "../x", &"a".repeat(41)] {
            assert!(compare_url(GITHUB_API, bad, BRANCH).is_err(), "{bad}");
        }
        for bad in ["", "a b", "a..b", "/a", "a/", "a?b", "a#b", "a%2fb", "a\nb"] {
            assert!(compare_url(GITHUB_API, SHA, bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_compare_response_is_reduced_to_what_the_ui_shows() {
        let parsed = parse_compare(&compare_json(3)).unwrap();
        assert_eq!(parsed.ahead_by, 3);
        assert_eq!(parsed.commits.len(), 3);
        assert_eq!(parsed.commits[1].message, "feat: change 1", "only the first line");
        assert_eq!(parsed.commits[1].sha, format!("{:040x}", 1));
        assert!(parsed.html_url.starts_with("https://github.com/beaglemoo/unsloth/compare/"));
    }

    #[test]
    fn a_long_commit_list_is_capped() {
        let parsed = parse_compare(&compare_json(45)).unwrap();
        assert_eq!(parsed.ahead_by, 45);
        assert_eq!(parsed.commits.len(), MAX_COMMITS);
    }

    #[test]
    fn an_unreadable_response_is_an_error_not_a_panic() {
        assert!(parse_compare("not json").is_err());
        assert!(parse_compare("{}").is_err());
        assert!(parse_compare(r#"{"ahead_by":"3"}"#).is_err());
        assert_eq!(parse_compare(r#"{"ahead_by":0}"#).unwrap().commits, vec![]);
    }

    #[test]
    fn fetch_sends_the_documented_request_and_parses_the_answer() {
        let server = mock_github(200, compare_json(2));
        let result = block_on(fetch_compare(&server.base, SHA, BRANCH, Duration::from_secs(5))).unwrap();
        assert_eq!(result.ahead_by, 2);
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let head = requests[0].to_lowercase();
        assert!(
            requests[0].starts_with(&format!("GET /repos/beaglemoo/unsloth/compare/{SHA}...{BRANCH} HTTP/1.1")),
            "{}",
            requests[0]
        );
        assert!(head.contains("user-agent: unsloth-fork-update-check"), "GitHub rejects a request without one");
        assert!(head.contains("accept: application/vnd.github+json"));
        assert!(!head.contains("authorization"), "the check is unauthenticated");
    }

    #[test]
    fn http_failures_become_readable_errors() {
        let cases = [
            (404, "not on github.com/beaglemoo/unsloth"),
            (403, "rate limiting"),
            (429, "rate limiting"),
            (500, "HTTP 500"),
        ];
        for (status, expected) in cases {
            let server = mock_github(status, "{}".to_string());
            let error = block_on(fetch_compare(&server.base, SHA, BRANCH, Duration::from_secs(5))).unwrap_err();
            assert!(error.contains(expected), "{status}: {error}");
        }
    }

    #[test]
    fn an_unreachable_server_is_an_error() {
        let free = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", free.local_addr().unwrap());
        drop(free);
        let error = block_on(fetch_compare(&base, SHA, BRANCH, Duration::from_secs(5))).unwrap_err();
        assert!(error.contains("Could not reach GitHub"), "{error}");
    }

    #[test]
    fn a_server_that_never_answers_times_out() {
        let base = silent_server();
        let started = std::time::Instant::now();
        let error = block_on(fetch_compare(&base, SHA, BRANCH, Duration::from_millis(300))).unwrap_err();
        assert!(error.contains("did not answer"), "{error}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn the_production_timeout_is_ten_seconds() {
        assert_eq!(CHECK_TIMEOUT, Duration::from_secs(10));
        assert_eq!(CHECK_INTERVAL_SECS, 86_400);
    }

    #[test]
    fn a_check_is_cached_for_a_day_and_force_bypasses_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("fork-update.json");
        let server = mock_github(200, compare_json(2));
        let now = 1_000_000;

        let first = block_on(run_check(&plan(&server.base, Some(&cache), now), false));
        assert_eq!(first.ahead_by, Some(2));
        assert_eq!(first.commits.len(), 2);
        assert_eq!(first.short_sha.as_deref(), Some("5426c052d"));
        assert_eq!(server.requests.lock().unwrap().len(), 1);

        // within a day: from the cache, no request
        let again = block_on(run_check(&plan(&server.base, Some(&cache), now + 3_600), false));
        assert_eq!(again.ahead_by, Some(2));
        assert_eq!(again.checked_at, Some(now));
        assert_eq!(server.requests.lock().unwrap().len(), 1);

        // on demand: asks again
        block_on(run_check(&plan(&server.base, Some(&cache), now + 3_700), true));
        assert_eq!(server.requests.lock().unwrap().len(), 2);

        // after a day: asks again
        let later = block_on(run_check(&plan(&server.base, Some(&cache), now + 3_700 + CHECK_INTERVAL_SECS), false));
        assert_eq!(server.requests.lock().unwrap().len(), 3);
        assert_eq!(later.checked_at, Some(now + 3_700 + CHECK_INTERVAL_SECS));
    }

    #[test]
    fn a_new_build_never_reuses_another_builds_answer() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("fork-update.json");
        let server = mock_github(200, compare_json(1));
        block_on(run_check(&plan(&server.base, Some(&cache), 10), false));
        let mut other = plan(&server.base, Some(&cache), 20);
        other.commit = Some("0123456789abcdef0123456789abcdef01234567");
        let info = block_on(run_check(&other, false));
        assert_eq!(server.requests.lock().unwrap().len(), 2);
        assert_eq!(info.build_sha.as_deref(), Some("0123456789abcdef0123456789abcdef01234567"));
    }

    #[test]
    fn an_up_to_date_build_reports_zero() {
        let server = mock_github(200, compare_json(0));
        let info = block_on(run_check(&plan(&server.base, None, 5), true));
        assert_eq!(info.ahead_by, Some(0));
        assert!(info.commits.is_empty());
        assert!(info.error.is_none());
    }

    #[test]
    fn a_failed_check_is_remembered_but_retried_within_the_hour() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("fork-update.json");
        let failing = mock_github(500, "{}".to_string());
        let info = block_on(run_check(&plan(&failing.base, Some(&cache), 100), false));
        assert_eq!(info.ahead_by, None);
        assert!(info.error.as_deref().unwrap().contains("HTTP 500"));
        // inside the retry window the failure is served from the cache
        block_on(run_check(&plan(&failing.base, Some(&cache), 200), false));
        assert_eq!(failing.requests.lock().unwrap().len(), 1);
        // after it, a good answer replaces the failure
        let good = mock_github(200, compare_json(4));
        let info = block_on(run_check(&plan(&good.base, Some(&cache), 100 + ERROR_RETRY_SECS), false));
        assert_eq!(info.ahead_by, Some(4));
        assert!(info.error.is_none());
    }

    #[test]
    fn a_failed_check_keeps_the_last_good_answer_visible() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("fork-update.json");
        let good = mock_github(200, compare_json(2));
        block_on(run_check(&plan(&good.base, Some(&cache), 100), false));
        let failing = mock_github(500, "{}".to_string());
        let info = block_on(run_check(&plan(&failing.base, Some(&cache), 200), true));
        assert_eq!(info.ahead_by, Some(2));
        assert!(info.error.is_some());
    }

    #[test]
    fn a_disabled_build_makes_no_request_and_reports_disabled() {
        let server = mock_github(200, compare_json(2));
        let mut disabled = plan(&server.base, None, 5);
        disabled.enabled = false;
        let info = block_on(run_check(&disabled, true));
        assert!(!info.enabled);
        assert!(info.build_sha.is_none());
        assert!(server.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn a_build_without_a_baked_commit_makes_no_request() {
        let server = mock_github(200, compare_json(2));
        let mut unbaked = plan(&server.base, None, 5);
        unbaked.commit = None;
        let info = block_on(run_check(&unbaked, true));
        assert!(info.enabled);
        assert_eq!(info.ahead_by, None);
        assert!(info.error.as_deref().unwrap().contains("did not record"));
        assert!(server.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn the_feature_gate_decides_whether_the_commands_do_anything() {
        assert_eq!(ENABLED, cfg!(feature = "attached-engines"));
        assert_eq!(fork_update_info().enabled, cfg!(feature = "attached-engines"));
        #[cfg(not(feature = "attached-engines"))]
        {
            assert!(fork_update_open_terminal().is_err());
            assert!(block_on(fork_update_check(true)).build_sha.is_none());
        }
    }

    #[test]
    fn terminal_is_opened_on_the_script_without_a_shell_or_automation() {
        // One argv, no shell and no AppleScript: a hostile path stays one argument.
        assert_eq!(
            terminal_open_args("/Users/a \"b\"/it's/update-fork.sh"),
            vec!["-a", "Terminal", "/Users/a \"b\"/it's/update-fork.sh"]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_script_without_the_executable_bit_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("update-fork.sh");
        std::fs::write(&script, "#!/bin/bash\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(!is_executable(&script));
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(is_executable(&script));
    }

    #[test]
    fn the_update_script_is_found_only_in_the_baked_checkout() {
        let dir = tempfile::tempdir().unwrap();
        assert!(update_script_path(None).is_none());
        assert!(update_script_path(Some(dir.path().to_str().unwrap())).is_none());
        let scripts = dir.path().join("studio/scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(scripts.join("update-fork.sh"), "#!/bin/bash\n").unwrap();
        assert_eq!(
            update_script_path(Some(dir.path().to_str().unwrap())),
            Some(scripts.join("update-fork.sh"))
        );
    }

    #[test]
    fn info_ignores_a_cached_answer_for_another_build() {
        let cached = Cached {
            checked_at: 7,
            build_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            branch: BRANCH.into(),
            compare: Some(ForkCompare { ahead_by: 9, commits: vec![], html_url: "u".into() }),
            error: None,
        };
        let info = info_from(true, Some(SHA), Some(BRANCH), Some(&cached), true);
        assert_eq!(info.ahead_by, None);
        assert_eq!(info.checked_at, None);
        assert!(info.can_update);
    }

    #[test]
    fn a_corrupt_cache_file_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("fork-update.json");
        std::fs::write(&cache, "{{{").unwrap();
        assert!(read_cache(&cache).is_none());
        let server = mock_github(200, compare_json(1));
        let info = block_on(run_check(&plan(&server.base, Some(&cache), 9), false));
        assert_eq!(info.ahead_by, Some(1));
        assert!(read_cache(&cache).is_some(), "rewritten");
    }
}

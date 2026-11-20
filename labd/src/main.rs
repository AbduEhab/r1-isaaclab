//! labd â€” an in-memory training queue for the R1 Isaac Lab tasks.
//!
//! One binary, one process: an HTTP server for the team on the LAN, and a worker that runs one
//! training job at a time by invoking the repository's own `scripts/train.py` with Isaac Lab's
//! Python. Jobs live in memory; restarting empties the queue.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxPath, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

const INDEX: &str = include_str!("index.html");
const LOG_CAP: usize = 4000;

// ------------------------------------------------------------------------------------------- config

#[derive(Parser, Clone)]
#[command(name = "labd", about = "In-memory training queue for the R1 Isaac Lab tasks")]
struct Cfg {
    /// Path to the r1-isaaclab checkout (contains scripts/train.py).
    #[arg(long)]
    repo: PathBuf,
    /// Path to the IsaacLab checkout (contains python.bat / python.sh).
    #[arg(long)]
    isaaclab: PathBuf,
    #[arg(long, default_value = "8080")]
    port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    bind: String,
    /// Where finished runs are copied. Default: <repo>/labd-artifacts
    #[arg(long)]
    artifacts: Option<PathBuf>,
    /// Comma-separated allowlist of task ids.
    #[arg(long, default_value = "Isaac-Velocity-Flat-R1,Isaac-Velocity-Rough-R1")]
    tasks: String,
}

impl Cfg {
    fn task_list(&self) -> Vec<String> {
        self.tasks
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// The interpreter to run the task scripts with, and whether it is the `isaaclab` wrapper
    /// (which needs `-p` before the script).
    ///
    /// Accepts both layouts: an `IsaacLab` checkout with `_isaac_sim` linked (the documented binary
    /// install), and an `isaaclab.bat`/`isaaclab.sh` wrapper at the root.
    fn interpreter(&self) -> (PathBuf, bool) {
        let name = if cfg!(windows) { "python.bat" } else { "python.sh" };
        let direct = self.isaaclab.join(name);
        if direct.is_file() {
            return (direct, false);
        }
        let linked = self.isaaclab.join("_isaac_sim").join(name);
        if linked.is_file() {
            return (linked, false);
        }
        let wrapper = self.isaaclab.join(if cfg!(windows) { "isaaclab.bat" } else { "isaaclab.sh" });
        (wrapper, true)
    }

    fn artifacts_root(&self) -> PathBuf {
        self.artifacts.clone().unwrap_or_else(|| self.repo.join("labd-artifacts"))
    }
}

// -------------------------------------------------------------------------------------------- jobs

#[derive(Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Status {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Serialize, Default)]
struct Progress {
    iteration: u32,
    total: u32,
    reward: Option<f64>,
    ep_len: Option<f64>,
    eta_s: Option<u64>,
    iter_s: Option<f64>,
}

#[derive(Clone)]
struct Job {
    id: u32,
    created: String,
    created_unix: u64,
    requester: String,
    task: String,
    num_envs: u32,
    iterations: u32,
    seed: Option<u32>,
    status: Status,
    cancelled: bool,
    progress: Progress,
    iter_times: VecDeque<f64>,
    log: VecDeque<String>,
    exit_code: Option<i32>,
    artifacts: Option<PathBuf>,
}

#[derive(Serialize)]
struct JobView {
    id: u32,
    created: String,
    requester: String,
    task: String,
    num_envs: u32,
    iterations: u32,
    seed: Option<u32>,
    status: Status,
    progress: Progress,
    exit_code: Option<i32>,
    has_artifacts: bool,
}

impl Job {
    fn view(&self) -> JobView {
        JobView {
            id: self.id,
            created: self.created.clone(),
            requester: self.requester.clone(),
            task: self.task.clone(),
            num_envs: self.num_envs,
            iterations: self.iterations,
            seed: self.seed,
            status: self.status.clone(),
            progress: self.progress.clone(),
            exit_code: self.exit_code,
            has_artifacts: self.artifacts.is_some(),
        }
    }
}

struct Inner {
    queue: VecDeque<u32>,
    jobs: Vec<Job>,
    next_id: u32,
    running: Option<u32>,
    child_pid: Option<u32>,
}

struct App {
    cfg: Cfg,
    inner: Mutex<Inner>,
    tx: broadcast::Sender<serde_json::Value>,
}

type Shared = std::sync::Arc<App>;

// -------------------------------------------------------------------------------------------- time

fn now_string() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ------------------------------------------------------------------------------------------- worker

async fn worker(app: Shared) {
    loop {
        let next = {
            let mut g = app.inner.lock().unwrap();
            match g.queue.pop_front() {
                Some(id) => {
                    let idx = g.jobs.iter().position(|j| j.id == id).unwrap();
                    g.jobs[idx].status = Status::Running;
                    g.running = Some(id);
                    let job = g.jobs[idx].clone();
                    let _ = app.tx.send(json!({"type": "jobs"}));
                    Some(job)
                }
                None => None,
            }
        };

        match next {
            Some(job) => run_job(app.clone(), job).await,
            None => tokio::time::sleep(Duration::from_millis(400)).await,
        }
    }
}

async fn run_job(app: Shared, job: Job) {
    let cfg = app.cfg.clone();
    let (python, needs_p) = cfg.interpreter();
    let script = cfg.repo.join("scripts").join("train.py");

    let mut cmd = if cfg!(windows) && python.extension().is_some_and(|e| e.eq_ignore_ascii_case("bat")) {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/c").arg(&python);
        c
    } else {
        tokio::process::Command::new(&python)
    };
    if needs_p {
        cmd.arg("-p");
    }
    cmd.arg(&script)
        .arg("--rl_library").arg("rsl_rl")
        .arg("--task").arg(&job.task)
        .arg("--viz").arg("none")
        .arg("--num_envs").arg(job.num_envs.to_string())
        .arg("--max_iterations").arg(job.iterations.to_string());
    if let Some(seed) = job.seed {
        cmd.arg("--seed").arg(seed.to_string());
    }
    cmd.current_dir(&cfg.repo)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1");

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            push_line(&app, job.id, format!("failed to start {}: {e}", python.display()));
            finish(&app, job.id, None, None);
            return;
        }
    };

    let pid = child.id();
    {
        let mut g = app.inner.lock().unwrap();
        g.child_pid = pid;
    }
    push_line(&app, job.id, format!("$ {} {}", python.display(), script.display()));

    if let Some(out) = child.stdout.take() {
        let app2 = app.clone();
        let id = job.id;
        tokio::spawn(async move {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                handle_line(&app2, id, line);
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        let app2 = app.clone();
        let id = job.id;
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                handle_line(&app2, id, line);
            }
        });
    }

    let status = child.wait().await;
    let code = status.as_ref().ok().and_then(|s| s.code());
    {
        let mut g = app.inner.lock().unwrap();
        g.child_pid = None;
        g.running = None;
    }

    let cancelled = {
        let g = app.inner.lock().unwrap();
        g.jobs.iter().find(|j| j.id == job.id).map(|j| j.cancelled).unwrap_or(false)
    };
    let success = !cancelled && code == Some(0);
    let artifacts = if success { collect_artifacts(&app, &job) } else { None };
    finish(&app, job.id, code, artifacts);
    let _ = app.tx.send(json!({"type": "jobs"}));
}

fn finish(app: &Shared, id: u32, code: Option<i32>, artifacts: Option<PathBuf>) {
    let mut g = app.inner.lock().unwrap();
    if let Some(job) = g.jobs.iter_mut().find(|j| j.id == id) {
        job.exit_code = code;
        if job.cancelled {
            job.status = Status::Cancelled;
        } else if code == Some(0) {
            job.status = Status::Done;
        } else {
            job.status = Status::Failed;
        }
        job.artifacts = artifacts;
    }
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // CSI sequence: ESC [ ... final byte in @..~
            if chars.peek() == Some(&'[') {
                chars.next();
                for c2 in chars.by_ref() {
                    if ('@'..='~').contains(&c2) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn handle_line(app: &Shared, id: u32, line: String) {
    {
        let mut g = app.inner.lock().unwrap();
        let Some(job) = g.jobs.iter_mut().find(|j| j.id == id) else { return };
        if job.log.len() >= LOG_CAP {
            job.log.pop_front();
        }
        job.log.push_back(line.clone());

        let clean = strip_ansi(&line);
        let t = clean.trim();

        if let Some(pos) = t.find("Learning iteration ") {
            let rest = &t[pos + "Learning iteration ".len()..];
            let rest = rest.trim_start();
            if let Some((it_s, total_s)) = rest.split_once('/') {
                if let Ok(it) = it_s.trim().parse::<u32>() {
                    job.progress.iteration = it;
                    if let Some(total) = total_s.split_whitespace().next().and_then(|s| s.parse::<u32>().ok()) {
                        job.progress.total = total;
                    }
                    let avg = if job.iter_times.is_empty() {
                        None
                    } else {
                        Some(job.iter_times.iter().sum::<f64>() / job.iter_times.len() as f64)
                    };
                    if let (Some(a), true) = (avg, job.progress.total > it) {
                        job.progress.iter_s = Some(a);
                        job.progress.eta_s = Some(((job.progress.total - it) as f64 * a) as u64);
                    }
                }
            }
        } else if let Some(pos) = t.find("Mean reward:") {
            job.progress.reward = t[pos + "Mean reward:".len()..].trim().parse::<f64>().ok();
        } else if let Some(pos) = t.find("Mean episode length:") {
            job.progress.ep_len = t[pos + "Mean episode length:".len()..].trim().parse::<f64>().ok();
        } else if let Some(pos) = t.find("Iteration time:") {
            let v = t[pos + "Iteration time:".len()..].trim().trim_end_matches('s');
            if let Ok(v) = v.parse::<f64>() {
                job.iter_times.push_back(v);
                while job.iter_times.len() > 20 {
                    job.iter_times.pop_front();
                }
            }
        }

        let progress = job.progress.clone();
        let _ = app.tx.send(json!({"type": "progress", "id": id, "progress": progress}));
    }
    let _ = app.tx.send(json!({"type": "log", "id": id, "line": line}));
}
fn push_line(app: &Shared, id: u32, line: String) {
    {
        let mut g = app.inner.lock().unwrap();
        if let Some(job) = g.jobs.iter_mut().find(|j| j.id == id) {
            if job.log.len() >= LOG_CAP {
                job.log.pop_front();
            }
            job.log.push_back(line.clone());
        }
    }
    let _ = app.tx.send(json!({"type": "log", "id": id, "line": line}));
}

fn collect_artifacts(app: &Shared, job: &Job) -> Option<PathBuf> {
    // Isaac Lab writes runs to logs/rsl_rl/<experiment_name>/<timestamp>/, where the experiment name
    // comes from the agent config (`r1_flat`), not from the task id. Scan every experiment directory
    // and take the newest run created after this job was queued.
    // Isaac Lab resolves its log root against the process working directory, and labd runs the task
    // with the repository as cwd, so runs can be under either tree depending on how the job was
    // launched. Scan both.
    let roots = [
        app.cfg.repo.join("logs").join("rsl_rl"),
        app.cfg.isaaclab.join("logs").join("rsl_rl"),
    ];
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    for root in roots.iter().filter(|r| r.is_dir()) {
    for experiment in std::fs::read_dir(root).ok().into_iter().flatten().flatten() {
        let exp_path = experiment.path();
        if !exp_path.is_dir() {
            continue;
        }
        let runs = match std::fs::read_dir(&exp_path) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for run in runs.flatten() {
            let path = run.path();
            if !path.is_dir() {
                continue;
            }
            let modified = run.metadata().and_then(|m| m.modified()).unwrap_or(UNIX_EPOCH);
            let fresh = modified
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() + 10 >= job.created_unix)
                .unwrap_or(true);
            if !fresh {
                continue;
            }
            let looks_like_a_run = path.join("params").is_dir() || path.join("run.json").is_file();
            if !looks_like_a_run {
                continue;
            }
            if newest.as_ref().is_none_or(|(t, _)| modified > *t) {
                newest = Some((modified, path));
            }
        }
    }
    }
    let (_, run_dir) = newest?;
    let dst = app.cfg.artifacts_root().join(job.id.to_string());
    if copy_dir(&run_dir, &dst).is_ok() {
        Some(dst)
    } else {
        None
    }
}
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------- routes

async fn index() -> Html<&'static str> {
    Html(INDEX)
}

async fn list_jobs(State(app): State<Shared>) -> Json<Vec<JobView>> {
    let g = app.inner.lock().unwrap();
    Json(g.jobs.iter().map(|j| j.view()).collect())
}

#[derive(Deserialize)]
struct NewJob {
    task: String,
    num_envs: u32,
    max_iterations: u32,
    seed: Option<String>,
    requester: Option<String>,
}

async fn create_job(State(app): State<Shared>, Form(form): Form<NewJob>) -> Response {
    let cfg = app.cfg.clone();
    if !cfg.task_list().contains(&form.task) {
        return (StatusCode::BAD_REQUEST, format!("task '{}' is not enabled on this machine", form.task)).into_response();
    }
    if !(1..=16384).contains(&form.num_envs) {
        return (StatusCode::BAD_REQUEST, "num_envs must be 1..16384").into_response();
    }
    if !(1..=20000).contains(&form.max_iterations) {
        return (StatusCode::BAD_REQUEST, "max_iterations must be 1..20000").into_response();
    }
    let seed = form.seed.and_then(|s| s.trim().parse::<u32>().ok());
    let requester = form.requester.unwrap_or_else(|| "anonymous".into()).chars().take(32).collect();
    {
        let mut g = app.inner.lock().unwrap();
        let id = g.next_id;
        g.next_id += 1;
        g.queue.push_back(id);
        g.jobs.push(Job {
            id,
            created: now_string(),
            created_unix: now_unix(),
            requester,
            task: form.task,
            num_envs: form.num_envs,
            iterations: form.max_iterations,
            seed,
            status: Status::Queued,
            cancelled: false,
            progress: Progress::default(),
            iter_times: VecDeque::new(),
            log: VecDeque::new(),
            exit_code: None,
            artifacts: None,
        });
        let _ = app.tx.send(json!({"type": "jobs"}));
    }
    Redirect::to("/").into_response()
}

async fn job_log(State(app): State<Shared>, AxPath(id): AxPath<u32>) -> String {
    let g = app.inner.lock().unwrap();
    g.jobs.iter().find(|j| j.id == id).map(|j| j.log.iter().cloned().collect::<Vec<_>>().join("\n")).unwrap_or_default()
}

async fn cancel_job(State(app): State<Shared>, AxPath(id): AxPath<u32>) -> Response {
    let mut pid = None;
    let mut queued = false;
    {
        let mut g = app.inner.lock().unwrap();
        let Some(idx) = g.jobs.iter().position(|j| j.id == id) else {
            return (StatusCode::NOT_FOUND, "no such job").into_response();
        };
        g.jobs[idx].cancelled = true;
        match g.jobs[idx].status {
            Status::Queued => {
                g.queue.retain(|q| *q != id);
                g.jobs[idx].status = Status::Cancelled;
                queued = true;
            }
            Status::Running => pid = g.child_pid,
            _ => return (StatusCode::CONFLICT, "job is not cancellable").into_response(),
        }
    }
    if let Some(pid) = pid {
        #[cfg(windows)]
        let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).output();
        #[cfg(not(windows))]
        let _ = std::process::Command::new("kill").args(["-TERM", &pid.to_string()]).output();
    }
    let _ = app.tx.send(json!({"type": "jobs"}));
    if queued { (StatusCode::OK, "cancelled").into_response() } else { (StatusCode::ACCEPTED, "stopping").into_response() }
}

async fn health() -> &'static str {
    "ok"
}

async fn config(State(app): State<Shared>) -> Json<serde_json::Value> {
    json!({
        "tasks": app.cfg.task_list(),
        "default_envs": 2048,
        "default_iterations": 1500
    })
    .into()
}

async fn events(State(app): State<Shared>) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = app.tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|m| m.ok())
        .map(|v| Ok(Event::default().data(v.to_string())));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn artifacts_list(State(app): State<Shared>, AxPath(id): AxPath<u32>) -> Json<Vec<String>> {
    let g = app.inner.lock().unwrap();
    let mut out = Vec::new();
    if let Some(dir) = g.jobs.iter().find(|j| j.id == id).and_then(|j| j.artifacts.clone()) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                out.push(e.file_name().to_string_lossy().to_string());
            }
        }
    }
    Json(out)
}

async fn artifact_file(State(app): State<Shared>, AxPath((id, name)): AxPath<(u32, String)>) -> Response {
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let path = {
        let g = app.inner.lock().unwrap();
        g.jobs.iter().find(|j| j.id == id).and_then(|j| j.artifacts.clone()).map(|d| d.join(&name))
    };
    let Some(path) = path else { return (StatusCode::NOT_FOUND, "no artifacts").into_response() };
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            [(header::CONTENT_TYPE, "application/octet-stream"), (header::CONTENT_DISPOSITION, "attachment")],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "no such file").into_response(),
    }
}

// --------------------------------------------------------------------------------------------- main

#[tokio::main]
async fn main() {
    let cfg = Cfg::parse();
    if !cfg.repo.join("scripts").join("train.py").is_file() {
        eprintln!("warning: {} not found under --repo", cfg.repo.join("scripts/train.py").display());
    }
    let (tx, _) = broadcast::channel::<serde_json::Value>(4096);
    let app: Shared = std::sync::Arc::new(App { cfg: cfg.clone(), inner: Mutex::new(Inner { queue: VecDeque::new(), jobs: Vec::new(), next_id: 1, running: None, child_pid: None }), tx });

    tokio::spawn(worker(app.clone()));

    let router = Router::new()
        .route("/", get(index))
        .route("/jobs", get(list_jobs).post(create_job))
        .route("/jobs/:id/log", get(job_log))
        .route("/jobs/:id/cancel", post(cancel_job))
        .route("/jobs/:id/artifacts", get(artifacts_list))
        .route("/jobs/:id/artifacts/*path", get(artifact_file))
        .route("/events", get(events))
        .route("/config", get(config)).route("/healthz", get(health))
        .with_state(app);

    let addr = format!("{}:{}", cfg.bind, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind failed");
    println!("labd listening on http://{addr}");
    println!("  repo       {}", cfg.repo.display());
    println!("  isaaclab   {}", cfg.isaaclab.display());
    println!("  artifacts  {}", cfg.artifacts_root().display());
    println!("  tasks      {}", cfg.task_list().join(", "));
    axum::serve(listener, router).await.expect("server failed");
}














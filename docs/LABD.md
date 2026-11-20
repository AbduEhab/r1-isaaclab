# labd â€” design

**Status:** implemented in `labd/` — one Rust binary. See [LABD_RUN.md](LABD_RUN.md) for running it on the GPU machine.
**One sentence:** a single executable on the GPU machine that serves a web page, owns an in-memory
queue of training jobs, runs them one at a time, and streams their logs to every browser on the LAN.

Related: [LABD_UI.md](LABD_UI.md) (interface), [`labd/ui_mockup.html`](../labd/ui_mockup.html)
(static mockup of the page).

---

## 1. Goals

- Team members queue a training run from a browser; no SSH, no client install.
- The GPU machine runs one job at a time, always, without configuration.
- Live progress and log in the browser.
- Finished artifacts (checkpoint, resolved configs, TensorBoard events) downloadable by URL.
- One binary, no database, no container orchestration, no external services.

## 2. Non-goals

- Persistence across restarts. The queue lives in memory; a restart empties it and marks the
  interrupted job failed.
- Multi-machine or multi-GPU scheduling.
- Arbitrary code execution from the web form.
- Authentication beyond a shared token (see Â§12). It is a LAN tool for a trusted team.
- Model registry, experiment comparison UI, billing.

## 3. Users and scenarios

| Who | Does what |
|---|---|
| Team member | Opens `http://<gpu-host>:8080`, picks a task, sets envs/iterations/seed, queues it, watches the log, downloads the checkpoint. |
| Maintainer | Starts the binary on the GPU host; updates the repo and restarts; reads `labd.log` when a job fails. |
| Everyone | Sees the queue order and the ETA before submitting, so nobody queues a 2-hour run blind. |

## 4. Constraints

- **One GPU, exclusive.** Two Isaac Sim instances halve throughput and can exhaust VRAM.
- **The job is a CLI invocation.** Training is `isaaclab -p scripts/train.py ... --viz none`;
  nothing about it needs an API.
- **Windows or Linux host.** The binary must not assume a shell: it invokes the interpreter directly
  (`isaaclab.bat` on Windows, `isaaclab.sh` on Linux) with an argv vector.
- **No network egress required.** Everything is on the LAN.

## 5. Architecture

One process, three parts:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ labd â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                                                               â”‚
â”‚  axum HTTP server  â”€â”€â”€â”€â”€â”€â”                                    â”‚
â”‚   /  /jobs  /events      â”‚        Arc<Mutex<State>>           â”‚
â”‚   /jobs/{id}/log         â”œâ”€â”€â”€â”€â”€â”€â–º  queue: VecDeque<Job>       â”‚
â”‚   /jobs/{id}/artifacts   â”‚         jobs:  Vec<Job>            â”‚
â”‚   /jobs/{id}/cancel      â”‚         next_id, running           â”‚
â”‚                          â”‚                                    â”‚
â”‚  worker task (tokio)  â”€â”€â”€â”˜        broadcast::Sender<Event>    â”‚
â”‚   pop FIFO â†’ spawn â†’ stream stdout â†’ parse â†’ copy artifacts   â”‚
â”‚                                          â”‚                    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                           â–¼
                     <artifacts>/<job-id>/  (model_*.pt, params/, tb/)
```

- **State** is a `VecDeque` (pending) plus a `Vec` (history) behind one mutex. Jobs are cloned into
  the response, never borrowed across await points.
- **Events** go through a `tokio::sync::broadcast` channel: every log line and progress update is
  sent once, and every connected browser receives it. Backpressure is handled by dropping the
  oldest events for a slow subscriber (broadcast lagged), not by slowing the job.
- **The worker** is one `tokio` task. It owns the child process, reads stdout line by line, updates
  the job under the lock, and broadcasts. On exit it copies the run directory and marks the job.

## 6. Job model and lifecycle

```rust
struct Job {
    id: u32,
    created_at: SystemTime,
    requester: String,        // from the form, free text, shown only
    task: TaskId,             // allowlisted
    num_envs: u32,
    max_iterations: u32,
    seed: Option<u32>,
    checkpoint: Option<PathBuf>,   // uploads/<file>, eval only
    status: Status,
    progress: Option<Progress>,    // iteration, total, reward, ep_len, eta_s
    log: Vec<String>,              // capped ring, last N lines for late joiners
    artifact_dir: Option<PathBuf>,
    exit_code: Option<i32>,
}
```

```
queued â”€â”€â–º running â”€â”€â–º done
   â”‚          â”‚
   â”‚          â””â”€â”€â”€â”€â”€â”€â–º failed (non-zero exit / spawn error / interrupted)
   â””â”€â”€â”€â”€â”€â”€â–º cancelled (only while queued)
```

Rules:

- Exactly one `running` at a time; `queued` is FIFO by `id`.
- A job cancelled while queued is removed before the worker takes it. A running job is cancelled by
  killing the child process tree; the job becomes `failed` with `exit_code = null` and a note.
- On startup, no job can be `running`: the queue is empty and any stale artifact directory is left
  untouched.

## 7. API

| Method | Path | Body / params | Response |
|---|---|---|---|
| GET | `/` | â€” | HTML page |
| POST | `/jobs` | form: `task, num_envs, max_iterations, seed, requester` | 303 to `/` with the job queued |
| GET | `/jobs` | â€” | JSON array of job summaries |
| GET | `/jobs/{id}/log` | â€” | `text/plain` last N lines (non-streaming fallback) |
| GET | `/events` | â€” | `text/event-stream` (SSE), see Â§8 |
| POST | `/jobs/{id}/cancel` | â€” | 202 / 409 if not cancellable |
| GET | `/jobs/{id}/artifacts/{name}` | â€” | file download (allowlisted names) |
| GET | `/healthz` | â€” | `200 ok` when the worker task is alive |

The form posts and redirects; the page itself is server-rendered HTML with ~100 lines of vanilla JS
for SSE. No build step, no frontend framework.

## 8. Progress and log streaming

The worker parses each stdout line. The two patterns that matter:

```
Learning iteration 804/1500
Mean reward: -12.14
Mean episode length: 982.15
Iteration time: 3.78s
Metrics/base_velocity/error_vel_xy: 0.6582
```

SSE frame:

```
event: progress
data: {"job":12,"iteration":804,"total":1500,"reward":-12.14,"ep_len":982.15,"eta_s":3060}

event: log
data: {"job":12,"line":"Learning iteration 805/1500"}
```

`eta_s` = remaining iterations Ã— the mean of the last 20 `Iteration time` values. Before the first
iteration the ETA is null and the UI shows "starting".

Lines are stored in the job's ring buffer (last 2 000) so a browser that connects late can be
populated from `GET /jobs/{id}/log` and then follow the live stream.

## 9. Artifacts

After the child exits successfully, the worker finds the newest
`<isaaclab>/logs/rsl_rl/<task>/<timestamp>/` and copies it to `<artifacts>/<job-id>/`:

```
<artifacts>/12/
â”œâ”€â”€ model_1499.pt          policy checkpoint
â”œâ”€â”€ params/env.yaml        fully resolved environment config
â”œâ”€â”€ params/agent.yaml      fully resolved agent config
â””â”€â”€ tb/events.out.tfevents.*
```

Retention: delete the oldest job's directory when the artifacts root exceeds a byte cap
(configuration `--max-artifacts-gb`, default 50). Downloads are served by an allowlist of file names
(`model_*.pt`, `params/*.yaml`, a zip of the whole directory) to avoid path traversal.

## 10. Failure modes

| Failure | Behaviour |
|---|---|
| Child exits non-zero | Job `failed`, last 200 log lines kept, stderr captured into the log. |
| Binary restarted mid-job | The job is lost with the process; the UI (after reload) shows an empty queue. Documented behaviour, not a bug. |
| GPU busy with someone else's process | The job runs slower; `nvidia-smi` is not consulted. The queue is the coordination mechanism. |
| Browser disconnects | Nothing happens; it can reconnect to `/jobs` and `/events`. |
| Log floods (e.g. MJWarp warnings) | Ring buffer caps memory; SSE drops for slow clients; the file log keeps everything. |
| Disk full | Job fails at artifact copy; the error is reported in the job. |

## 11. Configuration

```
labd --repo <r1-isaaclab> --isaaclab <IsaacLab> [--port 8080] [--bind 0.0.0.0]
     [--artifacts <dir>] [--max-artifacts-gb 50] [--token <string>]
     [--tasks Isaac-Velocity-Flat-R1,Isaac-Velocity-Rough-R1]
```

Everything else is derived: the interpreter is `<isaaclab>/python.bat` on Windows and
`<isaaclab>/isaaclab.sh -p` on Linux; `--repo` is where `scripts/train.py` lives.

## 12. Security

- LAN only, bound to an address the operator chooses.
- Optional shared token (`--token`): a cookie set by a tiny login form; every route checks it.
- The form cannot produce arbitrary arguments: the task is checked against the allowlist and all
  numeric fields are validated server-side (`num_envs` 1..=16384, `max_iterations` 1..=20000,
  `seed` 0..=u32::MAX).
- No file uploads in v1 (see Â§14). Artifact downloads are restricted to the job's own directory and
  an explicit name allowlist.

## 13. Deployment

- **Linux:** a systemd unit with `Restart=on-failure`, `WorkingDirectory=<repo>`, `User=lab`.
- **Windows:** `nssm install labd labd.exe --repo ...` or a scheduled task at logon.
- Port 8080 on the LAN; no reverse proxy needed. If TLS is required later, put Caddy in front.

## 14. Future (explicitly out of scope for v1)

1. Upload a checkpoint for an eval/play job (`POST /uploads`).
2. Per-user quotas and priorities; a paused queue.
3. Persisting history to SQLite (only if the team asks for a run archive).
4. Notifications (Teams/Slack webhook) on job completion.
5. Multi-GPU: a pool of workers, one per device, selected by `CUDA_VISIBLE_DEVICES`.

## 15. Acceptance criteria

1. `labd --repo ... --isaaclab ...` starts and serves the page on the LAN.
2. Submitting a job with valid fields queues it; the page shows it within one SSE tick.
3. A running job's iteration/reward/ETA update live in the browser.
4. A finished job offers its checkpoint and resolved configs for download.
5. A failing job shows a non-zero exit and its log; the queue continues with the next job.
6. Two queued jobs run strictly one after the other; the second starts after the first exits.
7. Killing the binary and restarting empties the queue without leaving a hung `running` job.


# labd — interface

Companion to [LABD.md](LABD.md). One page, server-rendered, live via SSE.
Open the static mockup at [`labd/ui_mockup.html`](../labd/ui_mockup.html) to see it rendered.

---

## 1. Principles

1. **One screen.** Submit form and queue on the same page; no navigation to queue a job.
2. **State is visible at a glance.** Running job with progress and ETA; queued jobs with queue
   position; done jobs with a download link; failed jobs with a log link.
3. **The log is one click away**, not always on screen (it is noisy and most people only need it
   when something breaks).
4. **No configuration on the client.** No accounts, no settings, no local storage.

## 2. Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  R1 Lab   ·   gpu-box:8080                            [ queue: 2 ]  [ ? ] │  header
├──────────────────────────────────────────────────────────────────────────┤
│  NEW JOB                                                                 │
│  Task [ Isaac-Velocity-Flat-R1 ▾ ]   Environments [ 2048 ]               │
│  Iterations [ 1500 ]    Seed [ 7 ]    Your name [ abdu ]                 │
│                                                    [ Queue job ]         │
├──────────────────────────────────────────────────────────────────────────┤
│  QUEUE                                                                   │
│  #12  ● running   Isaac-Velocity-Flat-R1   2048 env                      │
│       iter 804/1500  ████████████░░░░░░░░  53%   reward −12.1   ETA 51m  │
│       [ log ]                                                            │
│  #11  ○ queued    Isaac-Velocity-Rough-R1  4096 env   (1 ahead)          │
│  #10  ✓ done      Isaac-Velocity-Flat-R1   2048 env   reward +20.6       │
│       [ log ]  [ download ]                                              │
│  #09  ✗ failed    Isaac-Velocity-Flat-R1   1024 env   exit 1             │
│       [ log ]                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

Width: single column, max 960 px, centred. Background near-white; monospace for log and numbers.

## 3. Components

### 3.1 Header

- Product name, host:port.
- Queue badge: number of `queued` + `running` jobs.
- `?` opens a one-paragraph help (task list, what envs/iterations mean, where artifacts land).

### 3.2 New-job form

| Field | Control | Default | Validation (server-side too) |
|---|---|---|---|
| Task | `<select>` allowlist | `Isaac-Velocity-Flat-R1` | must be in `--tasks` |
| Environments | number | 2048 | 1 … 16384 |
| Iterations | number | 1500 | 1 … 20000 |
| Seed | number, empty allowed | empty (random) | 0 … 4294967295 |
| Your name | text, ≤ 32 chars | last used in this browser | non-empty |
| Queue job | submit | | disabled while a field is invalid |

On success: form clears (name kept), page reloads with the new job at the end of the queue and a
one-line confirmation: `Job #13 queued — 2 jobs ahead, ETA about 51 min.`

On rejection: the form keeps its values and shows the server's message next to the field.

### 3.3 Queue rows

One row per job, newest last. Each row:

```
#12  ● running   <task>   <envs> env
     iter 804/1500  ████████████░░░░  53%   reward −12.1   ETA 51m
     [ log ]                                  ← actions, right-aligned
```

- **Status dot**: ● blue running (pulsing), ○ grey queued, ✓ green done, ✗ red failed, ⊘ grey
  cancelled.
- **Progress bar** only for the running job; for queued jobs, queue position (`2 ahead`).
- **Numbers** in a tabular font so they don't jitter as they update.
- **Actions**: `log` (any state), `download` (done only), `cancel` (queued only).

### 3.4 Log drawer

Slides from the right, 60 % width, or full-width below the queue on narrow screens.

```
┌ job #12 · Isaac-Velocity-Flat-R1 · log ─────────────────── [ copy ] [ ✕ ] ┐
│ [14:02:11] Learning iteration 803/1500                                    │
│ [14:02:11] Mean reward: -12.14                                            │
│ [14:02:11] Mean episode length: 982.15                                    │
│ [14:02:15] Learning iteration 804/1500                                    │
│ ...                                                                       │
└───────────────────────────────────────────────────────────────────────────┘
```

- Auto-scroll pinned to the bottom; scrolling up unpins and shows a `jump to end` button.
- `copy` copies the visible buffer; the full log stays on the server
  (`GET /jobs/{id}/log?full=1`, plain text).
- Monospace, 12 px, dark background, one line per event, timestamps in grey.

### 3.5 Empty and edge states

| State | What is shown |
|---|---|
| No jobs ever | `Nothing queued yet. Pick a task above and set how long it should train.` |
| Queue empty but running | normal row |
| Worker dead (healthz fails) | red banner: `Worker is not running. Jobs will queue but not start.` |
| Server unreachable (SSE closed) | thin amber bar: `Lost connection — retrying…`, retried every 2 s |
| Job failed | row red, `exit N`, log drawer opens at the failure line |

## 4. Live updates

The page holds one `EventSource("/events")`.

| SSE event | UI effect |
|---|---|
| `progress` | update iteration, bar, reward, ETA in that job's row |
| `log` | append to the drawer if that job's drawer is open; keep the ring for later |
| `job` | status change (queued→running→done/failed): re-render that row |
| `queue` | update the header badge and queue positions |

Rules: updates are applied in place (no full reload); numbers are formatted with one decimal for
reward and two for velocity error; ETA `51m`, `1h 12m`, or `—` if unknown.

## 5. Copy

| Place | Text |
|---|---|
| Submit button | `Queue job` |
| Confirmation | `Job #13 queued — 2 jobs ahead, ETA about 51 min.` |
| Validation: envs | `Environments must be between 1 and 16384.` |
| Validation: iterations | `Iterations must be between 1 and 20000.` |
| Validation: task | `That task is not enabled on this machine.` |
| Cancel button | `cancel` → confirm `Cancel job #13?` |
| Download | `download` (zip of the run directory) |
| Done row | `reward +20.6` |
| Failed row | `exit 1` |

## 6. Accessibility and input

- Every control reaches `Tab` in reading order; the form validates on submit, not per keystroke.
- Status is text (`running`, `queued`, …), not colour alone.
- Log is a `<pre>` with `aria-live="polite"`; progress has `role="progressbar"` with
  `aria-valuenow`.
- Dark mode follows `prefers-color-scheme`; the page is usable on a phone (fields stack, drawer
  becomes full width).

## 7. What the page deliberately does not have

- No charts (TensorBoard is one URL away on the same host).
- No model registry or comparison table.
- No login screen in v1 (optional shared token in LABD §12).
- No uploads in v1.

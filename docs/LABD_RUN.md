# Running labd on the 5090 machine

`labd.exe` is a single file: an HTTP server with an in-memory training queue. It runs the sims on
the machine it is started on, one job at a time, and serves a page the team opens by URL.

**Fastest path:** the one-shot setup puts Isaac Sim, Isaac Lab, this repository, `labd` and its
artifacts under one root, then verifies the stack with a two-iteration run:

```powershell
.\scripts\setup-isaac.ps1 -IsaacSimZip D:\downloads\isaac-sim-6.2.0.zip
#   or, reusing an Isaac Sim already installed:
.\scripts\setup-isaac.ps1 -IsaacSimPath C:\isaac-sim
```

It creates `<Root>\start-labd.cmd` (default root `C:\isaac`); run that and open
`http://<machine>:8080`. The manual steps are below for reference.

Design and interface: [LABD.md](LABD.md) · [LABD_UI.md](LABD_UI.md).

## What the machine needs (once)

1. **Isaac Sim 6.2.0** (binary install, e.g. `C:\isaac-sim`).
2. **Isaac Lab v3.0.0** from the repository, installed with its bundled Python:
   ```
   git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git C:\tools\IsaacLab
   cd C:\tools\IsaacLab && mklink /D _isaac_sim C:\isaac-sim && isaaclab.bat --install
   ```
3. **This repository** (`r1-isaaclab`), because the daemon runs its `scripts/train.py`:
   ```
   git clone https://github.com/AbduEhab/r1-isaaclab.git C:\tools\r1-isaaclab
   ```
4. The **Visual C++ runtime** (normally already present; Windows otherwise asks).

`labd.exe` discovers Isaac Lab's Python as `<isaaclab>\python.bat`, then
`<isaaclab>\_isaac_sim\python.bat`, then the `isaaclab.bat -p` wrapper — so either layout works.

## Start it

```powershell
C:\tools\labd\labd.exe `
  --repo C:\tools\r1-isaaclab `
  --isaaclab C:\tools\IsaacLab `
  --port 8080
```

It binds `0.0.0.0:8080` and prints:

```
labd listening on http://0.0.0.0:8080
  repo       C:\tools\r1-isaaclab
  isaaclab   C:\tools\IsaacLab
  artifacts  C:\tools\r1-isaaclab\labd-artifacts
  tasks      Isaac-Velocity-Flat-R1, Isaac-Velocity-Rough-R1
```

Open `http://<machine-name-or-ip>:8080` from any laptop on the LAN. Allow the binary through Windows
Firewall on the first run (private networks), or add a rule for TCP 8080.

Flags:

| Flag | Meaning |
|---|---|
| `--repo` | the r1-isaaclab checkout (contains `scripts/train.py`) |
| `--isaaclab` | the IsaacLab checkout (contains the Python / wrapper) |
| `--port` | default 8080 |
| `--bind` | default `0.0.0.0`; use `127.0.0.1` to keep it local |
| `--artifacts` | where finished runs are copied; default `<repo>\labd-artifacts` |
| `--tasks` | comma-separated allowlist; default the two R1 velocity tasks |

## Use it

- Pick a task, environments, iterations, seed, your name → **Queue job**.
- The running row updates live: iteration, progress bar, reward, ETA. Click **log** for the stream.
- When done, **model** downloads the final checkpoint; **files** lists
  `model_*.pt`, `params/env.yaml`, `params/agent.yaml`, TensorBoard events.
- One job runs at a time; queued jobs wait in FIFO order. The queue is in memory — restarting the
  daemon clears it.

Measured on the A5000 laptop, for planning: 2048 envs ≈ 3.5–4.5 s/iteration, so the 1500-iteration
flat task is 1.5–2 h. An RTX 5090 should be faster; raise `--num_envs` in the form only if the
iteration time does not grow, and check the first iterations of the log.

## Stop it

`Ctrl+C` in its console. A running job is killed with its process tree; queued jobs are lost.
To run it as a service on Windows:

```powershell
nssm install labd C:\tools\labd\labd.exe --repo C:\tools\r1-isaaclab --isaaclab C:\tools\IsaacLab --port 8080
nssm start labd
```

## Notes

- Jobs run with the repository as working directory; Isaac Lab writes its run directories under
  `<repo>\logs\rsl_rl\<experiment>\<timestamp>\` (git-ignored). The daemon copies the newest run of
  the job into `--artifacts` when the job exits 0.
- No authentication: the daemon trusts the LAN. Bind `127.0.0.1` or use a firewall rule if that is
  not acceptable.
- The form cannot pass arbitrary arguments: task is checked against `--tasks`, and envs/iterations
  are validated server-side.
- Rebuild from source: `cargo build --release` in `labd/` (Rust 1.82+; the binary is
  `target/release/labd.exe`).

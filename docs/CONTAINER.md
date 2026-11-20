# The container

A Linux image holding the same stack this machine runs on Windows: **Isaac Sim 6.2.0, Isaac Lab
v3.0.0, this repository and `labd`**. It is the reproducible answer to "the setup script has
errors": the versions are pinned in the image, so a machine either has the image or it does not.

The Windows binary install cannot be containerised — Isaac Sim containers are Linux — so the image
is built from NVIDIA's official `nvcr.io/nvidia/isaac-sim:6.2.0` and installs Isaac Lab inside it
exactly as the Windows machine does (editable packages into Isaac Sim's bundled Python, no
separate environment to activate).

## Build

```powershell
# from the repository root
docker build -f docker/Dockerfile -t r1-isaaclab:6.2 .
```

- First pull is Isaac Sim's image (~8 GB), then the image build downloads Isaac Lab's dependencies
  (torch cu130, Newton, RSL-RL; several GB). Expect **1–2 hours** and **~35 GB** of disk.
- `docker build --progress=plain` shows the long `isaaclab.sh --install` step line by line.

## Run

```powershell
docker run --gpus all -p 8080:8080 -v r1-artifacts:/artifacts r1-isaaclab:6.2
```

then open `http://<machine>:8080` — the same queue page as `labd.exe`, with jobs running inside the
container. Artifacts land in the `r1-artifacts` volume:

```powershell
docker run --rm -v r1-artifacts:/artifacts -v ${PWD}:/out alpine cp -r /artifacts/. /out/
```

Or with compose (adds the Kit caches so restarts don't recompile shaders):

```powershell
docker compose -f docker/docker-compose.yml up -d --build
```

## What is inside

| Layer | Content |
|---|---|
| `nvcr.io/nvidia/isaac-sim:6.2.0` | Isaac Sim 6.2, its bundled Python 3.12, CUDA stack |
| `/opt/IsaacLab` | Isaac Lab v3.0.0 (repo), `_isaac_sim` → `/isaac-sim`, packages installed editable, Newton + RL extras |
| `/opt/r1-isaaclab` | this repository: task, asset, scripts |
| `/usr/local/bin/labd` | the queue daemon, built for Linux in a Rust stage |
| `/artifacts` | finished runs (checkpoints, `params/*.yaml`, TensorBoard events) |

`ENTRYPOINT` is `labd` with `--repo /opt/r1-isaaclab --isaaclab /opt/IsaacLab --artifacts /artifacts`,
so the image *is* the training server.

## Differences from the Windows setup

| | Windows (this machine) | Container |
|---|---|---|
| Isaac Sim | binary zip | official container image |
| Isaac Lab install | `isaaclab.bat --install` | `isaaclab.sh --install newton,rl` |
| Python | bundled 3.12 | bundled 3.12 (same) |
| GUI / `--viz kit` | possible | not available; training is headless (`--viz none`) |
| Video capture | failed with a Kit extension error | possible via `--video`, but headless UI is not |

Training and the R1 task config are identical: same task ids, same oracle of actions/observations,
same checkpoints (RSL-RL `.pt` files are portable).

## GPU requirements

The host needs a working NVIDIA driver and the Docker GPU runtime:

- **Linux**: `nvidia-container-toolkit`, then `docker run --gpus all`.
- **Windows**: Docker Desktop with the WSL2 backend and an NVIDIA driver; `--gpus all` works in WSL2.
  Validate with `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`.

Isaac Sim needs roughly 8 GB of video memory for a single training instance; our 2048-env run used
under 2 GB on the A5000, so the RTX 5090 has ample headroom.

## Rebuilding after a code change

Task code is baked in, so after changing `r1_isaaclab/` rebuild the top layer only (it is last, the
expensive install layers stay cached):

```powershell
docker build -f docker/Dockerfile -t r1-isaaclab:6.2 .
```

To iterate on task code without rebuilding, mount it over the copy:

```powershell
docker run --gpus all -p 8080:8080 -v ${PWD}:/opt/r1-isaaclab r1-isaaclab:6.2
```

`labd` serves `/jobs/{id}/artifacts/...` from `/artifacts`; the log drawer streams through the same
SSE endpoint. No authentication, as with the Windows binary: keep it on the LAN.

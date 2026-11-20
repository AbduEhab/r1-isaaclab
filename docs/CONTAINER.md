# The container

An **environment-only** image: Isaac Sim 6.2.0 + Isaac Lab v3.0.0, pinned. No project code, no
daemon. The repository is mounted at run time, the same way you would use it on Windows.

The reason for the image is reproducibility: Isaac Sim plus its Python stack is the expensive,
version-sensitive part. The task config changes every experiment; it does not belong in an image.

| Piece | Where |
|---|---|
| Isaac Sim 6.2.0 (official container, Linux) | image |
| Isaac Lab v3.0.0 + torch cu130 + Newton + RSL-RL | image |
| `r1-isaaclab` (asset, rewards, `scripts/train.py`) | bind mount |
| Checkpoints, TensorBoard events | host directory or volume |

The Windows binary install cannot be containerised — Isaac Sim containers are Linux — but this is
the same stack: same Isaac Sim version, same Isaac Lab tag, same task ids and the same action /
observation contract, so checkpoints move between them freely.

## Build

```powershell
# from the repository root
docker build -f docker/Dockerfile -t isaac-env:6.2 .
```

- First pull is Isaac Sim's image (~8 GB), then the build downloads Isaac Lab's dependencies
  (several GB). Expect **1–2 hours** and **~35 GB** on the first build.
- `--progress=plain` shows the `isaaclab.sh --install` step line by line.
- After that, changes to our repository need no rebuild at all.

## Run

```powershell
docker run --gpus all -it -v ${PWD}:/workspace isaac-env:6.2
```

That drops you into bash with the repository at `/workspace`. Isaac Lab is at `/opt/IsaacLab`; its
bundled Python is what `isaaclab.sh -p` uses.

Train the R1 flat task, exactly as on Windows:

```bash
cd /opt/IsaacLab
./isaaclab.sh -p /workspace/scripts/train.py \
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 \
  --viz none --num_envs 2048 --max_iterations 1500
```

Play a checkpoint (headless; there is no GUI in the container):

```bash
./isaaclab.sh -p /workspace/scripts/play.py \
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 \
  --checkpoint /workspace/models/r1_flat_1500it/model_1499.pt --viz none
```

Or with compose (adds the Kit shader caches so restarts do not recompile shaders):

```powershell
docker compose -f docker/docker-compose.yml run --rm isaac
```

## Does the R1 task work from a mount?

Yes. Nothing in the task is installed: `scripts/train.py` puts the repository root on `sys.path` and
imports `r1_isaaclab`, and the USD asset (9 MB, in `assets/r1`) is referenced with relative
payloads (`@./payloads/...@`), so it loads from any path. There are no absolute host paths in the
code — the asset path is derived from `Path(__file__)`. (The only Windows path in the tree is inside
a USD `doc` metadata string, which has no effect.)

## Differences from the Windows setup

| | Windows | Container |
|---|---|---|
| Isaac Sim | binary zip | official container image |
| Isaac Lab install | `isaaclab.bat --install` | `isaaclab.sh --install newton,rl` |
| Python | bundled 3.12 | bundled 3.12 (same) |
| GUI / `--viz kit` | possible | not available; train headless (`--viz none`) |
| `labd` queue | `labd.exe` on the host | not in this image; run it on the host if you want the web queue |
| Task code | working tree | bind mount |

## GPU requirements

- **Linux**: `nvidia-container-toolkit`, then `docker run --gpus all`.
- **Windows**: Docker Desktop with the WSL2 backend; `--gpus all` works in WSL2. Validate with
  `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`.

Isaac Sim wants roughly 8 GB of VRAM for one instance. Our 2048-env A5000 run used under 2 GB, so
an RTX 5090 has room for 4096–8192 envs.

## Notes

- `CMDLINE` default is `/bin/bash`; nothing starts by itself.
- The image is headless: no `--viz kit`, no video capture.
- GPU memory is not shared with the Windows host install: stop the Windows sim before training in
  the container, or the second process will contend for the GPU.

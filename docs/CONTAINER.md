# The container

An **environment-only** image: Isaac Sim 6.2.0 + Isaac Lab v3.0.0, pinned. No project code, no
daemon. The repository is mounted at run time, exactly as it is used on Windows.

The image exists because Isaac Sim plus its Python stack is the expensive, version-sensitive part.
The task config changes every experiment; it does not belong in a 48 GB image.

| Piece | Where |
|---|---|
| Isaac Sim 6.2.0 (official NVIDIA container, Linux) | image |
| Isaac Lab v3.0.0, torch 2.11.0+cu130, Newton 1.6.1, warp 1.17.0, RSL-RL | image |
| `r1-isaaclab` (asset, rewards, `scripts/train.py`) | bind mount |
| Checkpoints, TensorBoard events | host directory or volume |

The Windows binary install cannot be containerised (containers are Linux), but this is the same
stack: same Isaac Sim version, same Isaac Lab tag, same task ids and the same action/observation
contract, so checkpoints move between them freely.

## Build

```powershell
# from the repository root
docker build -f docker/Dockerfile -t isaac-env:6.2 .
```

- First pull is NVIDIA's Isaac Sim image (31.9 GB compressed layers), then the build downloads Isaac
  Lab's dependencies (several GB). Expect **30–90 minutes** and **48 GB** for the final image.
- `--progress=plain` shows the `isaaclab.sh --install` step line by line.
- After the install layer is cached, changes to the Dockerfile's end or to our repository need no
  rebuild; keep edits above the install step to a minimum.

### Build behind a corporate TLS proxy

If the network intercepts TLS (corporate proxy with its own root CA), pip and git inside the build
fail with `CERTIFICATE_VERIFY_FAILED` / `self-signed certificate in certificate chain`. Pass the CA
bundle as a build secret; the Dockerfile installs it into the image's trust store before cloning and
installing:

```powershell
docker build --secret id=corp_ca,src=C:\certs\corporate-roots.pem `
  -f docker/Dockerfile -t isaac-env:6.2 .
```

Without the secret the step is a no-op, so the same Dockerfile works on an unfiltered network.

## Verified

Built and tested on this machine (RTX A5000 16 GB, Docker Desktop with WSL2):

| Check | Result |
|---|---|
| `docker run --gpus all isaac-env:6.2 nvidia-smi` | GPU visible, driver 582.41 |
| `isaaclab`, `torch`, `warp` inside | `28.0.0`, `2.11.0+cu130`, `1.17.0`; `torch.cuda.is_available()` true |
| R1 task from the mounted repo | `Isaac-Velocity-Flat-R1`, 8 envs, 2 iterations, exit 0, 39.7 s |

## Run

```powershell
docker run --gpus all -it -v ${PWD}:/workspace isaac-env:6.2
```

That drops you into bash with the repository at `/workspace`; Isaac Lab is at `/opt/IsaacLab`.

Train the R1 flat task:

```bash
cd /opt/IsaacLab
./isaaclab.sh -p /workspace/scripts/train.py \
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 \
  --viz none --num_envs 2048 --max_iterations 1500
```

Play a checkpoint (headless — the image has no GUI):

```bash
./isaaclab.sh -p /workspace/scripts/play.py \
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 \
  --checkpoint /workspace/models/r1_flat_1500it/model_1499.pt --viz none
```

Or with compose:

```powershell
docker compose -f docker/docker-compose.yml run --rm isaac
```

## Using it on another machine

The image is portable; carry it as a file or push it to a registry.

```powershell
# this machine, once
docker save isaac-env:6.2 -o isaac-env-6.2.tar

# other machine
docker load -i isaac-env-6.2.tar
```

or

```powershell
docker tag isaac-env:6.2 <registry>/isaac-env:6.2
docker push <registry>/isaac-env:6.2
```

Requirements on the target: Linux (or Windows with Docker Desktop + WSL2), an **NVIDIA GPU** with a
recent driver, and the container GPU runtime (`nvidia-container-toolkit` on Linux; `--gpus all`).
macOS cannot run it.

## Does the R1 task work from a mount?

Yes. Nothing in the task is installed: `scripts/train.py` puts the repository root on `sys.path` and
imports `r1_isaaclab`, and the USD asset references its payloads relatively (`@./payloads/...@`), so
it loads from any path. There are no absolute host paths in the code — the asset path comes from
`Path(__file__)`. The only Windows path in the tree is inside a USD `doc` metadata string, which has
no effect.

## Differences from the Windows setup

| | Windows | Container |
|---|---|---|
| Isaac Sim | binary zip | official container image |
| Isaac Lab install | `isaaclab.bat --install` | `isaaclab.sh --install newton,rl` |
| Python | bundled 3.12 | bundled 3.12 (same) |
| GUI / `--viz kit` | possible | not available; train headless (`--viz none`) |
| `labd` queue | `labd.exe` on the host | not in this image; run it on the host |
| Task code | working tree | bind mount |

## Notes

- The image has no entrypoint: `docker run <image> <command>` runs the command directly, and the
  default is `/bin/bash`.
- GPU memory is per machine: stop the Windows sim before training in the container, or the two
  contend for the GPU.
- Isaac Sim wants roughly 8 GB of VRAM for one instance; our 2048-env A5000 run used under 2 GB, so
  an RTX 5090 has room for 4096–8192 envs.

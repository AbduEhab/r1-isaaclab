# Installing the environment

Everything below was verified on **Windows 11**, **Isaac Sim 6.2.0** (binary install at
`C:\isaac-sim`) and **Isaac Lab v3.0.0** from the repository, on an **RTX A5000 Laptop (16 GB)**.

## 1. Isaac Sim

Install Isaac Sim 6.2.0 from the binary download. The repository expects the layout that ships
`python.bat`, `isaac-sim.bat` and the `kit/` folder; the path used here is `C:\isaac-sim`.

## 2. Isaac Lab — the documented binary route

Isaac Lab's guide is explicit about two things that matter:

> Combining an Isaac Sim **binary** installation with a conda, uv or venv virtual environment is
> **no longer supported**. Use Isaac Sim's bundled Python via `isaaclab.bat -p` instead.

and, for a pip-wheel install into the pip form of Isaac Sim, a *different* guide applies. Do not
mix them: install Isaac Lab **from the repository** into the binary install's own Python.

```powershell
cd C:\per\thesis\tools            # or anywhere
git clone --branch v3.0.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
cmd /c mklink /D _isaac_sim C:\isaac-sim     # junction, no admin needed
isaaclab.bat --install                       # installs the submodules + RL frameworks
```

`--install` pins and downloads its own dependencies, including **torch 2.11.0+cu130** from
`download.pytorch.org` (large, ~1.9 GB) and **Newton 1.6.1**. Verify:

```powershell
isaaclab.bat -p -c "import torch, isaaclab; print(torch.__version__, torch.cuda.is_available(), isaaclab.__file__)"
# 2.11.0+cu130 True C:\...\IsaacLab\source\isaaclab\isaaclab\__init__.py
```

A warning that `_isaac_sim\setup_conda_env.bat` is missing is expected for some 6.2 binary
packages; it only affects conda/venv use, which is unsupported with a binary install anyway.

## 3. Sanity-check the framework before touching the R1

```powershell
# a stock task, NVIDIA's G1 humanoid, 5 iterations
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
isaaclab.bat -p scripts\reinforcement_learning\train.py --rl_library rsl_rl `
  --task Isaac-Velocity-Flat-G1 --viz none --num_envs 8 --max_iterations 5
```

It must finish with `Training time: ...` and exit 0. If it does not, fix the environment before
looking at this repository — the R1 task assumes a working framework.

## 4. This repository

Nothing to install. The task registers itself when `scripts/train.py` imports `r1_isaaclab`:

```powershell
isaaclab.bat -p C:\per\thesis\r1-isaaclab\scripts\train.py `
  --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 --viz none --num_envs 8 --max_iterations 3
```

## Version notes

| Component | Version used |
|---|---|
| Isaac Sim | 6.2.0 |
| Isaac Lab | v3.0.0 (repository, tag) |
| Python | 3.12 (Isaac Sim's bundled interpreter) |
| PyTorch | 2.11.0+cu130 (installed by `isaaclab.bat --install`) |
| Newton | 1.6.1 |
| GPU driver | 13.0 (CUDA runtime 12.9 reported by Warp) |

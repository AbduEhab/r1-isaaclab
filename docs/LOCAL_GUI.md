# Running the policy in the Isaac Sim window on this Windows machine

The window works again after the local install was put back together; the order of the pieces matters
because Isaac Sim and Isaac Lab carry different OpenUSD and PyTorch builds.

## The working configuration

| Piece | State |
|---|---|
| Isaac Sim | binary install at `C:\isaac-sim` (unchanged) |
| Isaac Lab | **pip wheel 3.0.0** in Isaac Sim's Python (`pip install isaaclab==3.0.0`) — *not* the editable repository install |
| OpenUSD | **no pip USD packages**; `site-packages\zz-kit-usd.pth` puts Kit's own `pxr` (from `extscache\omni.usd.libs-*`) on `sys.path` and adds its `bin` to the DLL path |
| PyTorch | Kit's prebundle `torch`/`torchvision`/`torchaudio` renamed to `*.disabled-for-pip-torch`, so Kit uses the pip `torch 2.11.0+cu130` instead of loading a second copy |
| Console | `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`, otherwise the loading-screen characters crash on cp1252 |

Why each one is needed:

- **Repository install breaks the window.** It puts Isaac Lab's pip OpenUSD into Isaac Sim's Python, so
  Kit's `omni.usd.libs` and the pip bindings are two USD builds in one process: `omni.usd` fails to
  load (`no attribute 'get_context'`) and the viewport never appears. The wheel + Kit's own `pxr`
  keeps a single USD.
- **Two PyTorch copies crash Kit.** An Isaac extension imports the prebundle torch, our code imports
  the pip one, and the mixed submodules fail with `NP_SUPPORTED_MODULES`. Renaming the prebundle
  makes both sides use the pip torch.
- **Encoding.** Without UTF-8 the play summary raises `UnicodeEncodeError` on the cp1252 console.

## Command

```powershell
set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& C:\isaac-sim\python.bat ^
  C:\per\thesis\r1-isaaclab\scripts\play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 ^
  --checkpoint C:\per\thesis\r1-isaaclab\models\r1_flat_selfcoll_1000it\model_final.pt ^
  --num_envs 8 --real-time --viz kit
```

Headless training and the container do not need any of this; it only affects the window.

## Restoring the headless path

If the editable repository install is needed again (e.g. to work on Isaac Lab itself), reverse the two
renames and remove `zz-kit-usd.pth`; the window will stop working again until the wheel configuration
is restored.

# Troubleshooting

Every entry below was hit while bringing the R1 up. The first section is environment-level; the
rest are Isaac Lab / asset specifics.

## Environment

### `pip install isaaclab` into a binary Isaac Sim's Python

Symptom: the framework imports, but stock tasks fail on `warp`/buffer type errors.

Cause: this mixes two installation routes. Isaac Lab's guide says a binary Isaac Sim install must
use Isaac Sim's bundled Python through `isaaclab.bat -p`, with Isaac Lab installed **from the
repository** (`isaaclab.bat --install`). The pip wheel (`isaaclab[...]`) belongs to the
pip-Isaac-Sim route.

Fix: clone Isaac Lab at the matching tag, link `_isaac_sim`, run `isaaclab.bat --install`, remove
any pip-installed `isaaclab`.

### `ModuleNotFoundError: No module named 'numpy'` / `DLL load failed ... _pocketfft_umath`

Cause: a pip `numpy` in `kit\python\Lib\site-packages` shadows the one Isaac Sim bundles in
`extscache\...\pip_prebundle`, and the two get mixed.

Fix: do not install numpy into Isaac Sim's Python by hand. Let `isaaclab.bat --install` manage it,
or remove the pip copy. Inside the app, `sys.path` puts `site-packages` before the prebundles, so a
hand-installed package wins — that is the mechanism.

### `UnicodeEncodeError: 'charmap' codec can't encode...`

Cause: Isaac Lab's loading screen prints box-drawing characters; redirecting stdout to a file on a
cp1252 console breaks.

Fix: set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` before running.

### `train.py: error: unrecognized arguments: --headless`

Isaac Lab 3.0 replaced it. Use `--viz none`.

## Asset and task

### `KeyError: No articulations matching pattern '...'` on Newton

Cause: the Newton backend builds its MuJoCo model from the USD's **`mujoco`** variant (the
`MjcJointAPI`s). A USD imported without multi-physics conversion (`--flat`) has no `mujoco`
variant, so there is nothing for Newton to read.

Fix: import with `run_multi_physics_conversion=True` (the default in
`scripts/import_r1_urdf.py`) and spawn with `variants={"Physics": "mujoco"}`.

### `ValueError: Not all regular expressions are matched ... Available strings: ['pelvis', ...]`

Cause: body-name mismatch. The MuJoCo-converted asset names its root `pelvis`; the URDF import
names it `pelvis_link`.

Fix: set `BASE` in `env_cfg.py` to the name the USD actually uses. The available-names list in the
error message is the ground truth.

### No contacts / feet rewards stay zero

Cause: the USD nests the links under a `Geometry` Xform
(`/World/envs/env_i/Robot/Geometry/<link>`), so Isaac Lab's stock contact-sensor pattern
`Robot/[^/]*` matches the Xform, not the bodies.

Fix: `self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"` (done in
`env_cfg.py`).

### `Actuator groups cover N joints; expected M after accounting for fixed tendons`

Cause: a joint has no actuator group — for the R1, usually the two head joints.

Fix: cover every joint, including `head_pitch_joint` and `head_yaw_joint`
(`r1_isaaclab/asset.py`).

### PhysX backend only: `argument 'joint_vel' ... passed value has type ProxyArray`

Cause: an Isaac Lab 3.0 defect in the PhysX articulation data path
(`isaaclab_physx/assets/articulation/articulation_data.py`): `joint_acc` passes a `ProxyArray`
wrapper to `wp.launch`. Unwrapping it is not enough — the wrapped buffer is on the **CPU** while
the kernel launches on `cuda:0`, producing `Warp CUDA error 700`.

Status: not hit by the Newton tasks in this repository, which is why they are the supported path
here. If you need the PhysX backend, this must be worked around (a host-side finite difference
works) or fixed upstream.

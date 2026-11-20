# The R1 asset

`assets/r1/` holds the R1 as Isaac Sim sees it: `R1/R1.usda` plus payloads (geometry, materials,
and one physics payload per backend). It is generated from the team's authoritative `urdf/R1.urdf`
with Isaac Sim's own URDF importer â€” not hand-edited, not converted from MuJoCo.

## Why not the MuJoCo-converted USD

An earlier asset existed (imported from Unitree's MJCF with the MuJoCo converter). It works for
the DDS bridge, but it has three problems for Isaac Lab:

- its physics lives in a variant set selected at reference time, with no articulation root;
- the URDF is the model of record in the team's `POLICY_CHECKLIST.md`, and the MJCF is flagged
  there for carrying the G1's torque limits;
- it has 24 joints (no head).

## Rebuilding it

Two scripts, run once. First stage the URDF and its meshes:

```powershell
python scripts\prepare_r1_urdf.py `
  --urdf  <arena-r1-kit>\urdf\R1.urdf `
  --meshes <arena-r1-kit>\assets\r1\source\meshes `
  --out build\urdf
```

`R1.urdf` references 43 meshes, 42 of which ship in the kit's mesh folder; the missing
`camera_link.STL` takes its link and joint out of the staged copy (the camera link is not actuated
and has no collision geometry worth keeping). The staged result is 26 revolute + 13 fixed + 1
floating joints.

Then import it:

```powershell
isaaclab.bat -p scripts\import_r1_urdf.py --urdf build\urdf\R1.urdf --out assets\r1
```

with `fix_base=False` (free base), `merge_fixed_joints=True` (the ankle linkage and sensor frames
are fixed joints) and **the multi-physics conversion kept on** â€” that is what writes the `mujoco`
variant.

## What the importer writes

`assets/r1/R1/R1.usda` has one `Physics` variant set:

| Variant | Contains | Used by |
|---|---|---|
| `mujoco` | 26 `MjcJointAPI`s, MJCF-shaped physics | **Isaac Lab's Newton backend** (the task uses this) |
| `physics` | generic USD physics + `PhysicsArticulationRootAPI` | plain USD physics |
| `physx` | PhysX payload | PhysX backend |
| `none` | empty | â€” |

The task selects `{"Physics": "mujoco"}`; that is the variant whose joint schemas Newton's
MuJoCo-Warp builder reads. `import_r1_urdf.py --flat` (no multi-physics conversion) drops the
`mujoco` variant and makes the asset unusable for Newton â€” that flag exists for experimenting with
the PhysX backend only.

## What the asset contains

- 26 revolute joints: 12 legs, 2 waist, 10 arms, 2 head.
- A free base. The root body is named **`pelvis_link`** (the MuJoCo-converted asset called it
  `pelvis` â€” matters for contact sensors and terminations).
- Links nested under a `Geometry` Xform: `/World/envs/env_i/Robot/Geometry/<link>`.
- Initial pose is Unitree's `HOME_KEYFRAME`: pelvis at 0.76 m, hip pitch −0.1, knee 0.3, ankle pitch −0.2,
  shoulder pitch 0.35, shoulder roll ±0.18, elbow 0.87 (a crouch — straight legs make a bouncing gait
  the path of least resistance), with `soft_joint_pos_limit_factor=0.9`.
- Contact reporting enabled; self-collisions off.

## Actuators (in `r1_isaaclab/asset.py`)

| Group | Joints | kp | kd | Effort | Velocity |
|---|---|---|---|---|---|
| legs | hips, knees | 100 | 2 | 60 N m | 32 rad/s |
| ankles | ankle pitch/roll | 40 | 2 | 50 N m | 32 |
| waist | waist roll/yaw | 100 | 2 | 60 N m | 32 |
| shoulders | shoulder pitch/roll | 40 | 2 | 60 N m | 32 |
| arms | shoulder yaw, elbow, wrist roll | 20 | 1 | 33 N m | 32 |
| head_pitch | head pitch | 50 | 2 | 33 N m | 32 |
| head_yaw | head yaw | 10 | 0.1 | 0.86 N m | 11 |

All groups carry `armature=0.01`. The gains and limits are the R1's own (team
`POLICY_CHECKLIST.md` rule 1), not the G1's 150â€“200. The head's yaw limit follows the URDF
(0.86 N m at 11 rad/s); older copies of the model list 33 N m and are wrong.

If an actuator group is missing, Isaac Lab warns:
`Actuator groups cover N joints; expected M after accounting for fixed tendons.` Every joint must
be covered, including the head, or those joints are undriven and flop.


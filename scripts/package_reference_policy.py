"""Package the public R1 reference policy (SpacemiT) as a deploy directory.

The reference already follows the deploy contract: 83 observations (no base linear velocity),
24 actions, single frame (no history). This writes ``deploy/runs/unitree_reference/`` with
``policy.onnx`` (the reference graph, untouched) and ``deploy.yaml`` in the R1 deploy format
(observation order and joint order from its ONNX metadata, gains/default pose/action scales from the
same metadata, motor slots from the R1 joint table, limits and torque limits from the URDF).

    C:\\isaac-sim\\python.bat scripts\\package_reference_policy.py
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "reference", "unitree_r1_policy")
URDF = os.path.join(REPO, "..", "arena-r1-kit", "urdf", "R1.urdf")

MOTOR_SLOTS = {
    "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2, "left_knee": 3,
    "left_ankle_pitch": 4, "left_ankle_roll": 5,
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8, "right_knee": 9,
    "right_ankle_pitch": 10, "right_ankle_roll": 11,
    "waist_roll": 12, "waist_yaw": 13,
    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
    "left_elbow": 18, "left_wrist_roll": 19,
    "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
    "right_elbow": 25, "right_wrist_roll": 26,
}

# The reference's observation order and per-frame dimensions (from its ONNX metadata).
OBS_ORDER = [
    ("base_ang_vel", 3, {}),
    ("projected_gravity", 3, {}),
    ("velocity_commands", 3, {"command_name": "base_velocity"}),
    ("gait_phase", 2, {"period": 0.6}),
    ("joint_pos_rel", 24, {}),
    ("joint_vel_rel", 24, {}),
    ("last_action", 24, {}),
]


def main() -> int:
    import yaml

    meta = json.load(open(os.path.join(REF, "metadata.json")))
    names = [n.removesuffix("_joint") for n in meta["joint_names"].split(",")]
    kp = [float(v) for v in meta["joint_stiffness"].split(",")]
    kd = [float(v) for v in meta["joint_damping"].split(",")]
    q0 = [float(v) for v in meta["default_joint_pos"].split(",")]
    # The ONNX metadata rounds the action scales to three decimals (0.312, 0.412); the R1 table's
    # exact values are 0.25 * torque_limit / kp, which is what the policy was trained with.
    def table_scale(name):
        if name.startswith(("left_hip", "right_hip", "left_knee", "right_knee", "waist")):
            return 0.15
        if "ankle" in name:
            return 0.3125
        if "shoulder_pitch" in name or "shoulder_roll" in name:
            return 0.375
        return 0.4125

    scales = [table_scale(n) for n in names]
    slots = [MOTOR_SLOTS[n] for n in names]

    root = ET.parse(URDF).getroot()
    lo_hi, effort = {}, {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        lo_hi[joint.get("name")] = [float(limit.get("lower")), float(limit.get("upper"))]
        effort[joint.get("name")] = float(limit.get("effort"))
    limits = [lo_hi[n + "_joint"] for n in names]
    torque = [effort[n + "_joint"] for n in names]

    observations = {}
    for name, dim, params in OBS_ORDER:
        observations[name] = {
            "params": params,
            "clip": None,
            "scale": [1.0] * dim,
            "history_length": 1,
        }

    out = os.path.join(REPO, "deploy", "runs", "unitree_reference")
    os.makedirs(out, exist_ok=True)
    shutil.copy2(os.path.join(REF, "policy.onnx"), os.path.join(out, "policy.onnx"))
    data = os.path.join(REF, "policy.onnx.data")
    if os.path.exists(data):
        shutil.copy2(data, os.path.join(out, "policy.onnx.data"))

    deploy = {
        "joint_ids_map": slots,
        "step_dt": 0.02,
        "stiffness": kp,
        "damping": kd,
        "default_joint_pos": q0,
        "commands": {"base_velocity": {"ranges": {
            "lin_vel_x": [-0.5, 1.0], "lin_vel_y": [-0.5, 0.5],
            "ang_vel_z": [-1.0, 1.0], "heading": None,
        }}},
        "actions": {"JointPositionAction": {
            "clip": None, "joint_names": [".*"], "scale": scales,
            "offset": q0, "joint_ids": None,
        }},
        "observations": observations,
        "joint_limits": limits,
        "torque_limits": torque,
        "meta": {
            "source": "SpacemiT model zoo (archive.spacemit.com), public R1 ONNX, not Unitree official",
            "run_path": meta.get("run_path"),
            "joint_names": meta["joint_names"],
            "observation_names": meta["observation_names"],
            "history_length": 1,
            "note": "single-frame contract: no base linear velocity and no observation history",
        },
    }
    with open(os.path.join(out, "deploy.yaml"), "w") as f:
        yaml.safe_dump(deploy, f, sort_keys=False, default_flow_style=None, width=110)
    print(f"[package] wrote {out}")
    print(f"[package] obs 83 -> actions 24, joints {len(names)}, slots {slots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

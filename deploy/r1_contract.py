"""Read a deploy directory (policy.onnx + deploy.yaml) and turn actions into motor targets.

The deploy.yaml is written by ``scripts/export_policy.py`` in Unitree's R1 deploy format: every list
is in the policy's own joint order (which is the action term's order, not the URDF order), and
``joint_ids_map`` gives the R1 motor slot for each of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Contract:
    joint_ids: list[int]          # motor slots, in policy order
    joint_names: list[str]        # with the _joint suffix
    stiffness: np.ndarray
    damping: np.ndarray
    default_pos: np.ndarray
    action_scale: np.ndarray
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    torque_limit: np.ndarray
    step_dt: float
    observations: list[tuple[str, dict, float, int]]  # (name, params, scale, history)
    command_ranges: dict
    policy_path: Path

    @property
    def n_joints(self) -> int:
        return len(self.joint_ids)


def load(deploy_dir: str | Path) -> Contract:
    d = Path(deploy_dir)
    cfg = yaml.safe_load((d / "deploy.yaml").read_text())
    meta = cfg.get("meta", {})
    names = [n + "_joint" if not n.endswith("_joint") else n for n in meta["joint_names"].split(",")]

    acts = cfg["actions"]["JointPositionAction"]
    obs = []
    for name, spec in cfg["observations"].items():
        scale = spec.get("scale") or [1.0]
        obs.append((name, spec.get("params") or {}, float(scale[0]), int(spec.get("history_length", 1))))

    limits = cfg["joint_limits"]
    return Contract(
        joint_ids=[int(s) for s in cfg["joint_ids_map"]],
        joint_names=names,
        stiffness=np.array(cfg["stiffness"], dtype=np.float32),
        damping=np.array(cfg["damping"], dtype=np.float32),
        default_pos=np.array(cfg["default_joint_pos"], dtype=np.float32),
        action_scale=np.array(acts["scale"], dtype=np.float32),
        joint_lo=np.array([l[0] for l in limits], dtype=np.float32),
        joint_hi=np.array([l[1] for l in limits], dtype=np.float32),
        torque_limit=np.array(cfg["torque_limits"], dtype=np.float32),
        step_dt=float(cfg["step_dt"]),
        observations=obs,
        command_ranges=cfg["commands"]["base_velocity"]["ranges"],
        policy_path=d / "policy.onnx",
    )


def projected_gravity(quat_xyzw, gravity_z: float = -1.0) -> np.ndarray:
    """World gravity direction expressed in the body frame. Quaternion order: (x, y, z, w)."""
    x, y, z, w = quat_xyzw
    # rotation matrix rows apply R^T to the world vector
    g = np.array([0.0, 0.0, gravity_z], dtype=np.float32)  # gravity in the world frame
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return r.T @ g


def phase_vector(seconds: float, period: float = 0.6) -> np.ndarray:
    a = 2.0 * math.pi * (seconds % period) / period
    return np.array([math.cos(a), math.sin(a)], dtype=np.float32)


@dataclass
class Targets:
    q_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


def actions_to_targets(contract: Contract, actions: np.ndarray, state: np.ndarray | None = None) -> Targets:
    """Raw policy actions -> joint targets, clamped to the joint and torque limits."""
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    if actions.size != contract.n_joints:
        raise ValueError(f"policy returned {actions.size} actions, expected {contract.n_joints}")
    q_des = contract.default_pos + actions * contract.action_scale
    q_des = np.clip(q_des, contract.joint_lo, contract.joint_hi)
    return Targets(q_des=q_des, kp=contract.stiffness, kd=contract.damping, tau_ff=np.zeros_like(q_des))

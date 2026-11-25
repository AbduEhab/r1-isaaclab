"""Export a trained RSL-RL checkpoint to policy.onnx + deploy.yaml for the real R1.

The ONNX carries its contract in metadata (joint order, stiffness, damping, default pose, action
scale, observation names, history length), and deploy.yaml mirrors Unitree's R1 deploy file so a
deployment program can build the observation and map the actions by joint name.

    C:\\isaac-sim\\python.bat scripts\\export_policy.py ^
        --checkpoint models/r1_flat_walk3_7300it/model_7300.pt --task Isaac-Velocity-Flat-R1-Deploy
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unitree R1 motor slots (r1/joints.py in the team kit, "About R1" joint table). Slots 14, 20, 21,
# 27, 28 and 31-34 are empty; head is 29/30 and is not part of the policy.
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


def motor_slot(joint_name: str) -> int:
    return MOTOR_SLOTS[joint_name.removesuffix("_joint")]
ACT_JOINT_PATTERNS = [
    ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint",
    ".*_ankle_pitch_joint", ".*_ankle_roll_joint", "waist_roll_joint", "waist_yaw_joint",
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_roll_joint",
]


def _round(xs, nd=5):
    return [round(float(x), nd) for x in xs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="Isaac-Velocity-Flat-R1-Deploy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import yaml

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    _ap = argparse.ArgumentParser()
    add_launcher_args(_ap)
    launcher_args = _ap.parse_args(["--device", args.device])
    launcher_args.enable_cameras = False

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = 0
    out = os.path.abspath(args.out or os.path.join(
        REPO, "deploy", "runs", f"{os.path.basename(os.path.dirname(args.checkpoint))}_{time.strftime('%Y%m%d_%H%M')}"
    ))
    os.makedirs(out, exist_ok=True)

    import gymnasium as gym
    import torch

    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]

        action_term = unwrapped.action_manager.get_term("joint_pos")
        # The policy's joint order is the action term's order, not the pattern-resolution order:
        # deploy.yaml must list joints, gains, scales and slots in exactly the order the network
        # was trained with.
        joint_names = list(getattr(action_term, "joint_names", []))
        if not joint_names:
            _, joint_names = robot.find_joints(ACT_JOINT_PATTERNS, preserve_order=True)
            joint_names = list(joint_names)

        art_index = {n: i for i, n in enumerate(robot.joint_names)}
        joint_idx = [art_index[n] for n in joint_names]
        default_pos = robot.data.default_joint_pos.torch[0, joint_idx].cpu().tolist()
        soft_limits = robot.data.soft_joint_pos_limits.torch[0, joint_idx].cpu().tolist()
        # Scale by joint name from the config: the term stores its scale tensor in a different
        # order than the action joint order (dict insertion order), which silently misaligned
        # joint_ids_map and the scale list the first time.
        sc_cfg = env_cfg.actions.joint_pos.scale
        if isinstance(sc_cfg, dict):
            scales = [float(sc_cfg[name]) for name in joint_names]
        else:
            sc = action_term._scale
            sc = sc.detach().cpu().reshape(-1)
            if sc.numel() != len(joint_names):
                raise RuntimeError(f"action scale has {sc.numel()} entries for {len(joint_names)} joints")
            scales = sc.tolist()
        # stiffness / damping per joint, resolved from the asset's actuator groups
        import re as _re

        def resolve_gains(name):
            found = None
            _act_list = (env_cfg.scene.robot.actuators.values() if isinstance(env_cfg.scene.robot.actuators, dict) else env_cfg.scene.robot.actuators)
            for act_cfg in _act_list:
                for pattern in act_cfg.joint_names_expr:
                    if _re.fullmatch(pattern, name):
                        kp, kd = act_cfg.stiffness, act_cfg.damping
                        if isinstance(kp, dict):
                            kp = kp[name]
                        if isinstance(kd, dict):
                            kd = kd[name]
                        found = (float(kp), float(kd), float(act_cfg.effort_limit_sim))
            if found is None:
                raise RuntimeError(f"no actuator group matches {name}")
            return found

        gains = [resolve_gains(n) for n in joint_names]
        kps = [g[0] for g in gains]
        kds = [g[1] for g in gains]
        effort = [g[2] for g in gains]

        obs_group = unwrapped.cfg.observations.policy
        obs_terms = list(unwrapped.observation_manager.active_terms["policy"])
        # group_obs_term_dim already includes history; deploy.yaml wants the per-frame dimension
        obs_dims_full = [int(d[-1]) for d in unwrapped.observation_manager.group_obs_term_dim["policy"]]
        history = int(obs_group.history_length or 1)
        obs_dims = [d // history for d in obs_dims_full]

        env.close()

    # ---- ONNX from the checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict")
    if sd is None:
        raise RuntimeError("checkpoint has neither actor_state_dict nor model_state_dict")

    def mlp_from(prefix="mlp"):
        layers = [
            torch.nn.Linear(sd[f"{prefix}.0.weight"].shape[1], sd[f"{prefix}.0.weight"].shape[0]),
            torch.nn.ELU(),
            torch.nn.Linear(sd[f"{prefix}.2.weight"].shape[1], sd[f"{prefix}.2.weight"].shape[0]),
            torch.nn.ELU(),
            torch.nn.Linear(sd[f"{prefix}.4.weight"].shape[1], sd[f"{prefix}.4.weight"].shape[0]),
            torch.nn.ELU(),
            torch.nn.Linear(sd[f"{prefix}.6.weight"].shape[1], sd[f"{prefix}.6.weight"].shape[0]),
        ]
        model = torch.nn.Sequential(*layers)
        params = {
            "0.weight": sd[f"{prefix}.0.weight"], "0.bias": sd[f"{prefix}.0.bias"],
            "2.weight": sd[f"{prefix}.2.weight"], "2.bias": sd[f"{prefix}.2.bias"],
            "4.weight": sd[f"{prefix}.4.weight"], "4.bias": sd[f"{prefix}.4.bias"],
            "6.weight": sd[f"{prefix}.6.weight"], "6.bias": sd[f"{prefix}.6.bias"],
        }
        model.load_state_dict(params)
        model.eval()
        return model

    actor = mlp_from()
    in_dim = actor[0].in_features
    out_dim = actor[-1].out_features
    if in_dim != sum(obs_dims) * history:
        raise RuntimeError(
            f"checkpoint expects {in_dim} observations but the task produces {sum(obs_dims) * history} "
            f"({obs_terms} x history {history}); export with the task the checkpoint was trained on"
        )
    dummy = torch.zeros(1, in_dim)
    onnx_path = os.path.join(out, "policy.onnx")
    torch.onnx.export(
        actor, dummy, onnx_path,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # ---- metadata inside the ONNX
    import onnx

    model_proto = onnx.load(onnx_path)
    meta = {
        "run_path": os.path.basename(os.path.dirname(args.checkpoint)),
        "checkpoint": os.path.basename(args.checkpoint),
        "task": args.task,
        "joint_names": ",".join(joint_names),
        "joint_stiffness": ",".join(f"{v:.3f}" for v in kps),
        "joint_damping": ",".join(f"{v:.3f}" for v in kds),
        "default_joint_pos": ",".join(f"{v:.3f}" for v in default_pos),
        "action_scale": ",".join(f"{v:.3f}" for v in scales),
        "command_names": "twist",
        "observation_names": ",".join(obs_terms),
        "observation_dims": ",".join(str(d) for d in obs_dims),
        "history_length": str(history),
    }
    for k, v in meta.items():
        entry = model_proto.metadata_props.add()
        entry.key, entry.value = k, str(v)
    # inline any external-data tensors so policy.onnx is a single deployable file
    from onnx.external_data_helper import load_external_data_for_model

    onnx.save(model_proto, onnx_path)
    model_proto = onnx.load(onnx_path, load_external_data=False)
    load_external_data_for_model(model_proto, os.path.dirname(onnx_path))
    onnx.save(model_proto, onnx_path)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # ---- deploy.yaml in Unitree's format (ours: history on every term, 24 joints)
    obs_spec = {}
    ranges = env_cfg.commands.base_velocity.ranges
    for term, dim in zip(obs_terms, obs_dims):
        deploy_name = {
            "base_ang_vel": "base_ang_vel", "base_lin_vel": "base_lin_vel",
            "projected_gravity": "projected_gravity",
            "velocity_commands": "velocity_commands", "joint_pos": "joint_pos_rel",
            "joint_vel": "joint_vel_rel", "actions": "last_action", "gait_phase": "gait_phase",
        }[term]
        obs_spec[deploy_name] = {
            "params": {"command_name": "base_velocity"} if term == "velocity_commands" else (
                {"period": 0.6} if term == "gait_phase" else {}),
            "clip": None,
            "scale": [1.0] * dim,
            "history_length": history,
        }
    deploy = {
        "joint_ids_map": [motor_slot(n) for n in joint_names],
        "step_dt": float(unwrapped.step_dt) if "unwrapped" in dir() else 0.02,
        "stiffness": _round(kps), "damping": _round(kds), "default_joint_pos": _round(default_pos),
        "commands": {"base_velocity": {"ranges": {
            "lin_vel_x": list(ranges.lin_vel_x), "lin_vel_y": list(ranges.lin_vel_y),
            "ang_vel_z": list(ranges.ang_vel_z), "heading": None}}},
        "actions": {"JointPositionAction": {
            "clip": None, "joint_names": [".*"], "scale": _round(scales, 6),
            "offset": _round(default_pos), "joint_ids": None}},
        "observations": obs_spec,
        "joint_limits": [[float(lo), float(hi)] for lo, hi in soft_limits],
        "torque_limits": _round(effort, 3),
        "meta": meta,
    }
    with open(os.path.join(out, "deploy.yaml"), "w") as f:
        yaml.safe_dump(deploy, f, sort_keys=False, default_flow_style=None, width=110)

    print(f"[export] obs {in_dim} -> actions {out_dim}")
    print(f"[export] joints: {joint_names}")
    print(f"[export] wrote {onnx_path} and {os.path.join(out, 'deploy.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








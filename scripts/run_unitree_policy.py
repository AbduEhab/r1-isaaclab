"""Run the public Unitree R1 policy (SpacemiT model zoo) inside our Isaac Lab task.

The policy is an ONNX export with its contract in the file metadata:
inputs 83 (base_ang_vel, projected_gravity, command, phase, joint_pos, joint_vel, actions),
outputs 24 raw actions; action_scale and default_joint_pos are in metadata, and they match the
task's own action term, so the network output can be passed to env.step() unchanged.

    isaaclab.bat -p scripts/run_unitree_policy.py --vx 0.5 --wz 0.0 --steps 500

Downloads: reference/unitree_r1_policy/{policy.onnx, weights.npz, metadata.json}.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

import r1_isaaclab  # noqa: F401,E402

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from r1_isaaclab.mdp import gait_phase  # noqa: E402

TASK = "Isaac-Velocity-Flat-R1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_DIR = os.path.join(REPO, "reference", "unitree_r1_policy")


class Policy:
    def __init__(self, policy_dir: str):
        w = np.load(os.path.join(policy_dir, "weights.npz"))
        self.mean, self.div = w["mean"], w["div"]
        self.w = [(w[f"w{i}"], w[f"b{i}"]) for i in (0, 2, 4, 6)]
        with open(os.path.join(policy_dir, "metadata.json")) as f:
            self.meta = json.load(f)
        self.joint_names = self.meta["joint_names"].split(",")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.div
        for i, (w, b) in enumerate(self.w):
            h = h @ w.T + b
            if i < len(self.w) - 1:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))  # ELU
        return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    policy = Policy(POLICY_DIR)
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 11
    env = gym.make(TASK, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    # map the policy's joint order to the articulation's
    indices, _ = robot.find_joints(policy.joint_names, preserve_order=True)
    indices = torch.tensor(indices, device=unwrapped.device)
    default_pos = robot.data.default_joint_pos[:, indices].clone()

    env.reset()
    command_term = unwrapped.command_manager.get_term("base_velocity")

    states = {"vx": [], "vy": [], "wz": [], "height": [], "left": [], "right": [], "yaw": []}
    left_ankle = policy.joint_names.index("left_ankle_pitch_joint")
    right_ankle = policy.joint_names.index("right_ankle_pitch_joint")
    last_action = torch.zeros((unwrapped.num_envs, 24), device=unwrapped.device)
    dt = unwrapped.step_dt

    for _ in range(args.steps):
        command = command_term.command.clone()
        command[:, 0], command[:, 1], command[:, 2] = args.vx, 0.0, args.wz
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:] = command
        command_term.command[:] = command

        phase = gait_phase(unwrapped, period=0.6)
        obs = torch.cat(
            [
                robot.data.root_ang_vel_b,  # 3
                robot.data.projected_gravity_b,  # 3
                command,  # 3
                phase,  # 2
                robot.data.joint_pos[:, indices] - default_pos,  # 24
                robot.data.joint_vel[:, indices],  # 24
                last_action,  # 24
            ],
            dim=-1,
        )
        obs_np = obs.detach().cpu().numpy()
        actions = np.stack([policy(obs_np[i : i + 1])[0] for i in range(obs_np.shape[0])])
        action_t = torch.tensor(actions, dtype=torch.float32, device=unwrapped.device)
        last_action = action_t
        env.step(action_t)
        torch.cuda.synchronize()

        states["vx"].append(robot.data.root_lin_vel_b.torch[0, 0].item())
        states["vy"].append(robot.data.root_lin_vel_b.torch[0, 1].item())
        states["wz"].append(robot.data.root_ang_vel_b.torch[0, 2].item())
        states["height"].append(robot.data.root_pos_w.torch[0, 2].item())
        states["left"].append(robot.data.joint_pos.torch[0, indices[left_ankle]].item())
        states["right"].append(robot.data.joint_pos.torch[0, indices[right_ankle]].item())
        from isaaclab.utils.math import euler_xyz_from_quat

        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w.torch[0].unsqueeze(0))
        states["yaw"].append(yaw[0].item())

    diff = [l - r for l, r in zip(states["left"], states["right"])]
    mean_d = sum(diff) / len(diff)
    crossings = sum(1 for a, b in zip(diff, diff[1:]) if (a - mean_d) * (b - mean_d) < 0)
    yaw_total = 0.0
    for a, b in zip(states["yaw"], states["yaw"][1:]):
        d = b - a
        yaw_total += (d + math.pi) % (2 * math.pi) - math.pi

    def mean(xs):
        return sum(xs) / len(xs)

    print()
    print("unitree reference policy, obs contract 83 (no base linear velocity)")
    print(f"command            vx={args.vx} wz={args.wz}")
    print(f"mean forward vx    {mean(states['vx']):+.3f} m/s")
    print(f"mean lateral vy    {mean(states['vy']):+.3f} m/s")
    print(f"mean yaw rate      {mean(states['wz']):+.3f} rad/s")
    print(f"heading change     {yaw_total:+.2f} rad over {args.steps * dt:.1f} s")
    print(f"base height        {mean(states['height']):.3f} m (std {np.std(states['height']):.4f})")
    print(f"step frequency     {crossings / (2.0 * args.steps * dt):.2f} Hz")
    print(f"ankle correlation  {(np.corrcoef(states['left'], states['right'])[0, 1]):+.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

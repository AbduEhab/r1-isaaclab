"""Gait report: walk vs hop for a policy (RSL-RL checkpoint or the Unitree ONNX).

Measures, per foot (ankle_roll_link), from 10 s of simulation with a fixed velocity command:

* contact fraction and simultaneous-airborne fraction â€” a hop has both feet in the air together,
  a walk has at most one,
* swing height â€” the foot must lift clearly off the floor to step,
* step length â€” forward displacement of a foot while it is airborne (a hop moves both feet
  together, so its step length is ~0),
* step frequency and forward speed.

    isaaclab.bat -p scripts/gait_report.py --checkpoint models/r1_flat_4500it/model_4499.pt --vx 0.5
    isaaclab.bat -p scripts/gait_report.py --onnx reference/unitree_r1_policy --vx 0.5
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from r1_isaaclab.agents.rsl_rl_ppo_cfg import R1FlatPPORunnerCfg  # noqa: E402
from r1_isaaclab.mdp import gait_phase  # noqa: E402

TASK = "Isaac-Velocity-Flat-R1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class OnnxPolicy:
    def __init__(self, policy_dir):
        w = np.load(os.path.join(policy_dir, "weights.npz"))
        self.mean, self.div = w["mean"], w["div"]
        self.layers = [(w[f"w{i}"], w[f"b{i}"]) for i in (0, 2, 4, 6)]
        with open(os.path.join(policy_dir, "metadata.json")) as f:
            self.meta = json.load(f)
        self.joint_names = self.meta["joint_names"].split(",")

    def __call__(self, x):
        h = (x - self.mean) / self.div
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = elu(h)
        return h


def build_env(args):
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 11
    return gym.make(TASK, cfg=env_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint")
    ap.add_argument("--onnx", default=None, help="directory with weights.npz + metadata.json")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not args.checkpoint and not args.onnx:
        ap.error("give --checkpoint or --onnx")

    env = build_env(args)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    command_term = unwrapped.command_manager.get_term("base_velocity")
    feet_ids, _ = robot.find_bodies([".*_ankle_roll_link"], preserve_order=True)
    feet = torch.tensor(feet_ids, device=unwrapped.device)
    dt = unwrapped.step_dt

    onnx_policy = None
    policy = None
    joint_indices = None
    default_pos = None
    last_action = None

    env.reset()
    if args.onnx:
        onnx_policy = OnnxPolicy(args.onnx)
        idx, _ = robot.find_joints(onnx_policy.joint_names, preserve_order=True)
        joint_indices = torch.tensor(idx, device=unwrapped.device)
        default_pos = robot.data.default_joint_pos[:, joint_indices].clone()
        last_action = torch.zeros((unwrapped.num_envs, 24), device=unwrapped.device)
    else:
        agent_cfg = R1FlatPPORunnerCfg()
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: F401
        from importlib.metadata import version as _v

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _v("rsl-rl-lib"))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(wrapped, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=unwrapped.device)
        out = wrapped.reset()
        obs = out[0] if isinstance(out, tuple) else out

    foot_x, foot_z, speeds = [], [], []
    for _ in range(args.steps):
        command = command_term.command.clone()
        command[:, 0], command[:, 1], command[:, 2] = args.vx, 0.0, args.wz
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:] = command
        command_term.command[:] = command

        if onnx_policy is not None:
            phase = gait_phase(unwrapped, period=0.6)
            obs_t = torch.cat(
                [
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    command,
                    phase,
                    robot.data.joint_pos[:, joint_indices] - default_pos,
                    robot.data.joint_vel[:, joint_indices],
                    last_action,
                ],
                dim=-1,
            ).detach().cpu().numpy()
            actions = np.stack([onnx_policy(obs_t[i : i + 1])[0] for i in range(obs_t.shape[0])])
            action_t = torch.tensor(actions, dtype=torch.float32, device=unwrapped.device)
            last_action = action_t
            with torch.inference_mode():
                env.step(action_t)
        else:
            with torch.inference_mode():
                actions = policy(obs)
            out = wrapped.step(actions)
            obs = out[0]

        pos = robot.data.body_pos_w.torch[:, feet, :].detach().cpu().numpy()
        foot_x.append(pos[:, :, 0])
        foot_z.append(pos[:, :, 2])
        speeds.append(robot.data.root_lin_vel_b.torch[0, 0].item())

    foot_x = np.array(foot_x)  # (steps, envs, 2)
    foot_z = np.array(foot_z)
    zmin = foot_z.min(axis=(0, 1))
    contact = foot_z < (zmin[None, None, :] + 0.025)  # (steps, envs, 2)
    both_air = (~contact).all(axis=2)  # both feet airborne
    any_air = (~contact).any(axis=2)

    swing_h = (foot_z.max(axis=0) - zmin)  # per env, per foot
    step_len = []
    for env_i in range(foot_z.shape[1]):
        for f in range(2):
            swings, start = [], None
            for t in range(contact.shape[0]):
                if not contact[t, env_i, f] and start is None:
                    start = t
                elif contact[t, env_i, f] and start is not None:
                    swings.append(foot_x[t, env_i, f] - foot_x[start, env_i, f])
                    start = None
            if swings:
                step_len.append(np.mean([abs(s) for s in swings]))
    fx = foot_x * 0
    fwd = np.array([foot_x[:, :, i].mean() for i in range(2)])

    cmd = f"{args.vx} m/s fwd" + (f", {args.wz} rad/s yaw" if args.wz else "")
    print()
    print("policy:", args.checkpoint or args.onnx)
    print(f"command: {cmd}")
    print(f"both feet airborne  {both_air.mean() * 100:5.1f} % of the time  (hop if high)")
    print(f"any foot airborne   {any_air.mean() * 100:5.1f} %")
    print(f"mean swing height   {swing_h.mean():.3f} m")
    print(f"mean step length    {np.mean(step_len) if step_len else 0.0:.3f} m")
    print(f"forward speed       {np.mean(speeds):+.3f} m/s")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


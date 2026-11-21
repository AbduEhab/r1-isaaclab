"""Headless gait probe: does the policy step, or slide/stand?

    isaaclab.bat -p scripts/probe_gait.py --checkpoint models/r1_flat_4500it/model_4499.pt ^
        --num_envs 8 --steps 500 --vx 0.5

Prints, for one environment, the mean forward speed, base height statistics and the step frequency
estimated from the alternating ankle-pitch motion (left minus right). A walker shows a clear periodic
signal at ~1-2 Hz and a negative left/right correlation; a hopper shows both legs moving together
(positive correlation); a standing policy shows near-zero amplitude.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import warp as wp

wp.config.enable_backward = False

import r1_isaaclab  # noqa: F401,E402  (registers the task)

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

INSTALLED_RSL_RL = _pkg_version("rsl-rl-lib")

from r1_isaaclab.agents.rsl_rl_ppo_cfg import R1FlatPPORunnerCfg  # noqa: E402

TASK = "Isaac-Velocity-Flat-R1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500, help="control steps (0.02 s each)")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--wz", type=float, default=0.0, help="yaw rate command, rad/s")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 7
    agent_cfg = R1FlatPPORunnerCfg()
    agent_cfg.max_iterations = 1

    env = gym.make(args.task, cfg=env_cfg)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(vec_env, agent_cfg)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    ankle_ids, _ = robot.find_joints(["left_ankle_pitch_joint", "right_ankle_pitch_joint"])
    left_ankle, right_ankle = ankle_ids

    out = vec_env.reset()
    obs = out[0] if isinstance(out, tuple) else out

    command_term = unwrapped.command_manager.get_term("base_velocity")
    states = {"vx": [], "height": [], "left": [], "right": [], "wz": [], "vy": [], "yaw": []}

    print("command term type:", type(command_term).__name__)
    for step_i, _ in enumerate(range(args.steps)):
        command = command_term.command.clone()
        command[:, 0] = args.vx
        command[:, 1] = 0.0
        command[:, 2] = args.wz
        # The command manager rewrites `command` from its internal buffer every step, so the
        # buffer is what must be set, not the public tensor.
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:] = command
        command_term.command[:] = command

        with torch.inference_mode():
            actions = policy(obs)
        out = vec_env.step(actions)
        obs = out[0]
        if step_i in (0, 1, 50):
            print(f"  step {step_i}: env command = {command_term.command[0].tolist()}, policy-action mean {actions.mean().item():+.3f}")

        states["vx"].append(robot.data.root_lin_vel_b.torch[0, 0].item())
        states["height"].append(robot.data.root_pos_w.torch[0, 2].item())
        states["left"].append(robot.data.joint_pos.torch[0, left_ankle].item())
        states["right"].append(robot.data.joint_pos.torch[0, right_ankle].item())
        states["wz"].append(robot.data.root_ang_vel_b.torch[0, 2].item())
        states["vy"].append(robot.data.root_lin_vel_b.torch[0, 1].item())
        quat = robot.data.root_quat_w.torch[0]
        w, x, y, z = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        yaw_now = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if states["yaw"]:
            delta = yaw_now - states["yaw"][-1]
            delta = (delta + math.pi) % (2 * math.pi) - math.pi  # unwrap
            states.setdefault("yaw_total", []).append(delta)
        states["yaw"].append(yaw_now)

    dt = unwrapped.step_dt
    mean_vx = sum(states["vx"]) / len(states["vx"])
    heights = states["height"]
    mean_h = sum(heights) / len(heights)
    std_h = math.sqrt(sum((h - mean_h) ** 2 for h in heights) / len(heights))

    # Alternation signal: a walker oscillates this around zero; a hopper keeps it near constant.
    diff = [l - r for l, r in zip(states["left"], states["right"])]
    mean_d = sum(diff) / len(diff)
    crossings = sum(1 for a, b in zip(diff, diff[1:]) if (a - mean_d) * (b - mean_d) < 0)
    frequency = crossings / (2.0 * args.steps * dt)

    left = states["left"]
    right = states["right"]
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    num = sum((l - lm) * (r - rm) for l, r in zip(left, right))
    den = math.sqrt(sum((l - lm) ** 2 for l in left) * sum((r - rm) ** 2 for r in right))
    correlation = num / den if den > 0 else 0.0

    print()
    print(f"task            {args.task}")
    print(f"checkpoint      {args.checkpoint}")
    print(f"steps           {args.steps}  ({args.steps * dt:.1f} s simulated)")
    print(f"command vx      {args.vx} m/s")
    print(f"mean forward vx {mean_vx:+.3f} m/s")
    print(f"base height     {mean_h:.3f} m  (std {std_h:.4f} m)")
    print(f"ankle amplitude left {max(left) - min(left):.3f} rad, right {max(right) - min(right):.3f} rad")
    print(f"left/right ankle correlation {correlation:+.3f}   (walk: < 0, hop: > 0, stand: ~0)")
    print(f"step frequency  {frequency:.2f} Hz")
    if args.wz:
        mean_wz = sum(states["wz"]) / len(states["wz"])
        mean_abs_wz = sum(abs(v) for v in states["wz"]) / len(states["wz"])
        mean_vy = sum(abs(v) for v in states["vy"]) / len(states["vy"])
        yaw_delta = sum(states.get("yaw_total", []))
        print(f"command wz      {args.wz} rad/s")
        print(f"yaw rate        mean {mean_wz:+.3f}, mean |wz| {mean_abs_wz:.3f} rad/s (command {args.wz})")
        print(f"mean |vy|       {mean_vy:.3f} m/s  (strafing if large)")
        print(f"heading change  {yaw_delta:+.2f} rad over {args.steps * dt:.1f} s")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










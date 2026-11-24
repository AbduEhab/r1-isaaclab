"""Show the Unitree R1 reference policy in the Isaac Sim window.

Same observation contract and policy as ``run_unitree_policy.py``, but the app is started through
Isaac Lab's launcher with ``--viz kit``, so the window opens and the robots are visible. Used for
visual comparison with our own policies.

    set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& C:\\isaac-sim\\python.bat ^
      scripts\\run_unitree_gui.py --viz kit --num_envs 8 --vx 0.5
"""

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import r1_isaaclab  # noqa: F401,E402  (registers the task before the app starts)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from run_unitree_policy import Policy, POLICY_DIR, TASK, gait_phase  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100000)
    add_launcher_args(parser)
    args = parser.parse_args()

    policy = Policy(POLICY_DIR)
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 7

    import gymnasium as gym
    import torch

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]

        idx, _ = robot.find_joints(policy.joint_names, preserve_order=True)
        indices = torch.tensor(idx, device=unwrapped.device)
        default_pos = robot.data.default_joint_pos[:, indices].clone()
        command_term = unwrapped.command_manager.get_term("base_velocity")
        last_action = torch.zeros((unwrapped.num_envs, 24), device=unwrapped.device)

        env.reset()
        print("[ref-gui] running; Ctrl+C to stop", flush=True)
        for i in range(args.steps):
            command = command_term.command.clone()
            command[:, 0], command[:, 1], command[:, 2] = args.vx, 0.0, args.wz
            if hasattr(command_term, "vel_command_b"):
                command_term.vel_command_b[:] = command
            command_term.command[:] = command

            obs = torch.cat(
                [
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    command,
                    gait_phase(unwrapped, period=0.6),
                    robot.data.joint_pos[:, indices] - default_pos,
                    robot.data.joint_vel[:, indices],
                    last_action,
                ],
                dim=-1,
            ).detach().cpu().numpy()
            actions = policy(obs)  # batched: (num_envs, 83) -> (num_envs, 24)
            action_t = torch.tensor(actions, dtype=torch.float32, device=unwrapped.device)
            last_action = action_t
            env.step(action_t)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

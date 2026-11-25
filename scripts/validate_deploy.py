"""Validate an exported deploy directory inside the simulator, before the robot.

Two checks:
1. The observation built from deploy.yaml (the same builder the robot runner uses) matches the
   environment's own observation vector. This proves the deploy.yaml observation order, history and
   scaling match what the policy was trained on.
2. The ONNX policy drives the task in closed loop: speed, height and foot-airborne fraction.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="deploy directory (policy.onnx + deploy.yaml)")
    ap.add_argument("--task", default="Isaac-Velocity-Flat-R1-Deploy")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    lap = argparse.ArgumentParser()
    add_launcher_args(lap)
    launcher_args = lap.parse_args(["--device", args.device])

    from deploy import obs_builder, r1_contract

    contract = r1_contract.load(args.policy)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(contract.policy_path), providers=["CPUExecutionProvider"])

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = 5
    # parity must be exact: turn off the observation noise the training env adds
    import dataclasses

    from isaaclab.managers import ObservationTermCfg

    grp = env_cfg.observations.policy
    for f in dataclasses.fields(type(grp)):
        term = getattr(grp, f.name)
        if isinstance(term, ObservationTermCfg):
            term.noise = None

    import gymnasium as gym
    import torch

    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(args.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        builder = obs_builder.ObsBuilder(contract)

        joint_index = {n: i for i, n in enumerate(robot.joint_names)}
        order = [joint_index[n] for n in contract.joint_names]

        command_term = unwrapped.command_manager.get_term("base_velocity")
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
        builder.reset(0.0)

        diffs, speeds, heights, feet = [], [], [], []
        feet_ids, _ = robot.find_bodies([".*_ankle_roll_link"], preserve_order=True)
        dt = unwrapped.step_dt

        for step in range(args.steps):
            cmd = command_term.command.clone()
            cmd[:, 0], cmd[:, 1], cmd[:, 2] = args.vx, 0.0, args.wz
            if hasattr(command_term, "vel_command_b"):
                command_term.vel_command_b[:] = cmd
            command_term.command[:] = cmd

            q = robot.data.joint_pos.torch[0, order].cpu().numpy()
            dq = robot.data.joint_vel.torch[0, order].cpu().numpy()
            quat = robot.data.root_quat_w.torch[0].cpu().numpy()  # Isaac Lab: (x, y, z, w)
            gyro = robot.data.root_ang_vel_b.torch[0].cpu().numpy()
            t = float(unwrapped.episode_length_buf[0].item()) * dt

            obs_deploy = builder.build(t, q, dq, quat, gyro, cmd[0].cpu().numpy())
            obs_env = unwrapped.observation_manager.compute()["policy"][0].cpu().numpy()
            if step >= 5:
                diffs.append(float(np.max(np.abs(obs_deploy - obs_env))))
                if step == 10:
                    # per-term breakdown (history 1 only), to locate a mismatch
                    dims = [int(d[-1]) for d in unwrapped.observation_manager.group_obs_term_dim["policy"]]
                    names = list(unwrapped.observation_manager.active_terms["policy"])
                    idx = 0
                    for name, dim in zip(names, dims):
                        if idx + dim <= obs_deploy.size:
                            seg = np.s_[idx: idx + dim]
                            d = np.max(np.abs(obs_deploy[seg] - obs_env[seg]))
                            print(f"  term {name:18s} maxdiff {d:.4f} deploy {np.round(obs_deploy[seg][:6],3)} env {np.round(obs_env[seg][:6],3)}")
                            idx += dim

            actions = sess.run(None, {"obs": obs_deploy.reshape(1, -1).astype(np.float32)})[0][0]
            builder.record_action(actions)
            action_t = torch.tensor(actions, dtype=torch.float32, device=unwrapped.device).reshape(1, -1)
            out = env.step(action_t)
            speeds.append(float(robot.data.root_lin_vel_b.torch[0, 0].item()))
            heights.append(float(robot.data.root_pos_w.torch[0, 2].item()))
            z = robot.data.body_pos_w.torch[0, feet_ids, 2].cpu().numpy()
            feet.append(z)

        env.close()

    feet = np.array(feet)
    zmin = feet.min()
    # crude airborne fraction: a foot is up when it is >2.5 cm above its lowest point
    airborne = (feet > zmin + 0.025).sum(axis=1)
    both = (airborne == 2).mean() * 100.0

    print()
    print("policy:", args.policy)
    print(f"observation parity   max |deploy - env| = {max(diffs) if diffs else float('nan'):.6f}")
    print(f"forward speed        {np.mean(speeds):+.3f} m/s (command {args.vx})")
    print(f"base height          {np.mean(heights):.3f} m")
    print(f"both feet airborne   {both:.1f} %")
    ok = (not diffs) or max(diffs) < 2e-4
    print("PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Play a deploy directory (policy.onnx + deploy.yaml) in the simulator through the deploy contract.

The observation is built by the same code the robot runner uses (``deploy.obs_builder``), so this is
both the visual check and the numerical check of a deploy directory before hardware.

    # headless measurement
    C:\\isaac-sim\\python.bat scripts\\play_deploy.py --policy deploy\\runs\\unitree_reference --steps 400 --vx 0.5

    # in the window
    C:\\isaac-sim\\python.bat scripts\\play_deploy.py --policy deploy\\runs\\unitree_reference --viz kit --steps 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", default="Isaac-Velocity-Flat-R1")
    ap.add_argument("--steps", type=int, default=400, help="0 = run forever (window)")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--num_envs", type=int, default=1)
    raw = sys.argv[1:]
    _pre = argparse.ArgumentParser()
    _pre.add_argument("--policy", required=True)
    _pre.add_argument("--task", default="Isaac-Velocity-Flat-R1")
    _pre.add_argument("--steps", type=int, default=400)
    _pre.add_argument("--vx", type=float, default=0.5)
    _pre.add_argument("--wz", type=float, default=0.0)
    _pre.add_argument("--num_envs", type=int, default=1)
    _ns, _ = _pre.parse_known_args(raw)
    policy, task, steps, vx, wz, num_envs = (_ns.policy, _ns.task, _ns.steps, _ns.vx, _ns.wz, _ns.num_envs)
    from isaaclab.app import add_launcher_args, launch_simulation

    add_launcher_args(ap)
    launcher_args = ap.parse_args(raw)
    launcher_args = argparse.Namespace(**{k: v for k, v in vars(launcher_args).items()})
    launcher_args.steps = steps
    launcher_args.policy = policy
    launcher_args.task = task
    launcher_args.vx = vx
    launcher_args.wz = wz
    launcher_args.num_envs = num_envs

    from isaaclab_tasks.utils import parse_env_cfg

    from deploy import obs_builder, r1_contract

    contract = r1_contract.load(launcher_args.policy)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(contract.policy_path), providers=["CPUExecutionProvider"])

    env_cfg = parse_env_cfg(launcher_args.task, device="cuda:0", num_envs=launcher_args.num_envs)
    env_cfg.seed = 17

    import gymnasium as gym
    import torch

    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(launcher_args.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = unwrapped.device
        index = {n: i for i, n in enumerate(robot.joint_names)}
        order = torch.tensor([index[n] for n in contract.joint_names], device=device, dtype=torch.long)
        feet_ids, _ = robot.find_bodies([".*_ankle_roll_link"], preserve_order=True)
        feet = torch.tensor(feet_ids, device=device)
        command_term = unwrapped.command_manager.get_term("base_velocity")

        builders = [obs_builder.ObsBuilder(contract) for _ in range(launcher_args.num_envs)]
        env.reset()
        speeds, feet_z = [], []
        n_steps = launcher_args.steps if launcher_args.steps > 0 else 10**9
        for step in range(n_steps):
            now = step * unwrapped.step_dt
            cmd = np.array([launcher_args.vx, 0.0, launcher_args.wz], dtype=np.float32)
            actions = np.zeros((launcher_args.num_envs, contract.n_joints), dtype=np.float32)
            for i in range(launcher_args.num_envs):
                q = robot.data.joint_pos.torch[i, order].cpu().numpy()
                dq = robot.data.joint_vel.torch[i, order].cpu().numpy()
                quat = robot.data.root_quat_w.torch[i].cpu().numpy()
                gyro = robot.data.root_ang_vel_b.torch[i].cpu().numpy()
                obs = builders[i].build(now, q, dq, quat, gyro, cmd)
                actions[i] = sess.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0][0]
                builders[i].record_action(actions[i])
            action_t = torch.tensor(actions, device=device)
            env.step(action_t)
            speeds.append(float(robot.data.root_lin_vel_b.torch[0, 0].item()))
            feet_z.append(robot.data.body_pos_w.torch[0, feet, 2].cpu().numpy())
            cmd_t = command_term.command.clone()
            cmd_t[:, 0], cmd_t[:, 1], cmd_t[:, 2] = launcher_args.vx, 0.0, launcher_args.wz
            if hasattr(command_term, "vel_command_b"):
                command_term.vel_command_b[:] = cmd_t
            command_term.command[:] = cmd_t
            if step % 100 == 0:
                print(f"[play] step {step}", flush=True)

        if launcher_args.steps > 0:
            z = np.array(feet_z)
            zmin = z.min()
            airborne = (z > zmin + 0.025).sum(axis=1)
            print()
            print(f"policy               {launcher_args.policy}")
            print(f"forward speed        {np.mean(speeds):+.3f} m/s (command {launcher_args.vx})")
            print(f"base height          {np.mean(z.mean(axis=1)):.3f} m (feet mean)")
            print(f"both feet airborne   {(airborne == 2).mean() * 100:.1f} %")
        env.close()
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

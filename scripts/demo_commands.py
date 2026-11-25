"""Scripted command demo: the robot tracks velocity commands while the green arrow shows them.

Each segment holds one (vx, vy, wz) command, sets it on the command term so Isaac Lab's command
visualisation (the green arrow) matches what the policy gets, and reports the measured response.

    C:\\isaac-sim\\python.bat scripts\\demo_commands.py --viz kit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, vx, vy, wz) held for `dwell` seconds
SEGMENTS = [
    ("forward 0.2", 0.2, 0.0, 0.0),
    ("forward 0.5", 0.5, 0.0, 0.0),
    ("forward 0.7", 0.7, 0.0, 0.0),
    ("turn left", 0.0, 0.0, 0.8),
    ("forward + turn left", 0.4, 0.0, 0.6),
    ("turn right", 0.0, 0.0, -0.8),
    ("forward + turn right", 0.4, 0.0, -0.6),
    ("forward slow", 0.3, 0.0, 0.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(REPO, "deploy", "runs", "r1_distilled"))
    ap.add_argument("--task", default="Isaac-Velocity-Flat-R1")
    ap.add_argument("--dwell", type=float, default=5.0)
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--usd", default=None, help="spawn an environment first: warehouse_forklifts")
    ap.add_argument("--loop", action="store_true", help="repeat the command sequence forever")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()

    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    from deploy import obs_builder, r1_contract

    contract = r1_contract.load(args.policy)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(contract.policy_path), providers=["CPUExecutionProvider"])

    import gymnasium as gym
    import torch

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 29

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = unwrapped.device
        index = {name: i for i, name in enumerate(robot.joint_names)}
        order = torch.tensor([index[name] for name in contract.joint_names], device=device, dtype=torch.long)
        command_term = unwrapped.command_manager.get_term("base_velocity")
        dt = unwrapped.step_dt
        dwell_steps = max(1, int(round(args.dwell / dt)))
        builders = [obs_builder.ObsBuilder(contract) for _ in range(args.num_envs)]

        env.reset()
        total = 0
        if args.usd:
            from r1_isaaclab.warehouse import ENVS, Occupancy, spawn, spawn_conveyors

            print("[demo] loading environment ...", flush=True)
            spawn(ENVS[args.usd])
            spawn_conveyors()
            occ = Occupancy()
            occ.build_from_stage()
            spawn_xy = occ.nearest_free([0.0, -12.0])
            state = robot.data.default_root_state.clone()
            state[:, 0], state[:, 1] = float(spawn_xy[0]), float(spawn_xy[1])
            state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
            robot.write_root_pose_to_sim(state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            print(f"[demo] robot placed at {spawn_xy.round(1)}", flush=True)

        while True:
            for name, vx, vy, wz in SEGMENTS:
                cmd = np.array([vx, vy, wz], dtype=np.float32)
                vxs, yaws = [], []
                for k in range(dwell_steps):
                    # the arrow follows the actual command
                    cmd_t = command_term.command.clone()
                    cmd_t[:, 0], cmd_t[:, 1], cmd_t[:, 2] = vx, vy, wz
                    if hasattr(command_term, "vel_command_b"):
                        command_term.vel_command_b[:] = cmd_t
                    command_term.command[:] = cmd_t

                    actions = np.zeros((args.num_envs, contract.n_joints), dtype=np.float32)
                    for i in range(args.num_envs):
                        q = robot.data.joint_pos.torch[i, order].cpu().numpy()
                        dq = robot.data.joint_vel.torch[i, order].cpu().numpy()
                        quat = robot.data.root_quat_w.torch[i].cpu().numpy()
                        gyro = robot.data.root_ang_vel_b.torch[i].cpu().numpy()
                        obs = builders[i].build(total * dt, q, dq, quat, gyro, cmd)
                        actions[i] = sess.run(None, {"obs": obs.reshape(1, -1)})[0][0]
                        builders[i].record_action(actions[i])
                    env.step(torch.tensor(actions, device=device))
                    total += 1
                    vxs.append(float(robot.data.root_lin_vel_b.torch[0, 0].item()))
                    yaws.append(float(robot.data.root_ang_vel_b.torch[0, 2].item()))

                print(f"[demo] {name:22s} cmd vx {vx:+.2f} wz {wz:+.2f} -> "
                      f"measured vx {np.mean(vxs):+.2f} m/s, yaw {np.mean(yaws):+.2f} rad/s", flush=True)

        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Navigate the room with the deployed walking policy.

A go-to-goal controller turns the goal into a velocity command (turn towards the goal, walk), and
the deploy directory's policy executes it through the same observation builder the robot runner uses.
The robot walks one episode at a time; a fall, a wall or an obstacle hit ends that episode as a
failure. Metrics: success rate, mean time to goal, failures.

    C:\\isaac-sim\\python.bat scripts\\nav_demo.py --policy deploy\\runs\\r1_distilled --episodes 12 --viz none
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOM_HALF = 2.2
GOAL_TOL = 0.35
OBSTACLES = ((0.9, 1.1), (-1.2, -0.6), (0.4, -1.6))
OBS_CLEAR = 0.45  # distance from an obstacle centre that counts as a collision


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(REPO, "deploy", "runs", "r1_distilled"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=700)  # 14 s
    ap.add_argument("--capture", default=None, help="directory for head-camera RGB/depth frames")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    if args.capture:
        os.environ["R1_CAMERAS"] = "1"
        args.enable_cameras = True

    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    from deploy import obs_builder, r1_contract

    contract = r1_contract.load(args.policy)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(contract.policy_path), providers=["CPUExecutionProvider"])

    import gymnasium as gym
    import torch

    from r1_isaaclab.nav_cfg import R1NavEnvCfg  # noqa: F401  (registers the task)

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1-Nav", device=args.device, num_envs=1)
    env_cfg.seed = 23
    obstacles = np.array(OBSTACLES)
    rng = np.random.default_rng(11)

    results = []
    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1-Nav", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        command_term = unwrapped.command_manager.get_term("base_velocity")
        device = unwrapped.device
        index = {name: i for i, name in enumerate(robot.joint_names)}
        order = torch.tensor([index[name] for name in contract.joint_names], device=device, dtype=torch.long)
        dt = unwrapped.step_dt
        builder = obs_builder.ObsBuilder(contract)

        for ep in range(args.episodes):
            env.reset()
            builder.reset(0.0)
            start = robot.data.root_pos_w.torch[0, :2].cpu().numpy()
            for _ in range(200):
                goal = rng.uniform(-ROOM_HALF, ROOM_HALF, 2)
                d_start = np.linalg.norm(goal - start)
                d_obs = np.min(np.linalg.norm(obstacles - goal, axis=1))
                if d_start > 1.2 and d_obs > 0.6:
                    break
            outcome = "timeout"
            steps = 0
            for step in range(args.max_steps):
                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                q = robot.data.root_quat_w.torch[0].cpu().numpy()  # (x, y, z, w)
                yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
                # potential field: attraction to the goal + repulsion from obstacles and walls
                delta = goal - pos[:2]
                dist = float(np.linalg.norm(delta))
                attract = delta / max(dist, 1e-6)
                repel = np.zeros(2)
                for ob in obstacles:
                    d = pos[:2] - ob
                    dn = float(np.linalg.norm(d))
                    if dn < 1.0 and dn > 1e-6:
                        repel += (d / dn) * (0.6 / max(dn - OBS_CLEAR, 0.05))
                for axis in (0, 1):
                    for wall in (-ROOM_HALF, ROOM_HALF):
                        d = pos[axis] - wall
                        if abs(d) < 0.9:
                            r = np.zeros(2)
                            r[axis] = np.sign(d) * (0.5 / max(abs(d) - 0.15, 0.05))
                            repel += r
                direction = attract + repel
                desired = np.arctan2(direction[1], direction[0])
                bearing = np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw))
                # walk when roughly facing the direction, slower in tight spots
                crowd = float(np.clip(1.0 - np.linalg.norm(repel) * 2.0, 0.35, 1.0))
                vx = float(min(dist, 0.6)) * crowd if abs(bearing) < 1.2 else 0.0
                wz = float(np.clip(bearing * 2.0, -1.0, 1.0))

                qi = robot.data.joint_pos.torch[0, order].cpu().numpy()
                dqi = robot.data.joint_vel.torch[0, order].cpu().numpy()
                gyro = robot.data.root_ang_vel_b.torch[0].cpu().numpy()
                cmd = np.array([vx, 0.0, wz], dtype=np.float32)
                obs = builder.build(step * dt, qi, dqi, q, gyro, cmd)
                action = sess.run(None, {"obs": obs.reshape(1, -1)})[0][0]
                builder.record_action(action)
                cmd_t = command_term.command.clone()
                cmd_t[:, 0], cmd_t[:, 1], cmd_t[:, 2] = vx, 0.0, wz
                if hasattr(command_term, "vel_command_b"):
                    command_term.vel_command_b[:] = cmd_t
                command_term.command[:] = cmd_t
                env.step(torch.tensor(action, device=device).reshape(1, -1))
                steps += 1

                if args.capture and step % 120 == 0 and "cam_left" in unwrapped.scene.sensors:
                    import numpy as np
                    from PIL import Image

                    for side in ("left", "right"):
                        cam = unwrapped.scene.sensors[f"cam_{side}"]
                        rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
                        depth = cam.data.output["distance_to_image_plane"].torch[0].cpu().numpy().squeeze().astype("float32")
                        os.makedirs(args.capture, exist_ok=True)
                        tag = f"ep{ep + 1:02d}_step{step:03d}_{side}"
                        Image.fromarray(rgb).save(os.path.join(args.capture, tag + "_rgb.png"))
                        vis = (np.clip(1.0 - depth / 3.0, 0.0, 1.0) * 255).astype("uint8")
                        Image.fromarray(vis).save(os.path.join(args.capture, tag + "_depth.png"))

                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                d_obs = np.min(np.linalg.norm(obstacles - pos[:2], axis=1))
                if pos[2] < 0.45:
                    outcome = "fall"
                    break
                if d_obs < OBS_CLEAR or np.abs(pos[:2]).max() > ROOM_HALF - 0.15:
                    outcome = "collision"
                    break
                if np.linalg.norm(goal - pos[:2]) < GOAL_TOL:
                    outcome = "success"
                    break
            results.append((outcome, steps, goal.copy(), start.copy()))
            print(f"[nav] episode {ep + 1}: {outcome} in {steps * dt:.1f} s", flush=True)

        env.close()

    success = [r for r in results if r[0] == "success"]
    print()
    print(f"episodes             {len(results)}")
    print(f"success rate         {100.0 * len(success) / max(1, len(results)):.1f} %")
    if success:
        print(f"time to goal (mean)  {np.mean([r[1] for r in success]) * dt:.2f} s")
    for name in ("fall", "collision", "timeout"):
        count = sum(1 for r in results if r[0] == name)
        if count:
            print(f"{name:20s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

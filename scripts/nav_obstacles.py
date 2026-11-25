"""Obstacle-course demo: boxes spawn in front of the robot and it must go around them.

The robot starts at the origin facing +x, the goal is 6 m ahead, and a barrier of boxes sits on the
direct line. The boxes are real colliders, so the occupancy grid sees them after they spawn; A* then
routes around the barrier and the deployed walking policy follows the path. Runs a few trials and
reports success, time, bumps and falls; `--spawn-at` adds extra boxes for custom courses.

    C:\\isaac-sim\\python.bat scripts\\nav_obstacles.py --trials 4 --viz none
    C:\\isaac-sim\\python.bat scripts\\nav_obstacles.py --trials 100 --viz kit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

from r1_isaaclab.warehouse import Occupancy  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOAL_TOL = 0.5
# barrier on the straight line from origin to the goal: (x, y, size)
BARRIER = ((2.6, 0.15, (0.8, 0.8, 0.8)),)


def spawn_boxes(args):
    """Boxes as kinematic colliders on the stage (so the occupancy map sees them)."""
    from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.shapes import spawn_cuboid

    for i, (x, y, size) in enumerate(list(BARRIER) + list(args.extra_obstacles)):
        spawn_cuboid(
            f"/World/obstacle_{i}",
            CuboidCfg(
                size=size,
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=MassPropertiesCfg(mass=30.0),
                collision_props=CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
            ),
            translation=(x, y, size[2] / 2),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(REPO, "deploy", "runs", "r1_distilled"))
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=1500)  # 30 s
    ap.add_argument("--goal", default="6.0,0.0")
    ap.add_argument("--spawn-at", default=None, help="extra boxes: 'x,y,size;x,y,size'")
    ap.add_argument("--capture", default=None)
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    args.extra_obstacles = []
    if args.spawn_at:
        for item in args.spawn_at.split(";"):
            parts = [float(v) for v in item.split(",")]
            args.extra_obstacles.append((parts[0], parts[1], (parts[2], parts[2], parts[2])))
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

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 31
    goal = np.array([float(v) for v in args.goal.split(",")])

    results = []
    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = unwrapped.device
        index = {name: i for i, name in enumerate(robot.joint_names)}
        order = torch.tensor([index[name] for name in contract.joint_names], device=device, dtype=torch.long)
        command_term = unwrapped.command_manager.get_term("base_velocity")
        dt = unwrapped.step_dt

        spawn_boxes(args)
        occ = Occupancy(bounds=((-4.0, 9.0), (-5.0, 5.0)), inflate_m=0.5)
        n = occ.build_from_stage(root_prim="/World")
        print(f"[obst] map: {n} colliders, free {100.0 * (~occ.occupied).mean():.0f} %", flush=True)

        builder = obs_builder.ObsBuilder(contract)
        for trial in range(args.trials):
            env.reset()
            builder.reset(0.0)
            start = np.zeros(2)
            if occ.astar(start, goal) is None:
                print("[obst] no path", flush=True)
                results.append(("no path", 0, 0))
                continue
            obstacle_xy = np.array([(x, y) for x, y, _ in list(BARRIER) + list(args.extra_obstacles)])

            outcome, steps, bumps = "timeout", 0, 0
            cmd = np.zeros(3, dtype=np.float32)
            inside = False
            for step in range(args.max_steps):
                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                q = robot.data.root_quat_w.torch[0].cpu().numpy()
                yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

                if step % 5 == 0:
                    delta = goal - pos[:2]
                    dist_goal = float(np.linalg.norm(delta))
                    attract = delta / max(dist_goal, 1e-6)
                    repel = np.zeros(2)
                    for ob in obstacle_xy:
                        d = pos[:2] - ob
                        dn = float(np.linalg.norm(d))
                        if dn < 1.2 and dn > 1e-6:
                            repel += (d / dn) * (0.45 / max(dn - 0.45, 0.05))
                    direction = attract + repel
                    desired = np.arctan2(direction[1], direction[0])
                    bearing = np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw))
                    crowd = float(np.clip(1.0 - np.linalg.norm(repel) * 1.5, 0.35, 1.0))
                    if abs(bearing) > 1.0:
                        cmd = np.array([0.0, 0.0, float(np.clip(bearing * 1.5, -0.8, 0.8))], dtype=np.float32)
                    else:
                        vmax = 0.45 if dist_goal > 1.2 else 0.25
                        cmd = np.array([vmax * crowd * max(0.0, float(np.cos(bearing))), 0.0,
                                        float(np.clip(bearing * 1.5, -0.8, 0.8))], dtype=np.float32)

                cmd_t = command_term.command.clone()
                cmd_t[:, 0], cmd_t[:, 1], cmd_t[:, 2] = float(cmd[0]), 0.0, float(cmd[2])
                if hasattr(command_term, "vel_command_b"):
                    command_term.vel_command_b[:] = cmd_t
                command_term.command[:] = cmd_t

                qi = robot.data.joint_pos.torch[0, order].cpu().numpy()
                dqi = robot.data.joint_vel.torch[0, order].cpu().numpy()
                gyro = robot.data.root_ang_vel_b.torch[0].cpu().numpy()
                obs = builder.build(step * dt, qi, dqi, q, gyro, np.asarray(cmd, dtype=np.float32))
                action = sess.run(None, {"obs": obs.reshape(1, -1)})[0][0]
                builder.record_action(action)
                env.step(torch.tensor(action, device=device).reshape(1, -1))
                steps += 1

                if args.capture and step % 120 == 0 and "cam_left" in unwrapped.scene.sensors:
                    from PIL import Image

                    cam = unwrapped.scene.sensors["cam_left"]
                    rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
                    depth = cam.data.output["distance_to_image_plane"].torch[0].cpu().numpy().squeeze().astype("float32")
                    os.makedirs(args.capture, exist_ok=True)
                    Image.fromarray(rgb).save(os.path.join(args.capture, f"t{trial + 1}_s{step:04d}_rgb.png"))
                    vis = (np.clip(1.0 - depth / 4.0, 0.0, 1.0) * 255).astype("uint8")
                    Image.fromarray(vis).save(os.path.join(args.capture, f"t{trial + 1}_s{step:04d}_depth.png"))

                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                if pos[2] < 0.45:
                    outcome = "fall"
                    break
                now_inside = occ.blocked_raw(pos[:2])
                if now_inside and not inside:
                    bumps += 1
                inside = now_inside
                if np.linalg.norm(goal - pos[:2]) < GOAL_TOL:
                    outcome = "success"
                    break
            results.append((outcome, steps, bumps))
            print(f"[obst] trial {trial + 1}: {outcome} in {steps * dt:.1f} s, bumps {bumps}", flush=True)
        env.close()

    success = [r for r in results if r[0] == "success"]
    print()
    print(f"trials               {len(results)}")
    print(f"success rate         {100.0 * len(success) / max(1, len(results)):.1f} %")
    if success:
        print(f"time to goal (mean)  {np.mean([r[1] for r in success]) * dt:.2f} s")
    print(f"bumps                {sum(r[2] for r in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

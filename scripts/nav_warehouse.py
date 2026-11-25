"""Navigate Isaac's forklift warehouse with the deployed walking policy.

The warehouse (24 x 39 m, racks, walls, forklifts) is loaded from Isaac's asset library, its colliders
are rasterised into a 2D occupancy grid, and an A* path is followed by a go-to-goal controller that
drives the deploy directory's walking policy through the robot runner's observation builder.

    C:\\isaac-sim\\python.bat scripts\\nav_warehouse.py --episodes 8 --viz none
    C:\\isaac-sim\\python.bat scripts\\nav_warehouse.py --episodes 30 --viz kit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402
from r1_isaaclab.warehouse import Occupancy, spawn, spawn_conveyors  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOAL_TOL = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(REPO, "deploy", "runs", "r1_distilled"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=3000)  # 60 s
    ap.add_argument("--capture", default=None)
    ap.add_argument("--spawn", default=None, help="spawn x,y (default: nearest free to the origin)")
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

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 19
    rng = np.random.default_rng(7)

    results = []
    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        command_term = unwrapped.command_manager.get_term("base_velocity")
        device = unwrapped.device
        index = {name: i for i, name in enumerate(robot.joint_names)}
        order = torch.tensor([index[name] for name in contract.joint_names], device=device, dtype=torch.long)
        dt = unwrapped.step_dt

        print("[nav] loading warehouse ...", flush=True)
        spawn()
        spawn_conveyors()
        occ = Occupancy(inflate_m=0.55)
        n_colliders = occ.build_from_stage()
        free_ratio = 100.0 * (~occ.occupied).mean()
        print(f"[nav] grid {occ.nx}x{occ.ny} cells, {n_colliders} colliders, free {free_ratio:.0f} %", flush=True)

        spawn_xy = np.array([0.0, -12.0]) if args.spawn is None else np.array([float(v) for v in args.spawn.split(",")])
        spawn_xy = occ.nearest_free(spawn_xy)
        print(f"[nav] spawn {spawn_xy.round(2)}", flush=True)

        builder = obs_builder.ObsBuilder(contract)
        for ep in range(args.episodes):
            env.reset()
            builder.reset(0.0)
            # place the robot at the chosen free cell, facing +x
            state = robot.data.default_root_state.clone()
            state[:, 0] = float(spawn_xy[0])
            state[:, 1] = float(spawn_xy[1])
            state[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
            robot.write_root_pose_to_sim(state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

            mask = occ.reachable_mask(spawn_xy)
            goal, path = None, None
            for _ in range(12):
                candidate = occ.random_free(rng, near=spawn_xy, min_dist=3.0, mask=mask)
                if candidate is None:
                    continue
                candidate_path = occ.astar(spawn_xy, candidate)
                if candidate_path is not None:
                    goal, path = candidate, candidate_path
                    break
            if goal is None:
                goal = np.array([spawn_xy[0] + 4.0, spawn_xy[1]])
            print(f"[nav] episode {ep + 1}: goal {goal.round(2)}, path {'found' if path is not None else 'MISSING'}", flush=True)
            if path is None:
                results.append(("no path", 0))
                continue

            outcome, steps = "timeout", 0
            bumps = 0
            cmd = np.zeros(3, dtype=np.float32)
            hold = 25  # steps a command is held before recomputing
            for step in range(args.max_steps):
                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                q = robot.data.root_quat_w.torch[0].cpu().numpy()
                yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

                if step % hold == 0:
                    # pure pursuit: closest path point, then a lookahead point ~0.6 m along the path
                    d2 = np.sum((path - pos[:2]) ** 2, axis=1)
                    i = int(np.argmin(d2))
                    if np.sqrt(d2[i]) > 0.8:
                        replanned = occ.astar(pos[:2], goal)
                        if replanned is not None:
                            path = replanned
                            i = 0
                    j = i
                    acc = 0.0
                    while j < len(path) - 1 and acc < 0.6:
                        acc += float(np.linalg.norm(path[j + 1] - path[j]))
                        j += 1
                    target = path[j]
                    delta = target - pos[:2]
                    bearing = np.arctan2(delta[1], delta[0]) - yaw
                    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
                    dist_goal = float(np.linalg.norm(goal - pos[:2]))
                    if abs(bearing) > 0.5:
                        # turn in place first; walking while turning is what made it wander
                        cmd = np.array([0.0, 0.0, float(np.clip(bearing * 1.2, -0.8, 0.8))], dtype=np.float32)
                    else:
                        vmax = 0.40 if dist_goal > 1.5 else 0.25
                        cmd = np.array([vmax * max(0.0, float(np.cos(bearing))), 0.0,
                                        float(np.clip(bearing * 1.2, -0.8, 0.8))], dtype=np.float32)
                vx, _, wz = float(cmd[0]), 0.0, float(cmd[2])
                qi = robot.data.joint_pos.torch[0, order].cpu().numpy()
                dqi = robot.data.joint_vel.torch[0, order].cpu().numpy()
                gyro = robot.data.root_ang_vel_b.torch[0].cpu().numpy()
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

                if args.capture and step % 240 == 0 and "cam_left" in unwrapped.scene.sensors:
                    from PIL import Image

                    cam = unwrapped.scene.sensors["cam_left"]
                    rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
                    depth = cam.data.output["distance_to_image_plane"].torch[0].cpu().numpy().squeeze().astype("float32")
                    os.makedirs(args.capture, exist_ok=True)
                    Image.fromarray(rgb).save(os.path.join(args.capture, f"ep{ep + 1:02d}_step{step:04d}_rgb.png"))

                if step % 600 == 0:
                    print(f"    step {step}: dist to goal {np.linalg.norm(goal - pos[:2]):.1f} m, "
                          f"vx {vx:.2f}", flush=True)

                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                if pos[2] < 0.45:
                    outcome = "fall"
                    break
                if occ.blocked_raw(pos[:2]):
                    bumps += 1
                    if bumps == 1:
                        print(f"    bump at {pos[:2].round(1)} (continuing)", flush=True)
                if np.linalg.norm(goal - pos[:2]) < GOAL_TOL:
                    outcome = "success"
                    break
            results.append((outcome, steps, bumps))
            print(f"[nav] episode {ep + 1}: {outcome} in {steps * dt:.1f} s", flush=True)
        env.close()

    success = [r for r in results if r[0] == "success"]
    print()
    print(f"episodes             {len(results)}")
    print(f"success rate         {100.0 * len(success) / max(1, len(results)):.1f} %")
    if success:
        print(f"time to goal (mean)  {np.mean([r[1] for r in success]) * dt:.2f} s")
    print(f"bumps                {sum(r[2] for r in results)}")
    for name in ("fall", "collision", "timeout", "no path"):
        count = sum(1 for r in results if r[0] == name)
        if count:
            print(f"{name:20s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

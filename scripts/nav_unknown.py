"""Navigate around obstacles with NO prior map: the robot maps them with a lidar as it walks.

The boxes are spawned before the environment, but the planner is never told where they are. A
MultiMesh ray-caster (360 degree lidar at 0.35 m) feeds an online occupancy grid that starts empty
(`r1_isaaclab.online_map`), and the controller avoids the cells the robot has actually observed.

    C:\\isaac-sim\\python.bat scripts\\nav_unknown.py --trials 6 --viz none
    C:\\isaac-sim\\python.bat scripts\\nav_unknown.py --trials 100 --viz kit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOAL_TOL = 0.5


def env_with_lidar(args, boxes):
    """Flat task + 360-degree lidar + the obstacles as scene entities (planner is not told)."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import MultiMeshRayCasterCfg
    from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
    from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 37
    for i, (x, y, size) in enumerate(boxes):
        setattr(env_cfg.scene, f"obstacle_{i}", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + f"obstacle_{i}",
            spawn=CuboidCfg(
                size=(size, size, size),
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=MassPropertiesCfg(mass=30.0),
                collision_props=CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, size / 2.0)),
        ))
    env_cfg.scene.lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/.*/obstacle_.*"),
        ],
        pattern_cfg=LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,
        ),
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        max_distance=8.0,
        update_period=0.02,
        debug_vis=False,
    )
    return env_cfg


def sample_boxes(rng):
    """Ground truth only used for the collision metric; the planner never sees it."""
    boxes = []
    for _ in range(2):
        size = float(rng.uniform(0.5, 0.8))
        boxes.append((float(rng.uniform(2.0, 4.0)), float(rng.uniform(-0.9, 0.9)), size))
    return boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(REPO, "deploy", "runs", "r1_distilled"))
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--goal", default="6.0,0.0")
    ap.add_argument("--seed", type=int, default=3)
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()

    from isaaclab.app import launch_simulation

    from deploy import obs_builder, r1_contract

    contract = r1_contract.load(args.policy)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(contract.policy_path), providers=["CPUExecutionProvider"])

    import gymnasium as gym
    import torch

    goal = np.array([float(v) for v in args.goal.split(",")])
    rng = np.random.default_rng(args.seed)
    boxes = sample_boxes(rng)
    env_cfg = env_with_lidar(args, boxes)

    results = []
    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        lidar = unwrapped.scene["lidar"]
        device = unwrapped.device
        index = {name: i for i, name in enumerate(robot.joint_names)}
        order = torch.tensor([index[name] for name in contract.joint_names], device=device, dtype=torch.long)
        command_term = unwrapped.command_manager.get_term("base_velocity")
        dt = unwrapped.step_dt

        from r1_isaaclab.online_map import OnlineOccupancy

        online = OnlineOccupancy(bounds=((-2.0, 9.0), (-5.0, 5.0)))
        builder = obs_builder.ObsBuilder(contract)
        print(f"[unknown] boxes at {[(round(x, 1), round(y, 1)) for x, y, _ in boxes]} (planner does not know them)", flush=True)

        for trial in range(args.trials):
            env.reset()
            builder.reset(0.0)
            online.reset()
            outcome, steps, bumps, detected_at = "timeout", 0, 0, None
            inside = False
            for step in range(args.max_steps):
                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                q = robot.data.root_quat_w.torch[0].cpu().numpy()
                yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

                # ---- perceive: lidar scan -> online map
                hits = lidar.data.ray_hits_w.torch[0].cpu().numpy()
                origin = lidar.data.pos_w.torch[0, :2].cpu().numpy()
                finite = np.isfinite(hits[:, :2]).all(axis=1)
                hits = hits[finite][:, :2]
                near = np.linalg.norm(hits - origin, axis=1) < 7.0
                online.update(origin, hits[near])
                if detected_at is None and online.occupied_count() > 0:
                    detected_at = step

                # ---- act on the map the robot just built
                known = online.obstacles_near(pos[:2], 1.5)
                delta = goal - pos[:2]
                dist_goal = float(np.linalg.norm(delta))
                attract = delta / max(dist_goal, 1e-6)
                repel = np.zeros(2)
                if known:
                    # nearest mapped cell decides the push: summing over every cell makes the force
                    # explode as the robot approaches a face
                    dists = [float(np.linalg.norm(pos[:2] - ob)) for ob in known]
                    nearest = known[int(np.argmin(dists))]
                    d = pos[:2] - nearest
                    dn = max(dists[int(np.argmin(dists))], 1e-6)
                    repel = (d / dn) * (0.8 / max(dn - 0.5, 0.05))
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

                pos = robot.data.root_pos_w.torch[0].cpu().numpy()
                if pos[2] < 0.45:
                    outcome = "fall"
                    break
                # collision metric uses ground truth, not the map
                hit = any(np.linalg.norm(pos[:2] - np.array([b[0], b[1]])) < 0.6 + b[2] / 2 for b in boxes)
                if hit and not inside:
                    bumps += 1
                inside = hit
                if np.linalg.norm(goal - pos[:2]) < GOAL_TOL:
                    outcome = "success"
                    break
            results.append((outcome, steps, bumps))
            print(f"[unknown] trial {trial + 1}: {outcome} in {steps * dt:.1f} s, bumps {bumps}, "
                  f"first detection at step {detected_at}, map known {100 * online.known_fraction():.0f} %", flush=True)
        env.close()

    success = [r for r in results if r[0] == "success"]
    print()
    print(f"trials               {len(results)}")
    print(f"success rate         {100.0 * len(success) / max(1, len(results)):.1f} %")
    if success:
        print(f"time to goal (mean)  {np.mean([r[1] for r in success]) * dt:.2f} s")
    print(f"bumps into a box     {sum(r[2] for r in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

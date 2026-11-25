"""Prior-map change detection: a "splat map" of the environment, then find NEW obstacles.

This is the simulated version of the Gaussian-splat pipeline: a prior 3D map is captured before the
run (the mapping pass), and during operation the robot compares live depth/lidar to that prior map.
Cells that are occupied now but were free in the prior map are *new* obstacles; they go into a novelty
layer that planning uses. The novelty layer is what a 3DGS-backed map would give you after rendering
the splats from the current pose and taking the depth residual.

    C:\\isaac-sim\\python.bat scripts\\nav_novelty.py --viz none
    C:\\isaac-sim\\python.bat scripts\\nav_novelty.py --viz kit --loop

Prior map = 1 box at (3.0, 0.8). New obstacle = 1 box spawned at (2.5, -0.6) AFTER the mapping pass.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

from r1_isaaclab.online_map import FREE, OCCUPIED, OnlineOccupancy  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR_BOX = (3.0, 0.8, 0.6)
NEW_BOX = (2.5, -0.6, 0.7)
# poses the mapping pass is taken from (x, y, yaw)
MAPPING_POSES = [(0.0, 0.0, 0.0), (2.0, -1.5, 0.5), (2.0, 1.5, -0.5), (4.0, 0.0, 3.14)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--loop", action="store_true")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()

    import numpy as np

    from isaaclab.app import launch_simulation
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sensors import MultiMeshRayCasterCfg
    from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
    from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import gymnasium as gym
    import torch

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 41
    # the prior map only knows this box
    setattr(env_cfg.scene, "prior_box", RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/prior_box",
        spawn=CuboidCfg(
            size=(PRIOR_BOX[2], PRIOR_BOX[2], PRIOR_BOX[2]),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=MassPropertiesCfg(mass=30.0),
            collision_props=CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(PRIOR_BOX[0], PRIOR_BOX[1], PRIOR_BOX[2] / 2)),
    ))
    env_cfg.scene.lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/.*/(prior_box|novel_box_.*)"),
        ],
        # tilted down so the rays hit the floor and obstacles (a horizontal 2D scan at 1.1 m sees nothing)
        pattern_cfg=LidarPatternCfg(channels=1, vertical_fov_range=(-10.0, -10.0),
                                    horizontal_fov_range=(-180.0, 180.0), horizontal_res=2.0),
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.35)),
        max_distance=8.0,
        update_period=0.02,
        update_mesh_ids=True,
        debug_vis=False,
    )

    prior = OnlineOccupancy(bounds=((-2.0, 8.0), (-4.0, 4.0)), cell=0.1)
    novelty = OnlineOccupancy(bounds=((-2.0, 8.0), (-4.0, 4.0)), cell=0.1)

    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        lidar = unwrapped.scene["lidar"]
        device = unwrapped.device

        def scan():
            hits = lidar.data.ray_hits_w.torch[0].cpu().numpy()
            origin = lidar.data.pos_w.torch[0, :2].cpu().numpy()
            finite = np.isfinite(hits).all(axis=1)
            hits = hits[finite]
            return origin, hits[np.linalg.norm(hits[:, :2] - origin, axis=1) < 7.0]

        def teleport(x, y, yaw):
            state = robot.data.default_root_state.clone()
            state[:, 0], state[:, 1] = x, y
            state[:, 3:7] = torch.tensor([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], device=device)
            robot.write_root_pose_to_sim(state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            for _ in range(4):
                env.step(torch.zeros((1, 24), device=device))

        env.reset()
        print("[novelty] mapping pass (prior map, no new box)", flush=True)
        for x, y, yaw in MAPPING_POSES:
            teleport(x, y, yaw)
            origin, hits = scan()
            prior.update_hits(origin, hits)
        print(f"[novelty] prior map: {prior.occupied_count()} occupied cells", flush=True)

        # the new obstacle appears only now: a box that is not in the prior map
        from isaaclab.sim.spawners.shapes import spawn_cuboid

        spawn_cuboid(
            "/World/novel_box_0",
            CuboidCfg(
                size=(NEW_BOX[2], NEW_BOX[2], NEW_BOX[2]),
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=MassPropertiesCfg(mass=30.0),
                collision_props=CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
            ),
            translation=(NEW_BOX[0], NEW_BOX[1], NEW_BOX[2] / 2),
        )
        print(f"[novelty] new obstacle spawned at {NEW_BOX[:2]}", flush=True)

        teleport(0.0, 0.0, 0.0)
        detected_at, novel_cells = None, 0
        for step in range(args.max_steps):
            origin, hits = scan()
            before = novelty.occupied_count()
            for hit in hits:
                if hit[2] <= 0.2:
                    continue  # floor, not an obstacle
                cell = prior.to_cell(hit[:2])
                if prior.in_grid(cell) and prior.state[cell] == FREE:
                    novelty.update(origin, hits=np.array([hit[:2]]))  # occupied now, free before -> new
            novel_cells = novelty.occupied_count()
            if detected_at is None and novel_cells > 0:
                detected_at = step
                print(f"[novelty] first novel detection at step {step}, {novel_cells} cells", flush=True)
            if step == 200:
                # ground truth check: are the novel cells where the new box is?
                cells = np.argwhere(novelty.state == OCCUPIED)
                if len(cells):
                    xy = np.array([novelty.to_xy(tuple(c)) for c in cells])
                    near = np.linalg.norm(xy - np.array(NEW_BOX[:2]), axis=1).min()
                    print(f"[novelty] novel cells: {len(xy)}, closest to new box {near:.2f} m", flush=True)
            env.step(torch.zeros((1, 24), device=device))
            if step > 200 and not args.loop:
                break
        print()
        print(f"prior occupied cells  {prior.occupied_count()}")
        print(f"novel cells           {novel_cells}")
        print(f"first detection       step {detected_at}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

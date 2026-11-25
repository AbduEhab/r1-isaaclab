"""Fly a camera over the whole warehouse and capture RGB-D frames for splatting.

No robot in the shots and no command markers: the robot is made invisible (it only drives the camera
mount) and the velocity-command debug visualisation is disabled. The camera rides at ~2.4 m on a
serpentine coverage path through the free floor, looking 20 degrees down, so the frames show the
warehouse itself.

    C:\\isaac-sim\\python.bat scripts\\capture_warehouse_flycam.py --frames-dir C:\\per\\thesis\\runs\\splat_flycam\\frames --max-frames 320
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=r"C:\per\thesis\runs\splat_flycam\frames")
    ap.add_argument("--max-frames", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--height-m", type=float, default=2.45, help="camera height above the floor")
    ap.add_argument("--row-step", type=float, default=2.2, help="serpentine row spacing")
    ap.add_argument("--point-step", type=float, default=1.0, help="frame spacing along a row")
    ap.add_argument("--warehouse", default="warehouse_forklifts")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    os.makedirs(args.frames_dir, exist_ok=True)

    import numpy as np
    from PIL import Image

    from isaaclab.app import launch_simulation
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import gymnasium as gym
    import torch

    from r1_isaaclab.warehouse import ENVS, Occupancy, spawn, spawn_conveyors

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 71
    # no command arrows in the frames
    env_cfg.commands.base_velocity.debug_vis = False
    focal, aperture = 18.0, 20.955
    # camera rides above the (hidden) robot, looking 20 degrees down
    env_cfg.scene.map_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0.0, args.height_m - 0.76),
                                        rot=(0.0, 0.1736, 0.0, 0.9848),  # +20 deg about Y
                                        convention="world"),
        spawn=PinholeCameraCfg(focal_length=focal, horizontal_aperture=aperture, clipping_range=(0.05, 100.0)),
        data_types=["rgb", "distance_to_image_plane"],
        width=args.width,
        height=args.height,
        update_period=0.02,
        update_latest_camera_pose=True,
        debug_vis=False,
    )

    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        cam = unwrapped.scene["map_cam"]
        device = unwrapped.device

        print("[fly] loading warehouse ...", flush=True)
        spawn(ENVS[args.warehouse])
        spawn_conveyors()
        try:
            from isaaclab.sim import DomeLightCfg
            from isaaclab.sim.spawners.lights import spawn_light

            spawn_light("/World/DomeLight", DomeLightCfg(intensity=1800.0))
        except Exception as exc:  # noqa: BLE001
            print(f"[fly] dome light failed: {exc}", flush=True)

        # hide the robot (it only carries the camera)
        try:
            from pxr import UsdGeom

            from isaaclab.sim import SimulationContext

            stage = SimulationContext.instance().stage
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith("/World/envs/env_0/Robot"):
                    UsdGeom.Imageable(prim).MakeInvisible()
            print("[fly] robot hidden", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[fly] could not hide robot: {exc}", flush=True)

        occ = Occupancy(inflate_m=0.4)
        occ.build_from_stage()

        # serpentine coverage: rows across the warehouse, direction alternating
        rows = []
        y = -16.5
        forward = True
        while y <= 18.5:
            xs = [x for x in np.arange(-10.5, 10.6, args.point_step) if occ.free(occ.to_cell(np.array([x, y])))]
            if len(xs) >= 3:
                rows.append((y, xs if forward else xs[::-1]))
                forward = not forward
            y += args.row_step
        poses = []
        for y, xs in rows:
            for x in xs:
                poses.append((np.array([x, y]), 0.0 if xs[1] > xs[0] else np.pi))
            # link rows with a short perpendicular move (keeps the path continuous)
        print(f"[fly] {len(rows)} rows, {len(poses)} candidate poses", flush=True)
        step = max(1, len(poses) // args.max_frames)
        poses = poses[::step][: args.max_frames]
        print(f"[fly] capturing {len(poses)} frames", flush=True)

        env.reset()
        meta = {"frames": [], "width": args.width, "height": args.height}
        for i, (xy, yaw) in enumerate(poses):
            state = robot.data.default_root_state.clone()
            state[:, 0], state[:, 1] = float(xy[0]), float(xy[1])
            state[:, 3:7] = torch.tensor([0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))], device=device)
            robot.write_root_pose_to_sim(state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            for _ in range(2):
                env.step(torch.zeros((1, 24), device=device))

            rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
            depth = cam.data.output["distance_to_image_plane"].torch[0].cpu().numpy().squeeze().astype("float32")
            pos = cam.data.pos_w.torch[0].cpu().numpy()
            quat = cam.data.quat_w_world.torch[0].cpu().numpy()
            intr = cam.data.intrinsic_matrices.torch[0].cpu().numpy()
            name = f"frame_{i:04d}.png"
            Image.fromarray(rgb).save(os.path.join(args.frames_dir, name))
            np.save(os.path.join(args.frames_dir, name.replace(".png", "_depth.npy")), depth.astype(np.float16))
            x, y2, z, w = quat
            R = np.array([
                [1 - 2 * (y2 * y2 + z * z), 2 * (x * y2 - z * w), 2 * (x * z + y2 * w)],
                [2 * (x * y2 + z * w), 1 - 2 * (x * x + z * z), 2 * (y2 * z - x * w)],
                [2 * (x * z - y2 * w), 2 * (y2 * z + x * w), 1 - 2 * (x * x + y2 * y2)],
            ])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R, pos
            meta["frames"].append({"name": name, "c2w": c2w.tolist(), "K": intr.tolist()})
            if (i + 1) % 50 == 0:
                print(f"[fly] {i + 1}/{len(poses)}", flush=True)

        with open(os.path.join(os.path.dirname(args.frames_dir), "fly_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[fly] wrote {len(poses)} frames", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

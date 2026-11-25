"""Capture an image sequence of the warehouse for VGGT-SLAM / DeltaSplat.

Renders monocular frames from a mapping camera at head height along a path through the warehouse,
with per-frame ground-truth pose and intrinsics so the reconstruction can be evaluated.

    C:\\isaac-sim\\python.bat scripts\\capture_warehouse_frames.py --frames-dir C:\\per\\thesis\\data\\vggt_warehouse --num-frames 60
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
    ap.add_argument("--frames-dir", default=r"C:\per\thesis\data\vggt_warehouse")
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
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
    env_cfg.seed = 53
    focal = 18.0
    aperture = 20.955
    env_cfg.scene.map_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.45, 0.0, 0.45),
                                        rot=(0.0, 0.1305, 0.0, 0.9914),  # ~ +15 deg about Y (look down)
                                        convention="world"),
        spawn=PinholeCameraCfg(focal_length=focal, horizontal_aperture=aperture,
                               clipping_range=(0.05, 100.0)),
        data_types=["rgb"],
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
        print("[capture] loading warehouse ...", flush=True)
        spawn(ENVS[args.warehouse])
        spawn_conveyors()
        # the warehouse inherits no interior lighting from the flat task; add a dome light
        try:
            from isaaclab.sim import DomeLightCfg
            from isaaclab.sim.spawners.lights import spawn_light

            spawn_light("/World/DomeLight", DomeLightCfg(intensity=1500.0))
            print("[capture] dome light spawned", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] dome light failed: {exc}", flush=True)
        occ = Occupancy()
        occ.build_from_stage()
        spawn_xy = occ.nearest_free([0.0, 0.0])
        print(f"[capture] spawn {spawn_xy.round(2)}", flush=True)

        # a curved path (arc of ~270 degrees, radius 1.6 m) gives rotation and parallax for SfM
        waypoints = []
        centre = spawn_xy + np.array([2.6, 0.0])
        for k in range(90):
            theta = -np.pi * 1.5 * (k / 89.0)
            xy = centre + 2.6 * np.array([np.cos(theta), np.sin(theta)])
            heading = theta + np.pi / 2.0  # tangent
            hop = occ.nearest_free(xy)
            if hop is None:
                continue
            waypoints.append((hop, heading))
        print(f"[capture] {len(waypoints)} path points", flush=True)

        env.reset()
        meta = {"frames": [], "width": args.width, "height": args.height, "focal": focal, "aperture": aperture}
        step = max(1, len(waypoints) // args.num_frames)
        captured = 0
        for i, (xy, yaw) in enumerate(waypoints):
            if i % step != 0 or captured >= args.num_frames:
                continue
            state = robot.data.default_root_state.clone()
            state[:, 0], state[:, 1] = float(xy[0]), float(xy[1])
            state[:, 3:7] = torch.tensor([0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))], device=device)
            robot.write_root_pose_to_sim(state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
            for _ in range(3):
                env.step(torch.zeros((1, 24), device=device))

            rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
            pos = cam.data.pos_w.torch[0].cpu().numpy()
            quat = cam.data.quat_w_world.torch[0].cpu().numpy()  # x, y, z, w
            intr = cam.data.intrinsic_matrices.torch[0].cpu().numpy()
            name = f"frame_{captured:04d}.png"
            Image.fromarray(rgb).save(os.path.join(args.frames_dir, name))
            x, y, z, w = quat
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R, pos
            meta["frames"].append({"name": name, "c2w": c2w.tolist(), "K": intr.tolist()})
            captured += 1
            if captured % 10 == 0:
                print(f"[capture] {captured} frames", flush=True)

        with open(os.path.join(args.frames_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[capture] wrote {captured} frames to {args.frames_dir}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

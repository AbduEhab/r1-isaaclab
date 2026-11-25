"""Walk a coverage route through the whole warehouse and capture RGB-D frames for splatting.

Route from the collider occupancy grid: free cells on a 2 m grid, connected with A* and visited
nearest-first, so the walkthrough covers the aisles. Each frame is saved with GT camera pose,
intrinsics and metric depth (used to initialise the Gaussian splats).

    C:\\isaac-sim\\python.bat scripts\\capture_warehouse_walkthrough.py --frames-dir C:\\per\\thesis\\runs\\warehouse_splat\\frames --max-frames 240
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
    ap.add_argument("--frames-dir", default=r"C:\per\thesis\runs\warehouse_splat\frames")
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--step-m", type=float, default=0.7, help="camera move per frame")
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
    env_cfg.seed = 61
    focal, aperture = 18.0, 20.955
    env_cfg.scene.map_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.45, 0.0, 0.45),
                                        rot=(0.0, 0.1305, 0.0, 0.9914),  # ~ +15 deg about Y
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
        print("[walk] loading warehouse ...", flush=True)
        spawn(ENVS[args.warehouse])
        spawn_conveyors()
        try:
            from isaaclab.sim import DomeLightCfg
            from isaaclab.sim.spawners.lights import spawn_light

            spawn_light("/World/DomeLight", DomeLightCfg(intensity=1500.0))
        except Exception as exc:  # noqa: BLE001
            print(f"[walk] dome light failed: {exc}", flush=True)
        occ = Occupancy(inflate_m=0.55)
        occ.build_from_stage()
        spawn_xy = occ.nearest_free([0.0, 0.0])
        print(f"[walk] spawn {spawn_xy.round(2)}", flush=True)

        candidates = []
        for x in np.arange(-10.0, 10.1, 2.0):
            for y in np.arange(-16.0, 18.1, 2.0):
                p = np.array([x, y])
                if occ.free(occ.to_cell(p)):
                    candidates.append(p)
        print(f"[walk] {len(candidates)} free candidate waypoints", flush=True)

        route = [spawn_xy.copy()]
        cur = spawn_xy.copy()
        remaining = list(candidates)
        total_len = 0.0
        max_len = args.max_frames * args.step_m
        while remaining and total_len < max_len:
            remaining.sort(key=lambda q: np.linalg.norm(q - cur))
            advanced = False
            for nxt in remaining[:12]:
                path = occ.astar(cur, nxt)
                if path is None or len(path) < 2:
                    continue
                for k in range(1, len(path)):
                    seg = float(np.linalg.norm(path[k] - path[k - 1]))
                    if seg > 1e-6:
                        route.append((path[k], path[k] - path[k - 1]))
                        total_len += seg
                cur = nxt
                idx = int(np.argmin([float(np.linalg.norm(q - nxt)) for q in remaining]))
                remaining.pop(idx)
                advanced = True
                break
            if not advanced:
                remaining.pop(0)
        print(f"[walk] route {total_len:.1f} m", flush=True)

        samples, acc = [], 0.0
        prev = route[0][0] if isinstance(route[0], tuple) else route[0]
        for item in route[1:]:
            p, direction = item if isinstance(item, tuple) else (item, None)
            acc += float(np.linalg.norm(p - prev))
            if acc >= args.step_m:
                d = direction if direction is not None else (p - prev)
                yaw = float(np.arctan2(d[1], d[0]))
                samples.append((p, yaw))
                acc = 0.0
            prev = p
        samples = samples[: args.max_frames]
        print(f"[walk] {len(samples)} capture poses", flush=True)

        env.reset()
        meta = {"frames": [], "width": args.width, "height": args.height}
        for i, (xy, yaw) in enumerate(samples):
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
            x, y, z, w = quat
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R, pos
            meta["frames"].append({"name": name, "c2w": c2w.tolist(), "K": intr.tolist()})
            if (i + 1) % 40 == 0:
                print(f"[walk] {i + 1}/{len(samples)} frames", flush=True)

        with open(os.path.join(os.path.dirname(args.frames_dir), "walk_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[walk] wrote {len(samples)} frames", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

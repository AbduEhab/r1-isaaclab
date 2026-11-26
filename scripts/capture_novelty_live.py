"""Capture live RGB-D in a warehouse that has NEW objects not present in the prior splat map.

The prior map is the flyover splat of the empty warehouse (runs/splat_flycam/final). Here we fly the
same camera rig through the same warehouse but with four static boxes added, record RGB-D + poses, and
record the ground-truth box positions. The novelty pipeline then has to find them by rendering the prior
splat from the live poses and differencing depths.

    C:\\isaac-sim\\python.bat scripts\\capture_novelty_live.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\per\thesis\runs\novelty_proof")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--warehouse", default="warehouse_forklifts")
    ap.add_argument("--empty", action="store_true", help="prior pass: same poses, no new objects")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    args.enable_cameras = True
    frames_dir = os.path.join(args.out, "live")
    os.makedirs(frames_dir, exist_ok=True)

    import numpy as np
    from PIL import Image

    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from isaaclab_tasks.utils import parse_env_cfg

    from r1_isaaclab.warehouse import ENVS, Occupancy, spawn, spawn_conveyors

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 91
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.scene.map_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0.0, 1.69), rot=(0.0, 0.1736, 0.0, 0.9848), convention="world"),
        spawn=PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        data_types=["rgb", "distance_to_image_plane"],
        width=args.width, height=args.height, update_period=0.02,
        update_latest_camera_pose=True, debug_vis=False,
    )

    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        cam = unwrapped.scene["map_cam"]
        device = unwrapped.device

        print("[novelty] loading warehouse ...", flush=True)
        spawn(ENVS[args.warehouse])
        spawn_conveyors()
        try:
            from isaaclab.sim import DomeLightCfg
            from isaaclab.sim.spawners.lights import spawn_light

            spawn_light("/World/DomeLight", DomeLightCfg(intensity=1800.0))
        except Exception as exc:  # noqa: BLE001
            print(f"[novelty] dome light failed: {exc}", flush=True)

        from pxr import Gf, UsdGeom

        from isaaclab.sim import SimulationContext

        stage = SimulationContext.instance().stage
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/envs/env_0/Robot"):
                UsdGeom.Imageable(prim).MakeInvisible()
        print("[novelty] robot hidden", flush=True)

        occ = Occupancy(inflate_m=0.4)
        occ.build_from_stage()

        # find a ~10 x 5 m free rectangle to host the new boxes
        centre = None
        for cx in np.arange(-6.0, 6.1, 0.5):
            for cy in np.arange(-10.0, 14.1, 0.5):
                pts = [(cx + dx, cy + dy) for dx in (-5.0, -2.5, 0.0, 2.5, 5.0) for dy in (-2.2, 0.0, 2.2)]
                if all(not occ.blocked_raw(np.array(p)) for p in pts):
                    centre = np.array([cx, cy])
                    break
            if centre is not None:
                break
        if centre is None:
            print("[novelty] no free rectangle found", flush=True)
            return 1
        print(f"[novelty] free area centre {centre.round(2)}", flush=True)

        # new objects: (dx, dy, size (x, y, height))
        boxes_spec = [] if args.empty else [(-2.0, 0.9, (0.6, 0.6, 0.8)),
                      (0.5, -0.9, (0.8, 0.8, 1.0)),
                      (2.8, 0.7, (1.0, 0.7, 1.2)),
                      (4.9, -0.6, (0.5, 1.2, 0.7))]
        boxes = []
        for i, (dx, dy, (sx, sy, sz)) in enumerate(boxes_spec):
            p = centre + np.array([dx, dy])
            cube = UsdGeom.Cube.Define(stage, f"/World/NewObject_{i}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(sz / 2)))
            xf.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
            boxes.append({"centre": [float(p[0]), float(p[1]), float(sz / 2)],
                          "size": [float(sx), float(sy), float(sz)]})
            print(f"[novelty] new object {i} at {p.round(2)} size {sx}x{sy}x{sz}", flush=True)

        # two passes along +x and -x through the boxes
        poses = []
        for i in range(36):
            poses.append((centre + np.array([-6.0 + i * 0.34, 0.0]), 0.0))
        for i in range(36):
            poses.append((centre + np.array([6.0 - i * 0.34, 0.0]), np.pi))

        env.reset()
        meta = {"frames": [], "width": args.width, "height": args.height, "boxes": boxes,
                "camera_height": 2.45, "note": "live scene with 4 new boxes; prior map = runs/splat_flycam/final"}
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
            name = f"live_{i:04d}.png"
            Image.fromarray(rgb).save(os.path.join(frames_dir, name))
            np.save(os.path.join(frames_dir, name.replace(".png", "_depth.npy")), depth.astype(np.float16))
            x, y, z, w = quat
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R, pos
            meta["frames"].append({"name": name, "c2w": c2w.tolist(), "K": intr.tolist()})
            if (i + 1) % 20 == 0:
                print(f"[novelty] {i + 1}/{len(poses)}", flush=True)

        with open(os.path.join(args.out, "live_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[novelty] wrote {len(poses)} live frames", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

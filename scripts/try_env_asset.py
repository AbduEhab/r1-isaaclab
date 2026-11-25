"""Spawn one of Isaac's environment USDs in our task and look at it through the head cameras.

    C:\\isaac-sim\\python.bat scripts/try_env_asset.py --usd warehouse_with_forklifts --frames data/env_check --viz none
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Environments/"
ENVS = {
    "warehouse": ROOT + "Simple_Warehouse/warehouse.usd",
    "warehouse_shelves": ROOT + "Simple_Warehouse/warehouse_multiple_shelves.usd",
    "warehouse_forklifts": ROOT + "Simple_Warehouse/warehouse_with_forklifts.usd",
    "warehouse_full": ROOT + "Simple_Warehouse/full_warehouse.usd",
    "digital_twin_small": ROOT + "Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
    "office": ROOT + "Office/office.usd",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default="warehouse_forklifts", help="key of ENVS or a URL")
    ap.add_argument("--frames", default=os.path.join(REPO, "data", "env_check"))
    ap.add_argument("--steps", type=int, default=60)
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    os.environ["R1_CAMERAS"] = "1"
    args.enable_cameras = True
    url = ENVS.get(args.usd, args.usd)

    from isaaclab.app import launch_simulation
    from isaaclab.sim import spawn_from_usd
    from isaaclab.sim.spawners import UsdFileCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device=args.device, num_envs=1)
    env_cfg.seed = 5
    os.makedirs(args.frames, exist_ok=True)

    with launch_simulation(env_cfg, args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        print(f"[env] spawning {url}", flush=True)
        spawn_from_usd("/World/Environment", UsdFileCfg(usd_path=url), translation=(0.0, 0.0, 0.0))

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        prims = list(stage.Traverse())
        colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(stage.GetPrimAtPath("/World/Environment"))
        rng = bbox.ComputeAlignedRange()
        print(f"[env] prims {len(prims)}, colliders {len(colliders)}", flush=True)
        print(f"[env] bbox min {tuple(round(v, 2) for v in rng.GetMin())} max {tuple(round(v, 2) for v in rng.GetMax())}", flush=True)

        env.reset()
        for step in range(args.steps):
            env.step(torch.zeros((1, 24), device=unwrapped.device))
            if step in (10, 55):
                for side in ("left", "right"):
                    cam = unwrapped.scene.sensors[f"cam_{side}"]
                    rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
                    depth = cam.data.output["distance_to_image_plane"].torch[0].cpu().numpy().squeeze().astype("float32")
                    tag = os.path.join(args.frames, f"step{step:03d}_{side}")
                    Image.fromarray(rgb).save(tag + "_rgb.png")
                    vis = (np.clip(1.0 - depth / 5.0, 0.0, 1.0) * 255).astype("uint8")
                    Image.fromarray(vis).save(tag + "_depth.png")
        print(f"[env] frames in {args.frames}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

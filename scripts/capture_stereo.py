"""Capture the R1's head stereo pair from simulation (RGB + metric depth).

Adds the two cameras (``R1_CAMERAS=1``), steps the robot, and writes for each camera a color image,
a depth image (raw .npy plus a visualisation), the intrinsic matrix, and the camera pose, plus a
stereo baseline summary.

    set PYTHONUTF8=1&& set PYTHONIOENCODING=utf-8&& C:\\isaac-sim\\python.bat ^
      scripts\\capture_stereo.py --headless --num_envs 1 --out data\\camera_samples\\demo
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("R1_CAMERAS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402  (registers the task; reads R1_CAMERAS)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Velocity-Flat-R1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default=None)
    add_launcher_args(ap)
    args = ap.parse_args()
    # camera rendering has to be on for headless capture
    args.enable_cameras = True

    env_cfg = parse_env_cfg(TASK, device=getattr(args, "device", "cuda:0"), num_envs=args.num_envs)
    env_cfg.seed = 3

    out = args.out or os.path.join(
        REPO, "data", "camera_samples", time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        env.reset()
        actions = torch.zeros((env.unwrapped.num_envs, 24), device=env.unwrapped.device)
        for _ in range(args.steps):
            env.step(actions)

        meta = {"task": TASK, "num_envs": args.num_envs, "steps": args.steps, "cameras": {}}
        for side in ("left", "right"):
            name = f"cam_{side}"
            cam = env.unwrapped.scene[name]
            rgb = cam.data.output["rgb"].torch[0].cpu().numpy().astype("uint8")
            depth = cam.data.output["distance_to_image_plane"].torch[0].cpu().numpy().squeeze()
            intr = cam.data.intrinsic_matrices.torch[0].cpu().numpy()

            Image.fromarray(rgb).save(os.path.join(out, f"{name}_rgb.png"))
            np.save(os.path.join(out, f"{name}_depth.npy"), depth)
            # depth visualisation: 0..2 m, near = bright
            depth = depth.astype("float32")
            vis = np.clip(1.0 - depth / 2.0, 0.0, 1.0)
            Image.fromarray((vis * 255).astype("uint8")).save(os.path.join(out, f"{name}_depth.png"))

            pos = cam.data.pos_w.torch[0].cpu().numpy()
            quat = cam.data.quat_w_world.torch[0].cpu().numpy()
            meta["cameras"][name] = {
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "intrinsics": intr.tolist(),
                "pos_w": pos.tolist(),
                "quat_w_xyzw": quat.tolist(),
                "depth_min": float(np.nanmin(depth)),
                "depth_max": float(np.nanmax(depth)),
            }
            print(f"[{name}] rgb {rgb.shape}, depth {depth.shape}, depth range "
                  f"{np.nanmin(depth):.2f}..{np.nanmax(depth):.2f} m, fx={intr[0, 0]:.1f}")

        baseline = float(np.linalg.norm(
            np.array(meta["cameras"]["cam_left"]["pos_w"]) - np.array(meta["cameras"]["cam_right"]["pos_w"])
        ))
        meta["stereo_baseline_m"] = baseline
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[capture] stereo baseline {baseline:.3f} m, saved to {out}")

        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




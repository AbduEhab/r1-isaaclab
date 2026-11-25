# VGGT-SLAM 2.0 on the Isaac Sim warehouse

Build: reproducible Docker image (Python 3.11 + torch 2.3.1 + GTSAM develop + SALAD + the MIT-SPARK
VGGT fork), because VGGT-SLAM 2.0 pins those versions and the Windows host has neither 3.11 nor a
compatible GTSAM wheel.

| file | role |
|---|---|
| `tools/vggt-slam/` | clone of MIT-SPARK/VGGT-SLAM (2.0, January 2026) |
| `tools/vggt-slam/Dockerfile.vggt` | python:3.11-slim + torch 2.3.1 cu121 + GTSAM develop wheel + SALAD + VGGT_SPARK |
| `tools/vggt-slam/Dockerfile.vggt-libs` | adds EGL/GL, libusb, libtbb for open3d (headless) |
| `r1-isaaclab/scripts/capture_warehouse_frames.py` | renders a curved 40-frame pass through the forklift warehouse from Isaac Sim, with GT camera poses |
| `r1-isaaclab/scripts/eval_vggt_warehouse.py` | Sim(3)-aligned ATE against the GT poses + point-cloud diagnostics |

## How the pieces fit

```
Isaac Sim warehouse --capture_warehouse_frames--> 40 RGB frames + GT poses (meta.json)
        |
        v
  vggt-slam:2.0 container  python main.py --image_folder /data/vggt_warehouse --log_results
        |
        v
  /out/poses.txt (per-frame camera pose) + /out/poses_points.pcd (dense map)
        |
        v
  eval_vggt_warehouse.py -> ATE RMSE, trajectory plot, top-down map
```

Notes learned while building it:

- the container's Python **must be 3.11**: `gtsam-develop` only publishes cp311/cp312 wheels;
  the official pytorch 2.3.1 image is Python 3.10.
- `gtsam-develop` is not on the corporate mirror; the cp311 wheel is downloaded on the host and
  copied into the image.
- open3d needs `libEGL`, `libusb-1.0`, `libtbb` even when headless.
- SALAD loop-closure retrieval needs `dino_salad.ckpt` (335 MB) in `torch.hub` checkpoints;
  VGGT-1B weights are 4.68 GB from `facebook/VGGT-1B`.
- the capture camera must be **outside the robot mesh** (offset 0.45 m forward) and the warehouse
  needs a dome light; camera poses need `update_latest_camera_pose=True`.

## Results

_(filled in by the run)_

# Splat map: prior 3D map + new-obstacle detection

We have the R1 3DGS pipeline in this repo already:

| artefact | contents |
|---|---|
| `runs/r1/r1_ws/` | COLMAP workspace: 1 487 frames, sparse model, `colmap_traj.tum` |
| `runs/r1/r1_splat_data/scene.npz` | poses (world→camera) 1 487×4×4, `K`, 43 321 sparse points |
| `runs/r1/r1_splat_out/model.npz` | 43 321 Gaussians (`pos`, `quat`, `scale`, `opac`, `color`), trained checkpoints, renders |
| `runs/r1/r1_splat_out/render_trained_*.png` | splat renders |
| `scripts/splat/` | prep, COLMAP→splat data, Taichi trainer, eval renderer |
| `scripts/vslam/` | stereo VO, COLMAP runners, trajectory evaluation |

`scripts/splat_novelty.py` (run with the thesis `.venv`) turns this into a navigation map and a
change-detection pipeline.

## 1. Prior map from the splat

- Filter floaters: keep Gaussians with opacity > 0.3 (9 801 of 43 321); a cell is an obstacle only if
  at least 2 solid Gaussians fall in it.
- Classify by height: points with z < 0.2 m are floor (free), everything else obstacle; inflate
  obstacles by 0.2 m for the robot radius.
- A* on the grid: a **5.22 m route** between two camera poses of the reconstruction.

## 2. Localisation in the splat map

The prior map is only useful with a pose in its frame. That is what the VSLAM/COLMAP pipeline is
for: the same images give COLMAP poses offline or stereo VO / ORB-SLAM / DROID / cuVSLAM poses online.
(For the change-detection demo the pose is taken from the reconstruction itself, which isolates the
perception step.)

## 3. New-obstacle detection by depth differencing

1. **Render the splat's depth** from the current pose: project Gaussian centres with the camera
   intrinsics into a z-buffer (centres are enough for a first version; rasterising ellipsoids would
   be denser).
2. **Fill holes**: the sparse splat leaves depth holes; a 9×9 dilation fills them with the
   background depth so a residual can be computed.
3. **Live depth** comes from the stereo pair (rectified left/right, SGBM) — this is the sensor the R1
   actually has; no lidar anywhere in this chain.
4. **Residual**: pixel is *new geometry* when live depth is nearer than the rendered depth by more
   than 0.25 m. RGB is deliberately not compared: lighting or texture change must not look like an
   obstacle.
5. **Back-project** the residual pixels with the live depth and the pose, add them to a novelty layer
   of the occupancy grid, and re-run A* on prior map + novelty layer (the layer decays so removed
   obstacles go away).

## Measured on the real R1 splat

Simulated new obstacle: a 0.9 × 0.9 m box at 2 m in front of a reconstruction pose.

| metric | value |
|---|---|
| Gaussians / solid | 43 321 / 9 801 |
| A* route on the splat map | 5.22 m |
| novel pixels detected | 2 702 of 3 185 (box silhouette) |
| **recall** | **84.8 %** |
| **precision** | **100 %** |

The novelty mask (`runs/r1/splat_novelty/novelty_mask.png`) is a clean box silhouette; the missed
pixels are where the splat has holes even after filling.

## Caveats that matter for real use

- **Floaters** in an under-trained splat become phantom walls: opacity filtering and the ≥2-Gaussian
  rule are necessary, not optional.
- **Pose accuracy** sets the residual size: a 0.1 m pose error makes a whole plane look "new".
  Localisation must come from VSLAM/COLMAP, not odometry alone.
- **Depth holes** at thin structures and textureless surfaces: fill or reject; un-filled holes show
  as false free space.
- **Transient vs permanent**: a person walking through is a novelty that must decay; the layer needs a
  lifetime.
- Comparing **rendered depth** with **live depth** is robust to lighting; comparing RGB is not.

## Reproduce

```powershell
C:\per\thesis\.venv\Scripts\python.exe C:\per\thesis\r1-isaaclab\scripts\splat_novelty.py
# -> runs/r1/splat_novelty/{prior_map_astar,prior_depth,live_depth,novelty_mask,novelty_replan}.png
#    runs/r1/splat_novelty/summary.json
```

## Next steps

1. Rasterise ellipsoid depth in the Taichi renderer for a dense prior depth (instead of centre
   z-buffer), which should push recall towards 100 %.
2. Replace the simulated live depth with the **sim R1 stereo pair**: spawn a new box, capture the
   stereo images, run SGBM, feed the same differencing code — closes the loop on real perception.
3. Use VSLAM poses instead of reconstruction poses, and fuse the novelty layer into the online map of
   `r1_isaaclab/online_map.py`.

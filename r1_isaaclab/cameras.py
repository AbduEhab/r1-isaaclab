"""Two head cameras on the R1, for visual policies and stereo depth.

The R1's URDF has a camera frame on the head::

    camera_joint: head_yaw_link -> camera_link, xyz (0.067, 0, -0.0216), pitch 0.349 rad

and the head link chain in the imported USD is
``pelvis_link / waist_roll_link / waist_yaw_link / head_pitch_link / head_yaw_link``.
The physical robot carries a stereo pair there, so the simulation models two cameras with a
baseline, each producing RGB and metric depth.

Cameras are only added when ``R1_CAMERAS=1`` so normal training does not pay the rendering cost::

    set R1_CAMERAS=1&& C:\\isaac-sim\\python.bat scripts\\capture_stereo.py --headless --num_envs 1
"""

from __future__ import annotations

import os

from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg


HEAD_CHAIN = "{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_roll_link/waist_yaw_link/head_pitch_link/head_yaw_link"

# camera_link offset in head_yaw_link (URDF), plus the stereo baseline.
_CAM_XYZ = (0.066971, 0.0, -0.021637)
_BASELINE = 0.065
_PITCH = 0.349  # rad, camera looks 20 degrees down as in the URDF
_W, _H = 320, 240

# world convention: forward +X, up +Z -> pitch about +Y by +angle looks down
_HALF = _PITCH / 2.0
_ROT = (0.0, 0.0, 0.0, 0.0)  # filled below


def _pitch_quat(angle: float) -> tuple[float, float, float, float]:
    import math

    return (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0))


def head_cameras(
    baseline: float = _BASELINE,
    width: int = _W,
    height: int = _H,
    update_period: float = 1.0 / 30.0,
) -> dict[str, TiledCameraCfg]:
    """Left/right head cameras: RGB + metric depth (`distance_to_image_plane`)."""
    rot = _pitch_quat(_PITCH)
    cams = {}
    for side, sign in (("left", +1.0), ("right", -1.0)):
        cams[f"cam_{side}"] = TiledCameraCfg(
            prim_path=f"{HEAD_CHAIN}/cam_{side}",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(_CAM_XYZ[0], sign * baseline / 2.0, _CAM_XYZ[2]),
                rot=rot,
                convention="world",
            ),
            data_types=["rgb", "distance_to_image_plane"],
            width=width,
            height=height,
            spawn=PinholeCameraCfg(
                focal_length=18.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
            update_period=update_period,
            debug_vis=False,
        )
    return cams


def add_head_cameras(env_cfg) -> None:
    """Attach the stereo pair to a task config (call from ``__post_init__``)."""
    for name, cfg in head_cameras().items():
        setattr(env_cfg.scene, name, cfg)


def cameras_enabled() -> bool:
    return os.environ.get("R1_CAMERAS", "0") == "1"


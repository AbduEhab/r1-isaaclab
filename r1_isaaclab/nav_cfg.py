"""A room for navigation: static walls and a few obstacles around the flat-ground task.

Everything is a kinematic rigid body, so it collides with the robot but never moves. The room is
5 x 5 m with 0.8 m walls and three 0.3 m boxes; the robot spawns at the centre.
"""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from .env_cfg import R1FlatEnvCfg

ROOM_HALF = 2.5
WALL_H = 0.8
WALL_T = 0.1
OBS_SIZE = 0.3
OBSTACLES = ((0.9, 1.1), (-1.2, -0.6), (0.4, -1.6))


def _static(name: str, size, pos) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=CuboidCfg(
            size=size,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=MassPropertiesCfg(mass=100.0),
            collision_props=CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
            visual_material=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class R1NavEnvCfg(R1FlatEnvCfg):
    """Flat-ground walking with a walled room and three obstacles."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.wall_front = _static("wall_front", (2 * ROOM_HALF, WALL_T, WALL_H), (0.0, ROOM_HALF, WALL_H / 2))
        self.scene.wall_back = _static("wall_back", (2 * ROOM_HALF, WALL_T, WALL_H), (0.0, -ROOM_HALF, WALL_H / 2))
        self.scene.wall_left = _static("wall_left", (WALL_T, 2 * ROOM_HALF, WALL_H), (-ROOM_HALF, 0.0, WALL_H / 2))
        self.scene.wall_right = _static("wall_right", (WALL_T, 2 * ROOM_HALF, WALL_H), (ROOM_HALF, 0.0, WALL_H / 2))
        for i, (x, y) in enumerate(OBSTACLES):
            setattr(self.scene, f"obstacle_{i}", _static(
                f"obstacle_{i}", (OBS_SIZE, OBS_SIZE, 2 * OBS_SIZE), (x, y, OBS_SIZE)))

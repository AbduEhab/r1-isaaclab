"""The R1 as an Isaac Lab asset: the Unitree R1, with Unitree's own R1 numbers.

The USD in ``assets/r1`` is built from the team's authoritative ``urdf/R1.urdf`` by Isaac Sim's
URDF importer (see ``docs/ASSET.md``). It is a 26-revolute-joint humanoid — 12 legs, 2 waist,
10 arms and 2 head joints — with a free base.

The gains, torque limits and armature follow the team's ``POLICY_CHECKLIST.md`` rule 1 (the R1's
table, never the G1's):

    hips, knees, waist         kp 100, kd 2, 60 N m
    ankles                     kp  40, kd 2, 50 N m
    shoulder pitch and roll    kp  40, kd 2, 60 N m
    shoulder yaw, elbows,
    wrist roll                 kp  20, kd 1, 33 N m

Armature 0.01 is the rotor inertia the MJCF carries and a USD import drops; without it the light
wrist and ankle joints ring against their own damping.
"""

from pathlib import Path

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

REPO_ROOT = Path(__file__).resolve().parents[1]
R1_USD = REPO_ROOT / "assets" / "r1" / "R1" / "R1.usda"

# Relative to prim_path: Isaac Lab appends it ("{ENV}/Robot/Geometry/pelvis_link").
ARTICULATION_ROOT = "/Geometry/pelvis_link"
# The variant that carries the Mjc* joint schemas. Isaac Lab's Newton backend builds its MuJoCo
# model from those, so this is the variant a Newton task must select.
PHYSICS_VARIANT = {"Physics": "mujoco"}

LEGS = [".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"]
ANKLES = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
WAIST = ["waist_roll_joint", "waist_yaw_joint"]
SHOULDERS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"]
ARMS = [".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]


def _actuator(names: list[str], kp: float, kd: float, effort: float, velocity: float = 32.0) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=names,
        stiffness=kp,
        damping=kd,
        effort_limit_sim=effort,
        velocity_limit_sim=velocity,
        armature=0.01,
    )


R1_CFG = ArticulationCfg(
    prim_path="/World/R1",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(R1_USD),
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        variants=dict(PHYSICS_VARIANT),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.90)),
    articulation_root_prim_path=ARTICULATION_ROOT,
    actuators={
        "legs": _actuator(LEGS, 100.0, 2.0, 60.0),
        "ankles": _actuator(ANKLES, 40.0, 2.0, 50.0),
        "waist": _actuator(WAIST, 100.0, 2.0, 60.0),
        "shoulders": _actuator(SHOULDERS, 40.0, 2.0, 60.0),
        "arms": _actuator(ARMS, 20.0, 1.0, 33.0),
        # The head stays out of the policy's action space; its own PD holds the default pose.
        # Pitch 50/2 at the URDF's 33 N m, yaw 10/0.1 at the 0.86 N m / 11 rad/s the URDF gives it
        # (POLICY_CHECKLIST.md, trap 1: older copies list 33 N m for the yaw, which it does not have).
        "head_pitch": _actuator(["head_pitch_joint"], 50.0, 2.0, 33.0),
        "head_yaw": _actuator(["head_yaw_joint"], 10.0, 0.1, 0.86, velocity=11.0),
    },
)

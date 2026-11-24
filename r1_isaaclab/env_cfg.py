"""Environment configuration for the R1 velocity-tracking task.

Mirrors Isaac Lab's stock G1 velocity task (isaaclab_tasks/core/velocity/config/g1), so the reward
terms, events and curriculum are the ones NVIDIA trains with, but with:

* the R1 asset (rebuilt from the team's ``urdf/R1.urdf``, see ``docs/ASSET.md``),
* the R1's body and joint names,
* the R1's gains and limits from the team's POLICY_CHECKLIST.md (see ``asset.py``),
* the per-joint action scale ``0.25 * torque_limit / kp`` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 0.15 rad per unit for hips, knees and
  waist, 0.31 for ankles ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â instead of one value for every joint.

Only the 12 leg joints and the 2 waist joints are actuated (action shape 14). The arms and head
hold their default pose through their own actuators, as they do on the robot while it walks.

The physics backend is Isaac Lab's Newton/MuJoCo-Warp preset, which is what the stock velocity
tasks use and what the URDF-rebuilt USD is imported for (its ``mujoco`` variant carries the Mjc
joint schemas Newton reads).
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from . import mdp as r1_mdp
from .asset import R1_CFG

LEG_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_roll_joint", "waist_yaw_joint"]
SHOULDER_JOINTS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"]
ARM_JOINTS = [".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"]
# Unitree's R1 acts all 24 non-head joints; the head is held by its own PD.
ACT_JOINTS = LEG_JOINTS + WAIST_JOINTS + SHOULDER_JOINTS + ARM_JOINTS
FEET = ".*_ankle_roll_link"
# The URDF import names the root body "pelvis_link" (the MuJoCo-converted asset calls it "pelvis").
BASE = "pelvis_link"

# 0.25 * torque_limit / kp per joint (POLICY_CHECKLIST.md rule 1), not one scale for all.
_SCALE_HIP_KNEE_WAIST = 0.25 * 60.0 / 100.0  # 0.15
_SCALE_ANKLE = 0.25 * 50.0 / 40.0  # 0.3125
_SCALE_SHOULDER = 0.25 * 60.0 / 40.0  # 0.375
_SCALE_ARM = 0.25 * 33.0 / 20.0  # 0.4125
ACTION_SCALE = {}
for _side in ("left", "right"):
    for _j in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
        ACTION_SCALE[f"{_side}_{_j}_joint"] = _SCALE_HIP_KNEE_WAIST
    for _j in ("ankle_pitch", "ankle_roll"):
        ACTION_SCALE[f"{_side}_{_j}_joint"] = _SCALE_ANKLE
for _j in WAIST_JOINTS:
    ACTION_SCALE[_j] = _SCALE_HIP_KNEE_WAIST
for _side in ("left", "right"):
    for _j in ("shoulder_pitch", "shoulder_roll"):
        ACTION_SCALE[f"{_side}_{_j}_joint"] = _SCALE_SHOULDER
    for _j in ("shoulder_yaw", "elbow", "wrist_roll"):
        ACTION_SCALE[f"{_side}_{_j}_joint"] = _SCALE_ARM


@configclass
class R1Rewards(RewardsCfg):
    """Reward terms for the MDP, with R1 names."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Hip roll only: penalising hip *yaw* deviation discourages the very joint that turns the robot,
    # which teaches it to strafe instead of rotating. G1's stock config includes hip yaw; the R1's
    # narrower stance cannot afford that shortcut, so yaw stays free here.
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    # Self-collision penalty: with self-collision enabled, contacts between the legs are counted
    # here. Without this term the cheapest way to place the feet is to let the shins overlap.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_hip_.*", ".*_knee_link", ".*_ankle_pitch_link"],
            ),
            "threshold": 1.0,
        },
    )
    single_support_time_penalty = RewTerm(
        func=r1_mdp.single_support_time_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
    )
    double_support_bonus = RewTerm(
        func=r1_mdp.double_support_bonus,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
    )
    swing_time_penalty = RewTerm(
        func=r1_mdp.swing_time_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
    )
    # Gait shaping. The velocity reward alone is satisfied by a one-leg hop: the flight phase is
    # free, no contact pattern is required, and the swing foot only has to move forward, not up.
    contact_schedule = RewTerm(
        func=r1_mdp.contact_schedule,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "period": 0.6,
        },
    )
    flight_phase_penalty = RewTerm(
        func=r1_mdp.flight_phase_penalty,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
    )
    swing_clearance = RewTerm(
        func=r1_mdp.swing_clearance,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    # Anti-strafe term: pays nothing while moving straight, penalises sideways speed during a turn.
    lateral_vel_when_turning = RewTerm(
        func=r1_mdp.lateral_velocity_when_turning,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS)},
    )


def _r1_robot_cfg():
    """The R1 asset, pointed at the scene's env namespace."""
    return replace(R1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class R1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """R1 velocity tracking on rough terrain."""

    rewards: R1Rewards = R1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # physics: the stock Newton/MuJoCo-Warp backend, sized for a 26-DOF humanoid
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        # scene
        self.scene.robot = _r1_robot_cfg()
        # our USD keeps the links under a Geometry Xform, so the stock "[^/]*" pattern would only
        # match that Xform instead of the bodies
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/{BASE}"
        # actions: legs and waist only
        self.actions.joint_pos.joint_names = ACT_JOINTS
        self.actions.joint_pos.scale = dict(ACTION_SCALE)
        # observations: Unitree's R1 contract has a gait clock and no base linear velocity (the real
        # robot has no sensor for it), and looks at the 24 actuated joints rather than all 26.
        # Ablation v3: keep base linear velocity. v2 dropped it to match Unitree's contract, and the
        # policy then never stepped (feet_air_time 0.0009) - standing is the safe optimum when the
        # policy cannot estimate its own speed. For deployment it needs a state estimator, or the
        # term is removed again once the gait is learned.
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        self.observations.policy.gait_phase = ObsTerm(func=r1_mdp.gait_phase, params={"period": 0.6})
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACT_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACT_JOINTS)
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.6)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # The base velocity task ships with heading_command=True: the yaw "command" is then
        # 0.5 * heading error, recomputed every step, not a rate the policy must follow. Stock G1
        # flat turns it off; we inherited it, which is why the policy never learned a real turn.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        # Wider yaw range: commands now ask for real turns (up to 1.5 rad/s, ~86 deg/s).
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        # rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        # terminations and events on the R1's base link
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


@configclass
class R1FlatEnvCfg(R1RoughEnvCfg):
    """R1 velocity tracking on a flat plane: the task to train first."""

    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 300
        # 12 per env x 2048 = 24 576 total, below the ~34 000 contacts the R1 makes: contacts were dropped.
        # Unitree's R1 uses 48 (rough) / auto (flat); 48 leaves headroom without a large allocation.
        newton_mjwarp.solver_cfg.nconmax = 48
        self.sim.physics.default = newton_mjwarp
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # commands: forward speed sweep for this run, so one gait covers walking speeds
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # The base velocity task ships with heading_command=True: the yaw "command" is then
        # 0.5 * heading error, recomputed every step, not a rate the policy must follow. Stock G1
        # flat turns it off; we inherited it, which is why the policy never learned a real turn.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        # Wider yaw range: commands now ask for real turns (up to 1.5 rad/s, ~86 deg/s).
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        # rewards
        # Turning was under-valued: yaw error now costs twice as much as linear error, and the
        # lateral-velocity penalty above removes the strafe-instead-of-turn shortcut.
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        # vertical motion penalty set below (-1.0) as part of the anti-hop shaping
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # Was 0.75: paying for air time is part of why the policy hopped.
        self.rewards.feet_air_time.weight = 0.2
        # Gait shaping, the anti-hop fix.
        self.rewards.contact_schedule.weight = 1.5
        self.rewards.flight_phase_penalty.weight = -2.0
        self.rewards.swing_clearance.weight = 1.0
        # Walk-like support phases and cadence (see mdp.double_support_bonus / swing_time_penalty).
        self.rewards.double_support_bonus.weight = 1.0
        self.rewards.single_support_time_penalty.weight = -1.0
        self.rewards.swing_time_penalty.weight = -1.0
        # Hop = bouncing: make vertical motion expensive.
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        # Strafing is usually skating: punish sliding feet harder so turning is done by stepping.
        self.rewards.feet_slide.weight = -0.25
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # curriculum
        self.curriculum.terrain_levels = None














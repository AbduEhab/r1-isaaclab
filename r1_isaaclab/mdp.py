"""Observation terms Isaac Lab's velocity task does not provide, in Unitree's R1 shape.

`gait_phase` is Unitree's R1 observation (their `deploy.yaml`: `gait_phase: {period: 0.6}`). Isaac
Lab's stock velocity tasks do not have it; without a clock a symmetric biped can satisfy the
velocity reward with a two-feet-together bounce instead of an alternating stride.
"""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gait_phase(env: ManagerBasedRLEnv, period: float = 0.6) -> torch.Tensor:
    """Two-dim sinusoidal gait clock in [cos, sin], one per environment, period in seconds."""
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    omega = 2.0 * math.pi / period
    return torch.stack([torch.cos(omega * t), torch.sin(omega * t)], dim=-1)

def lateral_velocity_when_turning(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_threshold: float = 0.25,
) -> torch.Tensor:
    """Absolute sideways velocity, counted only while a yaw command is active.

    Rewarding yaw tracking alone is not enough: a policy can satisfy a turn command by translating
    sideways, because the linear-velocity term is evaluated in the yaw frame and a strafe keeps the
    body facing the same way. Penalising lateral motion during turns removes that shortcut, and is
    what teaches the robot to rotate on its feet instead of side-stepping.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    turning = (command[:, 2].abs() > yaw_threshold).float()
    return turning * asset.data.root_lin_vel_b[:, 1].abs()


def _feet_airborne(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """(num_envs, num_feet) boolean: foot is off the ground, from the contact sensor's air time."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time.torch[:, sensor_cfg.body_ids] > 0.0


def flight_phase_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1.0 when both feet are in the air. A walk keeps at least one foot down at all times."""
    airborne = _feet_airborne(env, sensor_cfg)
    return airborne.all(dim=1).float()


def contact_schedule(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.6,
    command_name: str = "base_velocity",
    standing_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward the alternating contact pattern implied by the gait clock.

    Left foot is expected in stance during the first half of the cycle, right foot during the second
    half. A foot in contact when expected, or in the air when expected, earns +1; the opposite costs
    -1. This is what turns "one foot in front of the other" into a requirement rather than a lucky
    outcome of the velocity reward.
    """
    airborne = _feet_airborne(env, sensor_cfg)  # True = foot up
    command = env.command_manager.get_command(command_name)
    moving = command[:, :2].norm(dim=1) > standing_threshold
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    phase = torch.remainder(t / period, 1.0)
    left_stance_expected = phase < 0.5
    left_in_contact = ~airborne[:, 0]
    right_in_contact = ~airborne[:, 1]
    left_ok = left_in_contact == left_stance_expected
    right_ok = right_in_contact == (~left_stance_expected)
    match = left_ok.float() + right_ok.float() - 1.0  # -1 .. +1
    return match * moving.float()


def swing_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_height: float = 0.08,
) -> torch.Tensor:
    """Reward lifting the swinging foot: min(foot height, max_height) for airborne feet only.

    Without it the cheapest step is a scuff along the floor; with it the foot has to clear the
    ground, which is what makes the legs look like they are stepping.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    airborne = _feet_airborne(env, sensor_cfg)
    asset = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    clearance = heights.clamp(min=0.0, max=max_height)
    return (airborne.float() * clearance).sum(dim=1)

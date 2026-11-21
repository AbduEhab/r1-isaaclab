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


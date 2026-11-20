"""Observation terms Isaac Lab's velocity task does not provide, in Unitree's R1 shape.

`gait_phase` is Unitree's R1 observation (their `deploy.yaml`: `gait_phase: {period: 0.6}`). Isaac
Lab's stock velocity tasks do not have it; without a clock a symmetric biped can satisfy the
velocity reward with a two-feet-together bounce instead of an alternating stride.
"""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float = 0.6) -> torch.Tensor:
    """Two-dim sinusoidal gait clock in [cos, sin], one per environment, period in seconds."""
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    omega = 2.0 * math.pi / period
    return torch.stack([torch.cos(omega * t), torch.sin(omega * t)], dim=-1)

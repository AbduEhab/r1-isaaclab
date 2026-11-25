"""Build the policy observation from robot state, following deploy.yaml exactly.

History is per observation term, flattened oldest to newest, matching Isaac Lab's observation
history order. The gait clock is free-running from the moment the policy starts, like the episode
clock in training.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import r1_contract


class ObsBuilder:
    def __init__(self, contract: r1_contract.Contract):
        self.contract = contract
        self._hist = {name: deque(maxlen=max(1, hist)) for name, _p, _s, hist in contract.observations}
        self._t0 = None
        self._last_action = np.zeros(contract.n_joints, dtype=np.float32)

    def reset(self, now: float):
        self._t0 = now
        for q in self._hist.values():
            q.clear()

    @property
    def ready(self) -> bool:
        return all(len(q) > 0 for q in self._hist.values())

    def build(
        self,
        now: float,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        quat_wxyz: np.ndarray,
        gyro_body: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        c = self.contract
        if self._t0 is None:
            self.reset(now)
        terms = {
            "base_ang_vel": np.asarray(gyro_body, dtype=np.float32),
            "base_lin_vel": np.zeros(3, dtype=np.float32),  # not provided by the R1; deploy task omits it
            "projected_gravity": r1_contract.projected_gravity(quat_wxyz),
            "velocity_commands": np.asarray(command, dtype=np.float32),
            "joint_pos_rel": np.asarray(joint_pos, dtype=np.float32) - c.default_pos,
            "joint_vel_rel": np.asarray(joint_vel, dtype=np.float32),
            "last_action": self._last_action.copy(),
            "gait_phase": r1_contract.phase_vector(now - self._t0),
        }
        parts = []
        for name, params, scale, hist in c.observations:
            value = terms[name]
            buf = self._hist[name]
            if not buf:
                # first frame: the history buffer starts filled with the current value, as the
                # environment's does at reset
                for _ in range(max(1, hist)):
                    buf.append(value.copy())
            else:
                buf.append(value.copy())
            flat = np.concatenate(list(buf), dtype=np.float32)
            parts.append(flat * scale)
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def record_action(self, actions: np.ndarray):
        self._last_action = np.asarray(actions, dtype=np.float32).reshape(-1).copy()

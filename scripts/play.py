"""Play a trained R1 velocity policy in Isaac Lab 3.0.

    isaaclab.bat -p scripts/play.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 ^
        --checkpoint models/r1_flat_100it/model_99.pt --num_envs 16 --real-time --viz kit

Same registration trick as ``train.py``: importing :mod:`r1_isaaclab` before Isaac Lab's CLI runs
makes the task resolvable. ``--viz kit`` opens the viewport; use ``--viz none`` for headless.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp as wp

wp.config.enable_backward = False

import r1_isaaclab  # noqa: F401,E402  (registers Isaac-Velocity-Flat-R1 / -Rough-R1)

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_play_cli())

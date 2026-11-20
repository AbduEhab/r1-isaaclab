"""Train an R1 velocity policy with Isaac Lab 3.0.

    isaaclab.bat -p scripts/train.py --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 ^
        --viz none --num_envs 4096 --max_iterations 1500

Importing :mod:`r1_isaaclab` before Isaac Lab's CLI runs is what makes
``--task Isaac-Velocity-Flat-R1`` resolvable, without installing anything into Isaac Lab itself.
Warp autodiff is off for the same reason as in Isaac Lab's own ``train.py``: it roughly halves
cold kernel build time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp as wp

wp.config.enable_backward = False

import r1_isaaclab  # noqa: F401,E402  (registers Isaac-Velocity-Flat-R1 / -Rough-R1)

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_train_cli())

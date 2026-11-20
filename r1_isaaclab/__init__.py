"""R1 velocity-tracking tasks for Isaac Lab 3.0.

    import r1_isaaclab   # registers Isaac-Velocity-Flat-R1 and Isaac-Velocity-Rough-R1

Both tasks use the R1 asset in ``assets/r1`` (rebuilt from the team's ``urdf/R1.urdf``) and
Isaac Lab's Newton/MuJoCo-Warp backend.
"""

import gymnasium as gym

from . import agents

__version__ = "0.1.0"

gym.register(
    id="Isaac-Velocity-Flat-R1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:R1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1FlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-R1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:R1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1RoughPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

"""Play a policy in the Isaac Sim window, with Isaac Sim started *before* Isaac Lab is imported.

The local Windows install has two OpenUSD builds in one process: Kit ships its own (omni.usd.libs)
and the pip package usd-exchange brings another. Whichever loads second breaks: if Isaac Lab is
imported first (as scripts/play.py does), Kit's omni.usd extension fails with
"module 'omni.usd' has no attribute 'get_context'". Starting SimulationApp first lets Kit register
its USD, and the pip bindings are loaded afterwards.

    python_bat C:\\per\\thesis\\r1-isaaclab\\scripts\\play_gui.py --checkpoint ... --num_envs 8
"""

import argparse
import os
import sys

import numpy  # noqa: F401  (pip numpy/torch must be in sys.modules before Kit loads its prebundle)
import torch  # noqa: F401

from isaacsim import SimulationApp

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--headless", action="store_true")
_known, _ = _parser.parse_known_args()

app = SimulationApp({"headless": _known.headless})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.enable_backward = False

import r1_isaaclab  # noqa: E402,F401

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from r1_isaaclab.agents.rsl_rl_ppo_cfg import R1FlatPPORunnerCfg  # noqa: E402

TASK = "Isaac-Velocity-Flat-R1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100000)
    args = ap.parse_args()

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs, headless=app.is_running() is False)
    env_cfg.seed = 7
    env = gym.make(TASK, cfg=env_cfg)

    agent_cfg = R1FlatPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wrapped, agent_cfg)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    out = wrapped.reset()
    obs = out[0] if isinstance(out, tuple) else out
    print("[gui] running; close the window or press Ctrl+C to stop", flush=True)
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        out = wrapped.step(actions)
        obs = out[0]
        if not app.is_running():
            break
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


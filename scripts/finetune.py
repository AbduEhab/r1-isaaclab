"""Continue training an existing checkpoint with the current environment.

Used when the physics or the reward set changes in a way that needs the policy to adapt: here, the R1
got self-collision enabled (the legs no longer pass through each other) and an `undesired_contacts`
penalty, so the walking policy has to relearn foot placement without leg overlap.

    isaaclab.sh -p scripts/finetune.py --checkpoint models/r1_flat_4500it/model_4499.pt --iterations 1000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.enable_backward = False

import r1_isaaclab  # noqa: F401,E402

import gymnasium as gym  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from r1_isaaclab.agents.rsl_rl_ppo_cfg import R1FlatPPORunnerCfg  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "Isaac-Velocity-Flat-R1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    run_name = args.run_name or time.strftime("finetune_%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(REPO, "logs", "rsl_rl", "r1_flat", run_name)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 23
    env = gym.make(TASK, cfg=env_cfg)

    agent_cfg = R1FlatPPORunnerCfg()
    agent_cfg.max_iterations = args.iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wrapped, agent_cfg, log_dir=log_dir)
    runner.load(args.checkpoint)
    print(f"[finetune] loaded {args.checkpoint}", flush=True)
    print(f"[finetune] log dir {log_dir}", flush=True)

    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[finetune] saved {os.path.join(log_dir, 'model_final.pt')}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

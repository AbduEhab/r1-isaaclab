"""Diagnose the teacher-action reward: compare the clone's actions with the teacher's on the deploy env."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_actor(path, device):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict")
    model = torch.nn.Sequential(
        torch.nn.Linear(sd["mlp.0.weight"].shape[1], 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, sd["mlp.6.weight"].shape[0]),
    )
    model.load_state_dict({
        "0.weight": sd["mlp.0.weight"], "0.bias": sd["mlp.0.bias"],
        "2.weight": sd["mlp.2.weight"], "2.bias": sd["mlp.2.bias"],
        "4.weight": sd["mlp.4.weight"], "4.bias": sd["mlp.4.bias"],
        "6.weight": sd["mlp.6.weight"], "6.bias": sd["mlp.6.bias"],
    })
    return model.eval().to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher", default=os.path.join(REPO, "models", "r1_flat_walk3_7300it", "model_7300.pt"))
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    lap = argparse.ArgumentParser()
    add_launcher_args(lap)
    launcher_args = lap.parse_args(["--device", "cuda:0"])

    import gymnasium as gym
    import torch

    from r1_isaaclab import mdp as r1_mdp

    task = "Isaac-Velocity-Flat-R1-Deploy-Distill"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=8)

    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(task, cfg=env_cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device
        policy = load_actor(args.checkpoint, device)
        teacher = r1_mdp._load_teacher(args.teacher, device)

        env.reset()
        diffs = []
        for step in range(args.steps):
            obs = unwrapped.observation_manager.compute()["policy"]
            t_obs = r1_mdp._teacher_obs(unwrapped, args.teacher)
            with torch.no_grad():
                t_act = teacher(t_obs)
                actions = policy(obs)
            diffs.append((actions - t_act).abs().mean(dim=1))
            if step == 0:
                print("policy  action[:6]", actions[0, :6].cpu().numpy().round(3))
                print("teacher action[:6]", t_act[0, :6].cpu().numpy().round(3))
                print("command[:3]", unwrapped.command_manager.get_term("base_velocity").command[0].cpu().numpy().round(2))
            env.step(actions)
        d = torch.stack(diffs)
        mean_d = d.mean().item()
        print(f"mean |policy - teacher|: {mean_d:.4f} rad")
        print(f"teacher reward term (1.0 when matched): {1.0 / (1.0 + 50.0 * mean_d ** 2):.3f}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

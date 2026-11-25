"""Collect (deploy observation, teacher action) pairs by rolling the walking teacher out in the sim.

The teacher (``models/r1_flat_walk3_7300it``) observes base linear velocity and its own joint order
(the base task layout). The student must live on the deploy contract: 83 observations per frame and
five frames of history. This script runs the teacher in the base task, builds the deploy observation
(``obs415``) from the same robot state, and records teacher actions as labels.

    C:\\isaac-sim\\python.bat scripts\\collect_teacher.py --teacher models\\r1_flat_walk3_7300it\\model_7300.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_TASK = "Isaac-Velocity-Flat-R1"
HISTORY = 5


def load_actor(path):
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
    model.eval()
    return model


def gravity_body(quat):
    """quat is (x, y, z, w); returns world gravity (0,0,-1) in the body frame."""
    import torch

    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    g = torch.tensor([0.0, 0.0, -1.0], device=quat.device, dtype=quat.dtype)
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - z * w); r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w); r21 = 2 * (y * z + x * w); r22 = 1 - 2 * (x * x + y * y)
    return torch.stack([
        r00 * g[0] + r10 * g[1] + r20 * g[2],
        r01 * g[0] + r11 * g[1] + r21 * g[2],
        r02 * g[0] + r12 * g[1] + r22 * g[2],
    ], dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--num_envs", type=int, default=1024)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--steps_per_iter", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "teacher_rollouts.npz"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    lap = argparse.ArgumentParser()
    add_launcher_args(lap)
    launcher_args = lap.parse_args(["--device", args.device])

    import gymnasium as gym
    import torch

    from r1_isaaclab.env_cfg import ACT_JOINTS

    teacher = load_actor(args.teacher).to(args.device)

    env_cfg = parse_env_cfg(BASE_TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 11
    # the observation noise would leak into the teacher's input and the student's history
    import dataclasses

    from isaaclab.managers import ObservationTermCfg

    for f in dataclasses.fields(type(env_cfg.observations.policy)):
        term = getattr(env_cfg.observations.policy, f.name)
        if isinstance(term, ObservationTermCfg):
            term.noise = None

    obs_list, act_list = [], []
    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(BASE_TASK, cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = unwrapped.device
        num_envs = unwrapped.num_envs
        action_term = unwrapped.action_manager.get_term("joint_pos")
        term_names = list(getattr(action_term, "joint_names", None) or action_term._joint_names)
        art_index = {n: i for i, n in enumerate(robot.joint_names)}
        term_idx = torch.tensor([art_index[n] for n in term_names], device=device, dtype=torch.long)
        art_names = [n for n in robot.joint_names if n in set(term_names)]
        art_idx = torch.tensor([art_index[n] for n in art_names], device=device, dtype=torch.long)
        default_pos = robot.data.default_joint_pos.torch

        command_term = unwrapped.command_manager.get_term("base_velocity")
        dt = unwrapped.step_dt
        actions = torch.zeros((num_envs, len(term_names)), device=device)
        hist = torch.zeros((num_envs, HISTORY, 83), device=device)
        filled = torch.zeros(num_envs, device=device, dtype=torch.long)
        rng = torch.Generator(device="cpu").manual_seed(4)

        env.reset()
        steps = args.iterations * args.steps_per_iter
        for step in range(steps):
            # resample a command for every environment every 2 s
            if step % 100 == 0:
                vx = torch.rand(num_envs, generator=rng) * 0.5 + 0.2
                vy = (torch.rand(num_envs, generator=rng) - 0.5) * 0.8
                wz = (torch.rand(num_envs, generator=rng) - 0.5) * 2.0
                cmd = torch.stack([vx, vy, wz], dim=1).to(device)
                command_term.command[:] = cmd
                if hasattr(command_term, "vel_command_b"):
                    command_term.vel_command_b[:] = cmd

            cmd = command_term.command
            q = robot.data.joint_pos.torch
            dq = robot.data.joint_vel.torch
            quat = robot.data.root_quat_w.torch
            g_b = gravity_body(quat)
            ang_vel = robot.data.root_ang_vel_b.torch
            lin_vel = robot.data.root_lin_vel_b.torch
            t = unwrapped.episode_length_buf.to(dtype=torch.float32) * dt
            phase = torch.stack([torch.cos(2 * 3.141592653589793 * (t % 0.6) / 0.6),
                                 torch.sin(2 * 3.141592653589793 * (t % 0.6) / 0.6)], dim=-1)

            joint_pos_term = q[:, term_idx] - default_pos[:, term_idx]
            joint_vel_term = dq[:, term_idx]
            frame = torch.cat([ang_vel, g_b, cmd, joint_pos_term, joint_vel_term, actions, phase], dim=-1)

            if step > 0:
                hist = torch.cat([hist[:, 1:], frame.unsqueeze(1)], dim=1)
                filled = (filled + 1).clamp(max=HISTORY)
            else:
                hist = frame.unsqueeze(1).repeat(1, HISTORY, 1)
                filled[:] = 1
            obs415 = hist.reshape(num_envs, -1)

            joint_pos_art = q[:, art_idx] - default_pos[:, art_idx]
            joint_vel_art = dq[:, art_idx]
            # teacher layout: [base_lin_vel, base_ang_vel, gravity, command, joint_pos(art), joint_vel(art), actions, phase]
            teacher_obs = torch.cat([lin_vel, ang_vel, g_b, cmd, joint_pos_art, joint_vel_art, actions, phase], dim=-1)
            if step == 6:
                env_obs = unwrapped.observation_manager.compute()["policy"]
                print(f"[collect] teacher obs sanity: max |manual - env| = "
                      f"{(teacher_obs - env_obs).abs().max().item():.5f}", flush=True)
            with torch.no_grad():
                actions = teacher(teacher_obs)

            if step >= HISTORY and step % 2 == 0:
                obs_list.append(obs415.cpu())
                act_list.append(actions.cpu())

            env.step(actions)

            if step % 48 == 0:
                print(f"[collect] step {step}/{steps}: {sum(o.shape[0] for o in obs_list)} samples", flush=True)
        env.close()

    import numpy as np

    obs = torch.cat(obs_list).numpy().astype(np.float32)
    act = torch.cat(act_list).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, obs=obs, act=act)
    print(f"[collect] wrote {args.out}: obs {obs.shape}, act {act.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

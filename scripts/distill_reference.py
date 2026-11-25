"""Distil the reference R1 policy onto our own network, on its single-frame 83-observation contract.

Teacher and student share the exact observation layout here (the reference's: 83 numbers, no history,
no base linear velocity), so DAgger is straightforward: roll a policy, label every step with the
teacher on the same observation, train the student on the aggregate, repeat.

    # collect teacher rollouts
    C:\\isaac-sim\\python.bat scripts\\distill_reference.py collect --out data/ref_teacher.npz

    # behaviour clone + write a deploy directory
    C:\\isaac-sim\\python.bat scripts\\distill_reference.py train --data data/ref_teacher.npz ^
        --out deploy/runs/r1_distilled

    # DAgger round: roll the student, label with the teacher, retrain
    C:\\isaac-sim\\python.bat scripts\\distill_reference.py collect --student deploy/runs/r1_distilled \\ 
        --out data/ref_dagger.npz
    C:\\isaac-sim\\python.bat scripts\\distill_reference.py train --data data/ref_teacher.npz,data/ref_dagger.npz ^
        --out deploy/runs/r1_distilled
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r1_isaaclab  # noqa: F401,E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(REPO, "deploy", "runs", "unitree_reference")


def _onnx(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def collect(args) -> int:
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    lap = argparse.ArgumentParser()
    add_launcher_args(lap)
    launcher_args = lap.parse_args(["--device", "cuda:0", "--viz", "none"])

    import gymnasium as gym
    import numpy as np
    import torch

    from deploy import obs_builder, r1_contract

    contract = r1_contract.load(args.teacher)
    teacher = _onnx(contract.policy_path)
    student = _onnx(os.path.join(args.student, "policy.onnx")) if args.student else None

    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-R1", device="cuda:0", num_envs=args.num_envs)
    env_cfg.seed = 31

    obs_list, act_list = [], []
    with launch_simulation(env_cfg, launcher_args):
        env = gym.make("Isaac-Velocity-Flat-R1", cfg=env_cfg)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        device = unwrapped.device
        index = {n: i for i, n in enumerate(robot.joint_names)}
        order = torch.tensor([index[n] for n in contract.joint_names], device=device, dtype=torch.long)
        command_term = unwrapped.command_manager.get_term("base_velocity")
        builders = [obs_builder.ObsBuilder(contract) for _ in range(args.num_envs)]
        rng = np.random.default_rng(5)

        env.reset()
        for step in range(args.steps):
            if step % 100 == 0:
                cmd = np.stack([
                    rng.uniform(0.2, 0.7, args.num_envs),
                    rng.uniform(-0.4, 0.4, args.num_envs),
                    rng.uniform(-1.0, 1.0, args.num_envs),
                ], axis=1).astype(np.float32)
                cmd_t = torch.tensor(cmd, device=device)
                command_term.command[:] = cmd_t
                if hasattr(command_term, "vel_command_b"):
                    command_term.vel_command_b[:] = cmd_t

            now = step * unwrapped.step_dt
            actions = np.zeros((args.num_envs, contract.n_joints), dtype=np.float32)
            for i in range(args.num_envs):
                q = robot.data.joint_pos.torch[i, order].cpu().numpy()
                dq = robot.data.joint_vel.torch[i, order].cpu().numpy()
                quat = robot.data.root_quat_w.torch[i].cpu().numpy()
                gyro = robot.data.root_ang_vel_b.torch[i].cpu().numpy()
                obs = builders[i].build(now, q, dq, quat, gyro, command_term.command[i].cpu().numpy())
                teacher_action = teacher.run(None, {"obs": obs.reshape(1, -1)})[0][0]
                actions[i] = (student.run(None, {"obs": obs.reshape(1, -1)})[0][0]
                              if student is not None else teacher_action)
                builders[i].record_action(actions[i])
                if student is None:
                    obs_list.append(obs)
                    act_list.append(teacher_action)
                else:
                    obs_list.append(obs)
                    act_list.append(teacher_action)  # DAgger: label the student's state with the teacher
            env.step(torch.tensor(actions, device=device))
            if step % 50 == 0:
                print(f"[collect] step {step}/{args.steps}, {len(obs_list)} samples", flush=True)
        env.close()

    obs = np.asarray(obs_list, dtype=np.float32)
    act = np.asarray(act_list, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, obs=obs, act=act)
    print(f"[collect] wrote {args.out}: obs {obs.shape}, act {act.shape}")
    return 0


def train(args) -> int:
    import numpy as np
    import torch
    import yaml

    parts = [np.load(p) for p in args.data.split(",")]
    obs = torch.from_numpy(np.concatenate([p["obs"] for p in parts])).float()
    act = torch.from_numpy(np.concatenate([p["act"] for p in parts])).float()
    n = obs.shape[0]
    perm = torch.randperm(n)
    val = perm[: max(1, n // 20)]
    train_idx = perm[max(1, n // 20):]
    device = "cuda:0"
    print(f"[train] {n} samples ({obs.shape[1]} -> {act.shape[1]})")

    model = torch.nn.Sequential(
        torch.nn.Linear(obs.shape[1], 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, act.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    tr_obs, tr_act = obs[train_idx].to(device), act[train_idx].to(device)
    va_obs, va_act = obs[val].to(device), act[val].to(device)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(tr_obs.shape[0], device=device)
        total = 0.0
        for start in range(0, order.numel(), 4096):
            idx = order[start: start + 4096]
            opt.zero_grad()
            loss = loss_fn(model(tr_obs[idx]), tr_act[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(va_obs), va_act).item()
        print(f"[train] epoch {epoch + 1}/{args.epochs} train {total / tr_obs.shape[0]:.5f} val {vl:.5f}", flush=True)
    with torch.no_grad():
        rmse = torch.sqrt(loss_fn(model(va_obs), va_act)).item()
    print(f"[train] val action RMSE {rmse:.4f} rad")

    # ---- write the deploy directory: same contract as the reference, our weights
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(REF_DIR, "deploy.yaml")) as f:
        deploy = yaml.safe_load(f)
    deploy["meta"]["source"] = f"distilled from the SpacemiT reference; {n} samples"
    deploy["meta"]["val_action_rmse_rad"] = round(rmse, 5)
    with open(os.path.join(out, "deploy.yaml"), "w") as f:
        yaml.safe_dump(deploy, f, sort_keys=False, default_flow_style=None, width=110)

    model.eval().cpu()
    dummy = torch.zeros(1, obs.shape[1])
    onnx_path = os.path.join(out, "policy.onnx")
    torch.onnx.export(model, dummy, onnx_path, input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    # inline external data
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    proto = onnx.load(onnx_path, load_external_data=False)
    load_external_data_for_model(proto, out)
    onnx.save(proto, onnx_path)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    print(f"[train] wrote {out}/policy.onnx and deploy.yaml")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--teacher", default=REF_DIR)
    c.add_argument("--student", default=None)
    c.add_argument("--num_envs", type=int, default=1024)
    c.add_argument("--steps", type=int, default=192)
    c.add_argument("--out", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default=os.path.join(REPO, "deploy", "runs", "r1_distilled"))
    t.add_argument("--epochs", type=int, default=14)
    t.add_argument("--lr", type=float, default=1e-3)

    args = ap.parse_args()
    return collect(args) if args.mode == "collect" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Behaviour-clone the walking teacher onto the deploy observation contract.

Reads ``data/teacher_rollouts.npz`` (obs 415, act 24) and trains the deploy-contract network
(256-128-128 MLP) to reproduce the teacher's actions. Writes an RSL-RL checkpoint so
``scripts/finetune.py`` can continue with PPO on the deploy task.

    C:\\isaac-sim\\python.bat scripts\\bc_train.py --epochs 15
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "teacher_rollouts.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "deploy", "bc", "model_bc.pt"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import numpy as np
    import torch

    parts = [np.load(p) for p in args.data.split(",")]
    obs = torch.from_numpy(np.concatenate([p["obs"] for p in parts])).float()
    act = torch.from_numpy(np.concatenate([p["act"] for p in parts])).float()
    n = obs.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, n // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    in_dim, out_dim = obs.shape[1], act.shape[1]
    print(f"[bc] {n} samples, {in_dim} -> {out_dim}, val {n_val}")

    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, out_dim),
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    train_obs = obs[train_idx].to(args.device)
    train_act = act[train_idx].to(args.device)
    val_obs = obs[val_idx].to(args.device)
    val_act = act[val_idx].to(args.device)

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train_obs.shape[0], device=args.device)
        total = 0.0
        for start in range(0, order.numel(), args.batch):
            idx = order[start: start + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(train_obs[idx]), train_act[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_obs), val_act).item()
        print(f"[bc] epoch {epoch + 1}/{args.epochs} train mse {total / train_obs.shape[0]:.5f} val mse {val_loss:.5f}",
              flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(val_obs)
    print(f"[bc] val action rmse {torch.sqrt(loss_fn(pred, val_act)).item():.4f} rad")

    # save as an RSL-RL checkpoint: actor = clone, critic = same trunk with a scalar head
    actor_sd = {}
    for i, layer in ((0, 0), (2, 2), (4, 4), (6, 6)):
        actor_sd[f"mlp.{layer}.weight"] = model[i].weight.detach().cpu()
        actor_sd[f"mlp.{layer}.bias"] = model[i].bias.detach().cpu()
    actor_sd["distribution.std_param"] = torch.zeros(out_dim)

    critic = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 1),
    )
    torch.nn.init.zeros_(critic[-1].bias)
    critic_sd = {}
    for i, layer in ((0, 0), (2, 2), (4, 4), (6, 6)):
        critic_sd[f"mlp.{layer}.weight"] = critic[i].weight.detach().cpu()
        critic_sd[f"mlp.{layer}.bias"] = critic[i].bias.detach().cpu()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"actor_state_dict": actor_sd, "critic_state_dict": critic_sd, "iter": 0}, args.out)
    print(f"[bc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

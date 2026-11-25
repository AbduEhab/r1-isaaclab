import os
import numpy as np
import torch

import train_gsplat_warehouse as T

torch.manual_seed(0)
data = T.load_dataset("/data/frames", "/data/walk_meta.json")
print("images", data["images"].shape, "mean", float(data["images"].mean()))
print("viewmat[0]\n", data["viewmats"][0])
print("K\n", data["K"])
print("depth stats", float(data["depths"][data["depths"] < 30].mean()))

d = torch.load("/data/out/state.pt") if os.path.exists("/data/out/state.pt") else None
# rebuild gaussians the same way and render frame 0
means, quats, scales, opacities, colors = T.init_from_depth(data, "cuda")
from gsplat import rasterization
Ks = data["Ks"][:1].cuda()
vm = data["viewmats"][:1].cuda()
renders, alphas, info = rasterization(means, quats, scales, opacities, colors, vm, Ks, data["W"], data["H"],
                                      sh_degree=None, render_mode="RGB")
print("render", renders.shape, float(renders.min()), float(renders.max()), float(renders.mean()))
print("alpha", float(alphas.min()), float(alphas.max()), float(alphas.mean()))
gt = data["images"][0]
print("gt mean", float(gt.mean()))
import cv2
out = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite("/data/out/debug_render.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
g = (gt.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
cv2.imwrite("/data/out/debug_gt.png", cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
print("saved debug images")

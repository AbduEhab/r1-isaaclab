import numpy as np, torch
import train_gsplat_warehouse as T
from gsplat import rasterization
data = T.load_dataset("/data/frames", "/data/walk_meta.json")
means, quats, scales, opac, colors = T.init_from_depth(data, "cuda")
i=0; c2w_cv=T.prim_to_cv(data["c2ws"][i]); w2c=np.linalg.inv(c2w_cv)
K=data["K"]; W,H=data["W"],data["H"]
vm=torch.tensor(w2c,dtype=torch.float32)[None].cuda(); Ks=torch.tensor(K,dtype=torch.float32)[None].cuda()
print("opac stats", float(opac.min()), float(opac.max()), float(opac.mean()), int((~torch.isfinite(opac)).sum()))
for val in (-2.197, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
    o = torch.full_like(opac, val)
    r,a,_ = rasterization(means,quats,scales,o,colors,vm,Ks,W,H,sh_degree=None,render_mode="RGB")
    print(f"opac_logit {val:+.3f} -> render max {float(r.max()):.4f} alpha max {float(a.max()):.4f}")

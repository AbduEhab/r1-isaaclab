import numpy as np, torch
import train_gsplat_warehouse as T
from gsplat import rasterization
data = T.load_dataset("/data/frames", "/data/walk_meta.json")
i = 0
c2w_cv = T.prim_to_cv(data["c2ws"][i])
w2c = np.linalg.inv(c2w_cv)
K = data["K"]; W,H = data["W"], data["H"]
# points placed along the camera's optical axis, 3 m ahead
pts = []
for z in (2.0, 3.0, 4.0):
    p = c2w_cv @ np.array([0.0, 0.0, z, 1.0])
    pts.append(p[:3])
pts = torch.tensor(np.array(pts), dtype=torch.float32, device="cuda")
quats = torch.tensor([[1.0,0,0,0]]*3, device="cuda")
scales = torch.log(torch.full((3,3),0.2,device="cuda"))
opac = torch.tensor([2.0,2.0,2.0], device="cuda")
colors = torch.tensor([[1.0,0,0],[0,1.0,0],[0,0,1.0]], device="cuda")
vm = torch.tensor(w2c, dtype=torch.float32)[None].cuda()
Ks = torch.tensor(K, dtype=torch.float32)[None].cuda()
r,a,_ = rasterization(pts,quats,scales,opac,colors,vm,Ks,W,H,sh_degree=None,render_mode="RGB")
print("ahead-points render max", float(r.max()), "alpha max", float(a.max()))
# now the depth-initialised scene with the same viewmat
means, q2, s2, o2, c2 = T.init_from_depth(data, "cuda")
r2,a2,_ = rasterization(means,q2,s2,o2,c2,vm,Ks,W,H,sh_degree=None,render_mode="RGB")
print("scene render max", float(r2.max()), "alpha max", float(a2.max()))
print("scene means z percentiles", np.percentile(means[:,2].cpu().numpy(), [1,5,50,95]).round(2))

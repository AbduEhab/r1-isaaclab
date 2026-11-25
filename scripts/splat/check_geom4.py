import numpy as np, torch
import train_gsplat_warehouse as T
from gsplat import rasterization
data = T.load_dataset("/data/frames", "/data/walk_meta.json")
means, quats, scales, opac, colors = T.init_from_depth(data, "cuda")
i=0; c2w_cv=T.prim_to_cv(data["c2ws"][i]); w2c=np.linalg.inv(c2w_cv)
K=data["K"]; W,H=data["W"],data["H"]
vm=torch.tensor(w2c,dtype=torch.float32)[None].cuda(); Ks=torch.tensor(K,dtype=torch.float32)[None].cuda()
print("finite means", float(torch.isfinite(means).float().mean()), "scale finite", float(torch.isfinite(scales).float().mean()))
print("means min/max", means.min(0).values.tolist(), means.max(0).values.tolist())
pc = (torch.tensor(w2c[:3,:3],dtype=torch.float32).cuda() @ means.T).T + torch.tensor(w2c[:3,3],dtype=torch.float32).cuda()
front = pc[:,2] > 0.2
u = K[0,0]*pc[:,0]/pc[:,2].clamp(min=1e-6)+K[0,2]; v = K[1,1]*pc[:,1]/pc[:,2].clamp(min=1e-6)+K[1,2]
inside = front & (u>0)&(u<W)&(v>0)&(v<H)
print("front frac", float(front.float().mean()), "inside frac", float(inside.float().mean()))
sel = torch.where(inside)[0]
print("selected", len(sel))
if len(sel):
    m,q,s,o,c = means[sel],quats[sel],scales[sel],opac[sel],colors[sel]
    r,a,_ = rasterization(m,q,s,o,c,vm,Ks,W,H,sh_degree=None,render_mode="RGB")
    print("subset render max", float(r.max()), "alpha", float(a.max()))

import numpy as np, torch
import train_gsplat_warehouse as T
from gsplat import rasterization
data = T.load_dataset("/data/frames", "/data/walk_meta.json")
means, quats, scales, opac, colors = T.init_from_depth(data, "cuda")
i=0; c2w_cv=T.prim_to_cv(data["c2ws"][i]); w2c=np.linalg.inv(c2w_cv)
K=data["K"]; W,H=data["W"],data["H"]
vm=torch.tensor(w2c,dtype=torch.float32)[None].cuda(); Ks=torch.tensor(K,dtype=torch.float32)[None].cuda()
def run(tag, m,q,s,o,c):
    r,a,info = rasterization(m,q,s,o,c,vm,Ks,W,H,sh_degree=None,render_mode="RGB")
    print(tag, "max", round(float(r.max()),3), "alpha", round(float(a.max()),3), "tiles", info.get("tiles_per_gauss") is not None, len(info))
run("all default", means,quats,scales,opac,colors)
run("scales x10", means,quats,scales+np.log(10.0),opac,colors)
run("opac 5", means,quats,scales,torch.full_like(opac,5.0),colors)
run("both", means,quats,scales+np.log(10.0),torch.full_like(opac,5.0),colors)
small = means[:5000], quats[:5000], scales[:5000], opac[:5000], colors[:5000]
run("first 5k", *small)
run("first 5k scales x10", small[0], small[1], small[2]+np.log(10.0), small[3], small[4])

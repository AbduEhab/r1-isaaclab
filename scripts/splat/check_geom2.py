import numpy as np, torch, json
import train_gsplat_warehouse as T
data = T.load_dataset("/data/frames", "/data/walk_meta.json")
means, quats, scales, opacities, colors = T.init_from_depth(data, "cuda")
K = data["K"]; W,H = data["W"], data["H"]
for i in [0, 5]:
    w2c = np.linalg.inv(T.prim_to_cv(data["c2ws"][i]))
    p = means.detach().cpu().numpy()
    pc = (w2c[:3,:3] @ p.T).T + w2c[:3,3]
    z = pc[:,2]; front = z > 0.1
    u = K[0,0]*pc[:,0]/np.maximum(z,1e-6)+K[0,2]; v = K[1,1]*pc[:,1]/np.maximum(z,1e-6)+K[1,2]
    inside = front & (u>=0)&(u<W)&(v>=0)&(v<H)
    print(f"view {i}: front {front.mean():.2f} inside {inside.mean():.2f} z med {np.median(z[inside]) if inside.any() else -1:.2f}")
print("scale exp mean", float(scales.exp().mean()), "opacity", float(opacities.mean()), "means bbox", means.min(0).values.tolist(), means.max(0).values.tolist())

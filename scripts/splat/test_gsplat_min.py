import torch, numpy as np
from gsplat import rasterization
dev="cuda"
means = torch.tensor([[0.0,0.0,3.0],[0.5,0.2,3.0]], device=dev)
quats = torch.tensor([[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]], device=dev)
scales = torch.log(torch.full((2,3),0.3,device=dev))
opac = torch.tensor([2.0,2.0], device=dev)
colors = torch.tensor([[1.0,0.0,0.0],[0.0,1.0,0.0]], device=dev)
viewmats = torch.eye(4, device=dev)[None]
K = torch.tensor([[[100.0,0,32.0],[0,100.0,24.0],[0,0,1.0]]], device=dev)
r,a,info = rasterization(means,quats,scales,opac,colors,viewmats,K,64,48,sh_degree=None,render_mode="RGB")
print("render max", float(r.max()), "alpha max", float(a.max()), "shape", r.shape)

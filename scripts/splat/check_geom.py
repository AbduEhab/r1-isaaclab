import json, numpy as np
meta = json.load(open("/data/walk_meta.json"))
f = meta["frames"][0]
c2w = np.array(f["c2w"]); K = np.array(f["K"])
print("c2w rotation det", np.linalg.det(c2w[:3,:3]))
print("c2w\n", c2w.round(3))
D = np.diag([1.0,-1.0,-1.0,1.0])
c2w_cv = c2w @ D
w2c = np.linalg.inv(c2w_cv)
print("camera centre", c2w_cv[:3,3].round(3), " check", (w2c @ np.append(c2w_cv[:3,3],1))[:3].round(6))
# project a point 3 m straight ahead of the camera (world)
fwd = c2w_cv[:3,2]
p = c2w_cv[:3,3] + 3.0*fwd
p_cam = (w2c @ np.append(p,1))[:3]
print("point ahead -> cam coords", p_cam.round(3))
u = K[0,0]*p_cam[0]/p_cam[2] + K[0,2]; v = K[1,1]*p_cam[1]/p_cam[2] + K[1,2]
print("projects to pixel", round(u,1), round(v,1), "(expect 320,240)")
# depth of the frame's centre pixel: backproject and check world
d = np.load("/data/frames/frame_0000_depth.npy").astype(np.float32)
print("depth centre", float(d[240,320]))

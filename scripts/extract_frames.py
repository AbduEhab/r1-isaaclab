from moviepy.editor import VideoFileClip
import os
src = "/workspace/models/r1_flat_4500it/videos/play/clip_model_4499_0000.mp4"
out = "/workspace/models/r1_flat_4500it/videos/play/frames"
os.makedirs(out, exist_ok=True)
clip = VideoFileClip(src)
print("duration", clip.duration, "fps", clip.fps, "size", clip.size)
for i, t in enumerate([0.4, 1.6, 2.8, 4.0]):
    clip.save_frame(os.path.join(out, f"frame_{i}_{t:.1f}s.png"), t=t)
    print("wrote", t)

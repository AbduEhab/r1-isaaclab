# Navigation in a room (foundation)

A go-to-goal demonstration that drives the **deployed walking policy** (`deploy/runs/r1_distilled`)
around a room with walls and obstacles, using the same observation builder the robot runner uses.

```
scripts/nav_demo.py  ──►  potential-field planner  ──►  velocity command (vx, wz)
                                  │
                                  └─►  deploy/runs/r1_distilled  ──►  joint targets (24)
```

## The room

`r1_isaaclab/nav_cfg.py`: 5 x 5 m, walls 0.8 m high and 0.1 m thick (kinematic rigid bodies), three
0.3 m boxes as obstacles, robot spawns at the centre. Goals are sampled in the room, at least 1.2 m
from the start and 0.6 m from any obstacle.

## The controller

Potential field: attraction to the goal plus repulsion from obstacles within 1 m and walls within
0.9 m, then converted to a walking command (turn towards the field direction, walk, slow down in
tight spots). The planner uses the room's known geometry; perception is not yet in the loop for the
planner itself.

## Measured result (12 episodes, 14 s limit)

| metric | value |
|---|---|
| success rate | **91.7 %** (11/12) |
| mean time to goal | **6.75 s** |
| falls | 0 |
| collisions | 0 |
| timeouts | 1 |

Without the avoidance term the same setup reaches 50 % with 6 collisions (the blind controller walks
into the obstacles), so the avoidance term is what makes the room navigable.

## Cameras

With `--capture <dir>` the head stereo pair is enabled and RGB + metric depth frames are written every
120 steps (`data/nav_frames/...`, gitignored). A frame from the front camera shows the room's wall
corner from the robot's head, which is the visual input a learned navigation policy would consume.

## Limitations and next steps

- The planner uses **ground-truth** obstacle and goal positions. Real deployment needs either a map +
  localisation (the thesis VSLAM stack) or a reactive controller on the **depth** images (the stereo
  pair is already simulated and calibrated).
- The policy is a fixed walking controller; the goal-conditioned *learned* navigation policy (goals +
  depth as observations, trained with the room randomised) is the next step.
- Speed over ground is limited by the walker (0.3-0.7 m/s) and the room is small; larger spaces and
  multiple rooms are future work.

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

## Obstacle course (spawned boxes)

`scripts/nav_obstacles.py` tests avoidance directly: the robot starts at the origin, a goal is 6 m
ahead on the direct line, and a **0.8 m box is spawned 2.6 m ahead**, slightly off the line. Extra
boxes can be added with `--spawn-at "x,y,size;x,y,size"`. The boxes are kinematic colliders spawned on
the stage, so the occupancy map from colliders sees them (1 collider, 84 % free) and collision counting
works.

Controller: potential field (goal attraction + obstacle repulsion, slow-down in the repulsion field),
driving the deployed distilled walker through the robot-runner observation builder.

| trials | success | mean time | bumps into a box | falls |
|---|---|---|---|---|
| 8 | **87.5 % (7/8)** | 20.1 s | 0 | 0 |

The one failure is a timeout (the robot oscillates on the wrong side of the box); it never hits the
box. A barrier of three boxes directly across the line was also tried: the potential field then has a
symmetric local minimum and the robot stalls in front of it, which is why the reported course is a
single offset box. Fixing that needs a wall-following or A*+controller combination (A* paths exist for
the barrier, but the pure-pursuit follower is unreliable in narrow gaps at walking-speed turning).

Reproduce:

```powershell
C:\isaac-sim\python.bat scripts\nav_obstacles.py --trials 8 --max_steps 2000 --viz none
C:\isaac-sim\python.bat scripts\nav_obstacles.py --trials 100 --viz kit          # watch it
C:\isaac-sim\python.bat scripts\nav_obstacles.py --spawn-at "3.5,0.8,0.6;4.5,-0.8,0.6" --viz none
```

## No prior map: where does A* get its map?

A* needs a map. With no prior map the robot must build one from its sensors, or skip global planning:

1. **Online mapping (SLAM without loop closure)** - the robot integrates a lidar or stereo depth into an
   occupancy grid as it moves; A* runs on the live map, replanned every few steps. Unknown cells are
   treated as free (optimistic) so the robot keeps moving towards the goal instead of freezing at the
   frontier.
2. **Local / reactive planning** - potential fields, vector field histogram or dynamic window approach
   plan in the currently observed space only; no global map is needed, but they get stuck in local
   minima (our three-box barrier case).
3. **Learned navigation** - a policy conditioned on depth images + goal vector acts directly from
   perception; no explicit map. This is the robust route in clutter, at the cost of training.
4. **Exploration** - if the goal is far away in unknown space, a frontier-based explorer (or a map
   from the building plan) is needed; A* alone cannot reason about unobserved space.

### Implemented prototype: lidar -> online occupancy -> avoidance

`r1_isaaclab/online_map.py` (empty grid, cells become free/occupied only when a lidar ray crosses
them) and `scripts/nav_unknown.py`: obstacles spawn with the environment, but the planner is never
told where they are; a 360-degree MultiMesh ray-caster at 0.35 m feeds the online map each step and
the controller avoids the cells the robot has actually seen.

Measured (box positions randomised, 40 s limit): the lidar detects the boxes after 38-245 steps and
maps 45-60 % of the scene; the controller avoids some but still collides with boxes (2-4 bump events
per trial) and only 1 of 9 trials reached the goal. So the *mapping pipeline* works; the **local
controller on a partial, noisy map is the weak point**, and this is the honest state of the no-map
prototype. The known-map obstacle course above (87.5 %) is the reliable demo.

Next steps for the no-map case, in order of expected value: replace the potential-field follower with
A* on the online grid using a cost that penalises unknown cells (mildly), replan every 10 steps, and
use the pure-pursuit/waypoint controller with a heading-locked walk; then, if robust, train a
goal-conditioned depth policy for clutter.


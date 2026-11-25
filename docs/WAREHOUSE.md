# Warehouse / shop-floor test bed

## What is available (do we need to build a building?)

Isaac Sim 6.2 ships usable environment builds; there is no need to model a building from scratch.
The asset server's `Assets/Isaac/6.2/Isaac/Environments/` contains:

| Environment | Notes |
|---|---|
| `Simple_Warehouse/warehouse.usd` (1.5 MB) | racks, walls, crates |
| `Simple_Warehouse/warehouse_multiple_shelves.usd` (2.2 MB) | more shelving |
| `Simple_Warehouse/warehouse_with_forklifts.usd` | **used here** — warehouse plus forklifts |
| `Simple_Warehouse/full_warehouse.usd` (6.8 MB) | complete warehouse |
| `Simple_Warehouse/warehouse_with_forklifts_new.usd` | variant |
| `Digital_Twin_Warehouse/small_warehouse_digital_twin.usd` (45 MB) | high-detail digital twin |
| `Hospital/`, `Office/`, `Modular_Warehouse/`, `Simple_Room/` | other buildings |

Props include conveyors (`Props/Conveyors/ConveyorBelt_A01..A15`), forklifts, pallets, packing
tables, Sektion cabinets and factory parts, so a shop floor is assembled from existing USDs.

## What we use

`Simple_Warehouse/warehouse_with_forklifts.usd`, spawned at `/World/Environment` over the flat task:

```
loaded:    3 697 prims, 793 colliders, 0.8 m - 9.3 m height
bounds:    x [-12, 12]  y [-18, 20.8]   (24 m x 39 m building)
contents:  racking, crates on pallets, pallet jacks, forklifts, walls, roll-up doors
extras:    three conveyor belts (Props/Conveyors/ConveyorBelt_A01) in an open aisle
```

Loading downloads the USD once (it is cached under `%TEMP%/isaaclab_usd_*`); later runs load it from
the local copy.

## Colliders to a map

`r1_isaaclab/warehouse.py` rasterises the environment's collider bounding boxes into a 2 cm... 0.2 m
grid, keeping only colliders that intersect the robot's body band (0.15-1.6 m) and ignoring the floor.
The result is an occupancy grid with 57 % free space, inflated by 0.55 m for planning, plus:
- `blocked_raw` for collision tests (robot inside a collider footprint),
- `reachable_mask` (BFS component) so goals are sampled where a path can exist,
- A* over the inflated grid.

## Navigation attempt on the shop floor

`scripts/nav_warehouse.py` follows the A* path with pure pursuit (turn in place, then walk) driving
the deployed distilled walking policy.

| route length | result |
|---|---|
| 5.4 m | **success in 17.6-19.0 s, no bumps** |
| 9-10 m | circles near the goal, then timeout (60 s) |
| 24-29 m | reaches within 4 m of the goal, then drifts / collides; 0 successes |

So the environment, the map, the planner and the robot all work; the **path follower is the weak
link** on long routes inside narrow aisles (the walking policy lags on turning, and replanning churn
makes it wander). This is a controller problem, not an environment problem; the fix is a slower,
heading-locked follow (or a learned goal-conditioned policy), and the test bed is ready for it.

## How to run

```powershell
# look at the warehouse through the robot's head cameras (writes frames + prints prim/collider counts)
C:\isaac-sim\python.bat scripts\try_env_asset.py --usd warehouse_forklifts --frames data\env_check --viz none

# navigate it (open a window to watch)
C:\isaac-sim\python.bat scripts\nav_warehouse.py --episodes 2 --max_steps 2000 --viz kit
```

`--capture <dir>` writes head-camera RGB frames during a run. The window shows the R1 walking among
the racks and forklifts.

## Camera views

Head-camera frames from the test bed (`data/env_check`, `data/warehouse_frames`, both gitignored):
the spawn point faces a yellow forklift, and driving into the aisle puts racks and a pallet jack in
view — exactly the scene a perception stack has to survive.

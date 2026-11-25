"""Isaac's warehouse environment as a navigation test bed.

Isaac Sim's asset library ships usable buildings; the forklift warehouse is 24 x 39 m with 9.3 m
ceilings, 3 697 prims and 793 colliders (racks, walls, forklifts, crates), so it is a real shop-floor
test rather than a box room.

This module spawns it and turns its colliders into a 2D occupancy grid the planner can use.
"""

from __future__ import annotations

import heapq

import numpy as np

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Environments/"
ENVS = {
    "warehouse": ROOT + "Simple_Warehouse/warehouse.usd",
    "warehouse_shelves": ROOT + "Simple_Warehouse/warehouse_multiple_shelves.usd",
    "warehouse_forklifts": ROOT + "Simple_Warehouse/warehouse_with_forklifts.usd",
    "warehouse_full": ROOT + "Simple_Warehouse/full_warehouse.usd",
    "digital_twin_small": ROOT + "Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
    "office": ROOT + "Office/office.usd",
}
WAREHOUSE = ENVS["warehouse_forklifts"]
PRIM = "/World/Environment"


def spawn(url: str = WAREHOUSE, prim_path: str = PRIM, translation=(0.0, 0.0, 0.0)):
    from isaaclab.sim import spawn_from_usd
    from isaaclab.sim.spawners import UsdFileCfg

    spawn_from_usd(prim_path, UsdFileCfg(usd_path=url), translation=translation)


class Occupancy:
    """2D occupancy grid built from the environment's collider bounding boxes.

    A cell is occupied when a collider's AABB overlaps it and its height range intersects the robot's
    body band (walking robots do not go under racks). The floor slab is ignored, and the grid is
    dilated by the robot radius so the planner keeps clearance.
    """

    def __init__(self, cell: float = 0.2, bounds=((-13.0, 13.0), (-19.0, 22.0)),
                 z_band=(0.15, 1.6), inflate_m: float = 0.35):
        self.cell = cell
        self.x0, self.x1 = bounds[0]
        self.y0, self.y1 = bounds[1]
        self.nx = int(round((self.x1 - self.x0) / cell))
        self.ny = int(round((self.y1 - self.y0) / cell))
        self.z_band = z_band
        self.occupied = np.zeros((self.nx, self.ny), dtype=bool)
        self.raw = np.zeros((self.nx, self.ny), dtype=bool)
        self.inflate = max(1, int(round(inflate_m / cell)))

    def build_from_stage(self, root_prim: str = PRIM, exclude: tuple[str, ...] = ("/Robot",)):
        from pxr import Usd, UsdGeom, UsdPhysics

        from isaaclab.sim import SimulationContext

        stage = SimulationContext.instance().stage
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        zlo, zhi = self.z_band
        count = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(root_prim):
                continue
            if any(token in path for token in exclude):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
            if hi[2] < zlo or lo[2] > zhi:
                continue  # floor slab (below the legs) or ceiling
            if (lo[2] <= 0.02) and (hi[2] < 0.12):
                continue  # floor
            ix0 = max(0, int((lo[0] - self.x0) / self.cell))
            ix1 = min(self.nx - 1, int((hi[0] - self.x0) / self.cell))
            iy0 = max(0, int((lo[1] - self.y0) / self.cell))
            iy1 = min(self.ny - 1, int((hi[1] - self.y0) / self.cell))
            if ix0 <= ix1 and iy0 <= iy1:
                self.raw[ix0:ix1 + 1, iy0:iy1 + 1] = True
                self.occupied[ix0:ix1 + 1, iy0:iy1 + 1] = True
                count += 1
        if self.inflate:
            self._dilate(self.inflate)
        return count

    def _dilate(self, radius: int):
        occ = self.occupied
        pad = np.pad(occ, radius, mode="constant", constant_values=True)
        out = np.zeros_like(occ)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    out |= pad[radius + dx: radius + dx + occ.shape[0],
                               radius + dy: radius + dy + occ.shape[1]]
        self.occupied = out

    # ---------------------------------------------------------------- indexing
    def to_cell(self, xy):
        return (int((xy[0] - self.x0) / self.cell), int((xy[1] - self.y0) / self.cell))

    def to_xy(self, cell):
        return np.array([self.x0 + (cell[0] + 0.5) * self.cell, self.y0 + (cell[1] + 0.5) * self.cell])

    def free(self, cell) -> bool:
        ix, iy = cell
        return 0 <= ix < self.nx and 0 <= iy < self.ny and not self.occupied[ix, iy]

    def nearest_free(self, xy, max_radius_m: float = 6.0):
        if self.free(self.to_cell(xy)):
            return np.asarray(xy, dtype=float)
        best, best_d = None, np.inf
        for r in range(1, int(max_radius_m / self.cell)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    cell = (self.to_cell(xy)[0] + dx, self.to_cell(xy)[1] + dy)
                    if self.free(cell):
                        p = self.to_xy(cell)
                        d = np.linalg.norm(p - xy)
                        if d < best_d:
                            best, best_d = p, d
            if best is not None:
                return best
        return None

    def reachable_mask(self, start_xy):
        """Cells connected to the start through free (inflated) cells."""
        mask = np.zeros_like(self.occupied)
        start = self.to_cell(start_xy)
        if not self.free(start):
            return mask
        stack = [start]
        mask[start] = True
        while stack:
            ix, iy = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (ix + dx, iy + dy)
                if self.free(c) and not mask[c]:
                    mask[c] = True
                    stack.append(c)
        return mask

    def blocked_raw(self, xy) -> bool:
        """True when the point is inside a collider footprint (no inflation)."""
        ix, iy = self.to_cell(xy)
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return True
        return bool(self.raw[ix, iy])

    def random_free(self, rng, near=None, min_dist=0.0, mask=None):
        for _ in range(4000):
            cell = (int(rng.integers(0, self.nx)), int(rng.integers(0, self.ny)))
            if not self.free(cell):
                continue
            if mask is not None and not mask[cell]:
                continue
            p = self.to_xy(cell)
            if near is not None and np.linalg.norm(p - near) < min_dist:
                continue
            return p
        return None

    # ---------------------------------------------------------------- planning
    def astar(self, start_xy, goal_xy, max_expand: int = 200000):
        start = self.to_cell(start_xy)
        goal = self.to_cell(goal_xy)
        if not self.free(start) or not self.free(goal):
            return None
        nbrs = [(dx, dy, np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        open_heap = [(0.0, start)]
        came, g = {}, {start: 0.0}

        def h(c):
            return np.hypot(c[0] - goal[0], c[1] - goal[1]) * self.cell

        expanded = 0
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return np.array([self.to_xy(c) for c in reversed(path)])
            expanded += 1
            if expanded > max_expand:
                return None
            for dx, dy, cost in nbrs:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.free(nxt):
                    continue
                ng = g[current] + cost * self.cell
                if ng < g.get(nxt, np.inf):
                    g[nxt] = ng
                    came[nxt] = current
                    heapq.heappush(open_heap, (ng + h(nxt), nxt))
        return None

PROPS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Props/"
CONVEYORS = ((6.0, -14.0, 0.0), (6.0, -11.5, 0.0), (8.6, -14.0, 1.5708))


def spawn_conveyors(prim_root: str = "/World/Shop") -> None:
    """A small conveyor line on the open floor, so the test bed has shop-floor machinery."""
    from isaaclab.sim import spawn_from_usd
    from isaaclab.sim.spawners import UsdFileCfg

    for i, (x, y, yaw) in enumerate(CONVEYORS):
        quat = (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))
        spawn_from_usd(
            f"{prim_root}/conveyor_{i}",
            UsdFileCfg(usd_path=PROPS + "Conveyors/ConveyorBelt_A01.usd"),
            translation=(x, y, 0.0),
            orientation=quat,
        )


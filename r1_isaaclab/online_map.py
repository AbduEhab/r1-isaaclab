"""An online occupancy map: no prior knowledge, cells only exist once a sensor sees them.

The robot starts with an empty grid. Every lidar scan marks the cells between the sensor and each hit
as free and the hit cell as occupied, so the map grows as the robot explores. The controller then
avoids cells that are *known* occupied, and treats unknown cells as free (optimistic) so it keeps
making progress instead of freezing at the frontier.
"""

from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class OnlineOccupancy:
    def __init__(self, bounds=((-2.0, 10.0), (-5.0, 5.0)), cell: float = 0.1):
        self.cell = cell
        self.x0, self.x1 = bounds[0]
        self.y0, self.y1 = bounds[1]
        self.nx = int(round((self.x1 - self.x0) / cell))
        self.ny = int(round((self.y1 - self.y0) / cell))
        self.state = np.zeros((self.nx, self.ny), dtype=np.uint8)  # all UNKNOWN
        self.updates = 0

    def reset(self):
        self.state[:] = UNKNOWN
        self.updates = 0

    # ---------------------------------------------------------------- indexing
    def to_cell(self, xy):
        return (int((xy[0] - self.x0) / self.cell), int((xy[1] - self.y0) / self.cell))

    def in_grid(self, cell) -> bool:
        return 0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny

    def to_xy(self, cell):
        return np.array([self.x0 + (cell[0] + 0.5) * self.cell, self.y0 + (cell[1] + 0.5) * self.cell])

    # ---------------------------------------------------------------- mapping
    def _ray(self, origin_xy, hit_xy):
        """Mark free cells along the ray and the hit cell occupied (integer DDA)."""
        steps = int(np.ceil(np.linalg.norm(hit_xy - origin_xy) / (self.cell * 0.5))) + 1
        for t in np.linspace(0.0, 1.0, steps)[:-1]:
            cell = self.to_cell(origin_xy + t * (hit_xy - origin_xy))
            if self.in_grid(cell) and self.state[cell] != OCCUPIED:
                self.state[cell] = FREE
        hit_cell = self.to_cell(hit_xy)
        if self.in_grid(hit_cell):
            self.state[hit_cell] = OCCUPIED
        # the hit often lands on a face; mark a one-cell neighbourhood so avoidance has bulk
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                c = (hit_cell[0] + dx, hit_cell[1] + dy)
                if self.in_grid(c):
                    self.state[c] = OCCUPIED

    def update_hits(self, origin_xy, hits_xyz: np.ndarray, ground_z: float = 0.2) -> int:
        """Integrate 3D hits: free along every ray, occupied at hits above the floor."""
        before = int((self.state == OCCUPIED).sum())
        for hit in hits_xyz:
            if hit[2] <= ground_z:
                self._mark_free_ray(origin_xy, hit[:2])
                cell = self.to_cell(hit[:2])
                if self.in_grid(cell) and self.state[cell] != OCCUPIED:
                    self.state[cell] = FREE
            else:
                self._ray(origin_xy, hit[:2])
        self.updates += 1
        return int((self.state == OCCUPIED).sum()) - before

    def _mark_free_ray(self, origin_xy, hit_xy):
        steps = int(np.ceil(np.linalg.norm(hit_xy - origin_xy) / (self.cell * 0.5))) + 1
        for t in np.linspace(0.0, 1.0, steps):
            cell = self.to_cell(origin_xy + t * (hit_xy - origin_xy))
            if self.in_grid(cell) and self.state[cell] != OCCUPIED:
                self.state[cell] = FREE

    def update(self, origin_xy, hits_xy: np.ndarray) -> int:
        """Integrate one scan; returns the number of new occupied cells."""
        before = int((self.state == OCCUPIED).sum())
        for hit in hits_xy:
            self._ray(origin_xy, hit)
        self.updates += 1
        return int((self.state == OCCUPIED).sum()) - before

    # ---------------------------------------------------------------- queries
    def obstacles_near(self, xy, radius: float) -> list[np.ndarray]:
        """Centres of known-occupied cells within `radius` of a point."""
        cell = self.to_cell(xy)
        r = int(np.ceil(radius / self.cell))
        out = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                c = (cell[0] + dx, cell[1] + dy)
                if self.in_grid(c) and self.state[c] == OCCUPIED:
                    p = self.to_xy(c)
                    if np.linalg.norm(p - xy) <= radius:
                        out.append(p)
        return out

    def known_occupied(self, xy) -> bool:
        cell = self.to_cell(xy)
        return self.in_grid(cell) and self.state[cell] == OCCUPIED

    def occupied_count(self) -> int:
        return int((self.state == OCCUPIED).sum())

    def known_fraction(self) -> float:
        return float((self.state != UNKNOWN).mean())

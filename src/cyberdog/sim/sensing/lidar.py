"""A virtual Mid-360, so the dog can see what the map does not know.

Spec L2 step 2 asks the twin for a LiDAR raycaster publishing /lidar/points,
and until now the twin had no range sensor at all: every solid in the scene was
also in the occupancy grid, so nothing needed one. `obstacles.py` broke that on
purpose, and this is the sensor that finds them.

Sensor only. It reports where surfaces are and knows nothing about what they
mean -- deciding which returns the map already explains is perception.py's job,
and it is kept separate so the day a real Mid-360 arrives only the producer
changes. The output is shaped like the /lidar/points cloud in the spec's
interface table for the same reason.

Pattern. 360 degrees horizontally -- a real Mid-360 is non-repetitive rather
than a raster, but for finding a crate in a corridor a uniform pattern is
honest and cheaper to reason about -- and -7 to +52 degrees vertically, which
is the unit's own FOV. ~2000 rays comes out at ~1.5 ms a scan (1.3-1.7
depending on how much of the floor is in range), so the cost of carrying it
at every control tick is nothing.

Two things that are not obvious and cost an afternoon each:

- The sensor sits at MOUNT_H above the floor, not at the camera's 0.32 m. At
  lens height the rays start *inside* the Go2's own trunk and every single one
  of them returns the robot. MIN_R catches the rest, the way a real unit's
  minimum range does.
- mj_multiRay takes ONE origin for all rays, and its `normal` argument comes
  before `nray`. Get either wrong and it raises rather than misbehaving, which
  is the good case.
"""
import math

import mujoco
import numpy as np

NH, NV = 180, 11                # 1980 rays, ~1.5 ms a scan on an M-series Mac
EL_MIN, EL_MAX = -7.0, 52.0     # degrees, the Mid-360's vertical FOV
MOUNT_H = 0.50                  # above the floor -- clear of the dog's own back
MIN_R = 0.35                    # minimum range, and the self-hit guard
MAX_R = 12.0                    # beyond this it reports nothing


class Lidar:
    """Rays from a point above the dog, into whatever the scene is made of."""

    def __init__(self, model, data):
        self.model, self.data = model, data

        az = np.linspace(0.0, 2 * math.pi, NH, endpoint=False)
        el = np.radians(np.linspace(EL_MIN, EL_MAX, NV))
        A, E = np.meshgrid(az, el, indexing="ij")
        vec = np.stack([np.cos(E) * np.cos(A),
                        np.cos(E) * np.sin(A),
                        np.sin(E)], axis=-1).reshape(-1, 3)

        # The pattern is a full circle, so the dog's heading does not rotate it
        # and it can be built once.
        self.vec = np.ascontiguousarray(vec, dtype=np.float64)
        self.n = NH * NV
        self._flat = self.vec.flatten()
        self._geomid = np.zeros(self.n, dtype=np.int32)
        self._dist = np.zeros(self.n, dtype=np.float64)

    def scan(self, xy, floor_z):
        """(N, 3) world points, one per ray that hit something in range."""
        origin = np.array([xy[0], xy[1], floor_z + MOUNT_H], dtype=np.float64)
        mujoco.mj_multiRay(self.model, self.data, origin, self._flat,
                           None,          # geomgroup: all of them
                           1,             # flg_static: the building is static
                           -1,            # bodyexclude: MIN_R handles the dog
                           self._geomid, self._dist, None, self.n, MAX_R)
        hit = self._dist > MIN_R
        return origin + self.vec[hit] * self._dist[hit, None]

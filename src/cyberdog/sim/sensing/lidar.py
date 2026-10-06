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

Mounted nose-down, MOUNT_PITCH. The spec plans one Mid-360 for both jobs --
obstacles and the affordance elevation map, one /lidar/points -- and level,
its lowest ring (-7 deg from 0.5 m) meets the floor 4 m out: of the 2 m patch
in front of the dog the elevation map needs, it saw 15% even remembering the
last 3 m of walking. Tilted 20 deg forward it sees 88%, and the top of its
field still clears a standing person's head at 3 m. Tilted, the pattern turns
with the dog, so scan() needs the heading.

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
MOUNT_PITCH = 20.0              # degrees nose-down -- see the docstring


class Lidar:
    """Rays from a point above the dog, into whatever the scene is made of."""

    def __init__(self, model, data, pitch=MOUNT_PITCH):
        self.model, self.data = model, data

        az = np.linspace(0.0, 2 * math.pi, NH, endpoint=False)
        el = np.radians(np.linspace(EL_MIN, EL_MAX, NV))
        A, E = np.meshgrid(az, el, indexing="ij")
        vec = np.stack([np.cos(E) * np.cos(A),
                        np.cos(E) * np.sin(A),
                        np.sin(E)], axis=-1).reshape(-1, 3)

        # Pitched nose-down about the dog's own y axis, once; turned to the
        # dog's heading every scan. Level (pitch 0) it would be a full circle
        # that no heading changes.
        p = math.radians(pitch)
        tilt = np.array([[math.cos(p), 0.0, math.sin(p)],
                         [0.0, 1.0, 0.0],
                         [-math.sin(p), 0.0, math.cos(p)]])
        self.body = np.ascontiguousarray(vec @ tilt.T, dtype=np.float64)
        self.vec = self.body
        self.n = NH * NV
        self._geomid = np.zeros(self.n, dtype=np.int32)
        self._dist = np.zeros(self.n, dtype=np.float64)

    def scan(self, xy, floor_z, yaw=0.0):
        """(N, 3) world points, one per ray that hit something in range."""
        origin = np.array([xy[0], xy[1], floor_z + MOUNT_H], dtype=np.float64)
        c, s = math.cos(yaw), math.sin(yaw)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.vec = np.ascontiguousarray(self.body @ turn.T)
        mujoco.mj_multiRay(self.model, self.data, origin, self.vec.flatten(),
                           None,          # geomgroup: all of them
                           1,             # flg_static: the building is static
                           -1,            # bodyexclude: MIN_R handles the dog
                           self._geomid, self._dist, None, self.n, MAX_R)
        hit = self._dist > MIN_R
        return origin + self.vec[hit] * self._dist[hit, None]

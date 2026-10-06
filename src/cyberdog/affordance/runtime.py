"""The affordance network on a running dog: LiDAR scans in, a judgement out.

Spec L5 step 4: "build local elevation from /lidar/points; score". Two parts.

ElevationMemory keeps the last KEEP_M metres of walking's worth of scans. One
scan is not enough: even with the Mid-360 tilted 20 deg nose-down (lidar.py)
it sees about a sixth of the patch in front of the dog, because the ground
there is too close and too steep below the sensor. The ground the dog is
about to walk on was seen from further back, though, so a few metres of
memory sees ~88% of it. Only the ground band is kept (MAX_H): the patch is
about what the dog walks on, and the ceiling, which Isaac's downward scanner
never sees, otherwise wins every cell -- highest-wins -- and turns a corridor
into a field of 2 m walls.

Affordance asks the network about walking from the dog to a point ahead,
with the 0.2 refusal rule (model.REFUSE_P). It knows nothing of the map; it
judges ground. Pose comes from the twin; on the robot it would be odometry,
and the memory is only as good as that is over 3 m.
"""
import math

import numpy as np

from cyberdog.affordance import data, model as M

KEEP_M = 3.0            # metres of walking whose scans are remembered
MAX_H = 1.0             # metres above the floor: higher is not ground (ceiling, lamps)
NEAR_M = 2.5            # only points this close to the dog are kept -- the patch reaches 1.6 m
MIN_SEEN = 0.7          # under this share of the patch seen, no judgement is made:
                        # under 70% seen it wrongly refused open floor 45% of the time
WALL_M = 0.15           # metres from a mapped wall face that is wall, not floor: a
                        # return on the face sits in a world cell up to RES from it,
                        # looked up from a patch centre up to RES * sqrt(2) away


class ElevationMemory:
    """The ground around the dog: a world-frame grid, each cell its latest look.

    Newer replaces older, cell by cell. The first version kept the last
    KEEP_M of raw scans and took the highest point of all of them, and a
    person walking down the corridor left a trail of their own legs in it:
    on the shadow run 400 of 512 wrong refusals of open floor were on a few
    metres of floor-1 corridor that people had just walked, refused with
    certainty and every cell seen. A cell looked at again now takes what it
    shows now -- floor where the person was, the crate where the crate still
    is. Within one scan a cell keeps its highest return, as data.py does.
    """

    def __init__(self, keep_m=KEEP_M, max_h=MAX_H, res=data.RES):
        self.keep_m, self.max_h, self.res = keep_m, max_h, res
        self.cells = {}                 # (ix, iy) -> [height above floor, odometer]
        self.odometer = 0.0
        self.last_xy = None
        self.floor_z = None

    def add(self, points, xy, floor_z):
        """Fold one scan in, taken with the dog at `xy` on a floor at `floor_z`."""
        if self.floor_z is None or abs(floor_z - self.floor_z) > 0.5:
            self.cells, self.odometer, self.last_xy = {}, 0.0, None     # another floor
        self.floor_z = floor_z
        if self.last_xy is not None:
            self.odometer += math.dist(self.last_xy, xy)
        self.last_xy = tuple(xy)
        p = np.asarray(points, dtype=np.float64)
        if len(p):
            h = p[:, 2] - floor_z
            keep = ((h < self.max_h) & (h > -self.max_h)
                    & (np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) <= NEAR_M))
            p, h = p[keep], h[keep]
            ix = np.floor(p[:, 0] / self.res).astype(np.int64)
            iy = np.floor(p[:, 1] / self.res).astype(np.int64)
            key = ix * 1_000_003 + iy
            order = np.lexsort((-h, key))           # per cell, highest first
            key, ix, iy, h = key[order], ix[order], iy[order], h[order]
            first = np.r_[True, key[1:] != key[:-1]]
            for a, b, z in zip(ix[first], iy[first], h[first]):
                self.cells[(int(a), int(b))] = [float(z), self.odometer]
        # Forget what was seen more than keep_m of walking ago.
        old = self.odometer - self.keep_m
        if len(self.cells) > 4000:
            self.cells = {k: v for k, v in self.cells.items() if v[1] >= old}

    def patch(self, pose, wall_at=None):
        """(NY, NX) elevation patch at pose (x, y, yaw), NaN where unseen.

        `wall_at(x, y)`, metres to the nearest wall the map knows, makes those
        walls floor: cells within WALL_M of one read 0. Isaac has no walls, so
        the network took any tall thing in view for stairs and refused 52-96%
        of clear walks with a wall within 0.5 m (selftest affordance) -- every
        doorway. Walls are the map's to judge (the dream's clearance); the
        network judges the ground and what the map does not know is there."""
        # Each patch cell looks up the world cell its centre falls in. Splatting
        # world cells into the patch instead aliased: two landed in one patch
        # cell and the next stayed empty, and a floor seen whole read 44% seen.
        old = self.odometer - self.keep_m
        x, y, yaw = pose
        xs, ys = data.cell_centres()
        fx, fy = np.meshgrid(xs, ys)                    # [iy, ix], like the patch
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = x + c * fx - s * fy, y + s * fx + c * fy
        ix = np.floor(wx / self.res).astype(np.int64)
        iy = np.floor(wy / self.res).astype(np.int64)
        out = np.full(fx.shape, np.nan, dtype=np.float32)
        for r in range(out.shape[0]):
            for k in range(out.shape[1]):
                v = self.cells.get((int(ix[r, k]), int(iy[r, k])))
                if v is not None and v[1] >= old:
                    out[r, k] = v[0]
                if wall_at is not None and wall_at(wx[r, k], wy[r, k]) < WALL_M:
                    out[r, k] = 0.0
        return out


class Affordance:
    """The trained network, ready to judge a walk ahead."""

    def __init__(self, path=None, refuse_p=M.REFUSE_P):
        from cyberdog import paths
        self.net, self.info = M.load(path or paths.MODELS_DIR / "affordance" / "mlp.pt")
        self.refuse_p = refuse_p

    def judge(self, patch, target):
        """(class, probabilities, share of the patch seen) for walking to `target`,
        (dx, dy) in the dog's frame. Class None when too little was seen to say."""
        seen = float(np.isfinite(patch).mean())
        if seen < MIN_SEEN:
            return None, None, seen
        cls, probs = M.classify(self.net, patch[None], np.asarray([target], np.float32),
                                self.refuse_p)
        return int(cls[0]), probs[0], seen


def ahead(pose, xy, reach=1.2, min_d=data.TARGET_D[0]):
    """(dx, dy) in the dog's frame toward world point `xy`, at most `reach` away;
    None when it is nearer than min_d or outside the bearings the network
    was trained on (data.TARGET_BEARING) -- the dog is turning, not walking."""
    x, y, yaw = pose
    dx, dy = xy[0] - x, xy[1] - y
    d = math.hypot(dx, dy)
    if d < min_d:
        return None
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy = c * dx + s * dy, -s * dx + c * dy
    if abs(math.atan2(fy, fx)) > data.TARGET_BEARING:
        return None
    k = min(reach, d) / d
    return fx * k, fy * k

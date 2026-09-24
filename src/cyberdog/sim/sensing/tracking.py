"""Which of the things the LiDAR found are moving, and how fast.

`perception.py` says, honestly, that it has no memory: each scan stands alone.
For a crate that is the right trade -- a crate is where it was -- but it makes
a walking person indistinguishable from a post, and the two want opposite
answers. Step around a post. Stop for a person.

This is the smallest thing that tells them apart: cluster the unexplained
returns, match the clusters to last tick's, and keep a smoothed velocity.
No filter, no data association beyond nearest-centroid, no occlusion
reasoning. It runs on a few hundred points at 20 Hz and costs well under a
millisecond, which is the budget it has.

Velocity over a window, not between two ticks, and this is the whole reason
the module works at all. A range sensor sees the near face of a thing, so a
*stationary* crate's cluster is not a stable object: as the dog moves, its
visible face grows, shrinks, and every so often breaks into two clusters and
back. The centroid jumps half a metre in a tick when that happens, which as a
tick-to-tick difference is 10 m/s. Smoothing that does not help -- an EMA
turns one jump into a slow decay through the walking range, and the first
version of this stopped the dog nineteen times in an empty building.

What a crate cannot do is *travel*. Its centroid rattles about a fixed point;
a person's does not come back. So velocity is measured across WINDOW ticks of
history -- where it is now against where it was half a second ago -- and half
a second of rattle nets out to nearly nothing while half a second of walking
is over half a metre. Same threshold, a signal that can actually carry it.

Honest limits, and they are the same ones perception.py has:

- No occlusion model. A person who steps behind a crate is a lost track, and
  the track that reappears is a new one that has to earn `moving` again.
- Nearest-centroid matching swaps identities when two people pass each other.
  The velocities come back roughly right anyway -- both are walking, and the
  decision downstream is "is anyone about to cross", not "which one".
- Everything is in the world frame, so the dog's own motion is already out of
  it. That is only true because the returns are world points; do not "fix"
  this by working in the body frame.
"""
import math
from collections import deque

CELL = 0.35             # grid cell for clustering, metres
MATCH_D = 0.9           # how far a cluster may move in a tick and still be it
WINDOW = 10             # ticks of history the velocity is measured over: at
                        # 20 Hz, half a second. Long enough to average out a
                        # cluster breaking in two, short enough that a person
                        # who steps out is called moving within half a second
                        # -- which at 1.2 m/s is 0.6 m of their crossing gone.
MOVING_V = 0.55         # m/s before a track is called moving -- see above
MISS = 6                # ticks a track is kept alive without a match
# Two filters that are about shape rather than motion, and between them they
# removed every false stop in an empty building. Both come from the same
# observation: the thing being looked for is person-shaped, and the thing
# being confused with it is not.
MIN_POINTS = 4          # a cluster of one or two returns is a ray grazing a
                        # corner, and its "position" is wherever the ray
                        # happened to land, so it reads as moving at whatever
                        # speed the dog is. A person 8 m away is still a dozen
                        # returns; at the 4 m this decides anything at, forty.
MIN_RADIUS = 0.10       # ...and neither is a cluster with no width at all.
                        # This is not the same filter as MIN_POINTS and it is
                        # not redundant with it: the scan is 11 elevations per
                        # azimuth, and against a flat vertical face all eleven
                        # of them land on the same (x, y). So one azimuth of a
                        # crate is eleven returns at a single point -- plenty
                        # of points, no extent -- and where that point sits is
                        # decided by which single ray column happens to catch
                        # the corner this tick. A person has shoulders.
MAX_RADIUS = 0.50       # A cluster wider than this is not one person.
                        # The crates come back at 0.55-0.90 m -- a 1.2 m face,
                        # half seen -- and their centroid slides along that
                        # face as the dog walks past, at 0.6 m/s, which is a
                        # walking pace and is why a speed threshold alone
                        # cannot separate them. Shape can.


class Track:
    """One clustered thing, over time."""

    __slots__ = ("x", "y", "vx", "vy", "r", "age", "missed", "past", "dt")

    def __init__(self, x, y, r, dt):
        self.x, self.y, self.r, self.dt = x, y, r, dt
        self.vx = self.vy = 0.0
        self.age = 1
        self.missed = 0
        self.past = deque([(x, y)], maxlen=WINDOW)

    def observe(self, x, y, r):
        """Where it is now, and therefore how fast it has been going."""
        self.x, self.y, self.r = x, y, r
        self.past.append((x, y))
        self.age += 1
        self.missed = 0
        if len(self.past) > 1:
            ox, oy = self.past[0]
            span = (len(self.past) - 1) * self.dt
            self.vx, self.vy = (x - ox) / span, (y - oy) / span

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def moving(self):
        # A full window or nothing: a track two ticks old has a velocity
        # measured over two ticks, which is the noisy number this whole
        # design exists to avoid trusting.
        return (len(self.past) >= WINDOW
                and MIN_RADIUS <= self.r <= MAX_RADIUS
                and self.speed >= MOVING_V)

    def predict(self, t):
        """Where it will be in `t` seconds if it carries on. Straight line on
        purpose: over the two seconds this is asked about, a walking person is
        one, and anything cleverer would be inventing intent."""
        return (self.x + self.vx * t, self.y + self.vy * t)


def clusters(points, cell=CELL):
    """Group returns into things. Grid flood-fill, 8-connected.

    Not DBSCAN: the points come off a ray pattern, so they are already roughly
    gridded, and a dict of occupied cells is both simpler and faster than any
    library call at this size.
    """
    if len(points) == 0:
        return []
    bins = {}
    for p in points:
        bins.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)

    out, seen = [], set()
    for start in bins:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            group += bins[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in bins and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(group) < MIN_POINTS:
            continue
        xs = [p[0] for p in group]
        ys = [p[1] for p in group]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        r = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
        out.append((cx, cy, r))
    return out


class Tracker:
    """Clusters this tick, matched to the ones from last tick."""

    def __init__(self, dt):
        self.dt = dt
        self.tracks = []

    def update(self, points):
        """Fold in one scan's unexplained returns. Returns all live tracks."""
        found = clusters(points)
        unmatched = list(self.tracks)
        self.tracks = []

        for cx, cy, r in found:
            best, bd = None, MATCH_D
            for t in unmatched:
                d = math.hypot(cx - t.x, cy - t.y)
                if d < bd:
                    best, bd = t, d
            if best is None:
                self.tracks.append(Track(cx, cy, r, self.dt))
                continue
            unmatched.remove(best)
            best.observe(cx, cy, r)
            self.tracks.append(best)

        # A track that went unmatched is coasted for a few ticks rather than
        # dropped. A person half-hidden by a crate flickers in and out of the
        # returns, and a track that dies each time never reaches AGE, so the
        # dog would never believe anyone was moving.
        for t in unmatched:
            t.missed += 1
            if t.missed <= MISS:
                # Coasted, and the coasted position goes into the history too
                # -- otherwise the window holds a gap and the velocity across
                # it is measured over the wrong amount of time.
                t.x, t.y = t.predict(self.dt)
                t.past.append((t.x, t.y))
                self.tracks.append(t)
        return self.tracks

    def movers(self):
        return [t for t in self.tracks if t.moving]

    def unsettled(self):
        """Everything not yet known to be staying put.

        Moving things, and things too new to say. Both are the same answer to
        the only question that uses this -- whether to plan a detour around
        something -- because a detour is a commitment and you do not commit to
        a gap beside a thing you cannot yet describe. A crate first seen at
        8 m is in here for half a second, during which the dog covers 0.4 m
        and commits to nothing; a person is in here for as long as they are
        walking.

        Without the half-second, the tracker is still deciding what a person
        is while free_carrot is already choosing which side of them to pass,
        and the side it picks is chosen around somebody who will not be there.
        Measured: four runs in six ended against the floor-2 cart, having
        dodged a pedestrian into it.
        """
        return [t for t in self.tracks if t.moving or len(t.past) < WINDOW]


# -- yielding ---------------------------------------------------------------
#
# Whether to stop is not "is someone near me". Near is the wrong test in both
# directions: a person walking away two metres ahead is near and irrelevant,
# and one crossing four metres ahead at 1.4 m/s is far and about to be exactly
# where the dog will be. The question is whether the two paths meet, so that
# is the question asked -- closest approach between two straight lines over
# the next few seconds.
#
# The stop happens early for a reason that is about the handle, not the robot.
# A person is holding it. A stop timed so the dog halts a hand's width from a
# passing stranger is technically a success and is a bad experience for
# everybody involved; MISS_R buys the standoff that makes it read as courtesy.
HORIZON = 2.5           # seconds of future looked at
NEAR_R = 1.2            # and the range inside which this stops asking.
                        # Not a blind spot: something already this close is
                        # the emergency stop's business (run_building's
                        # `touching`) and free_carrot's, both of which act on
                        # it without caring whether it walks. What it is not
                        # is a yield -- deciding to wait for someone when they
                        # are half a metre away is not courtesy, it is a near
                        # miss. It also happens to be where the sensor is
                        # least able to tell what it is looking at: brushing
                        # past the floor-3 cartons, the handful of returns off
                        # one corner is person-sized and person-paced, and
                        # three runs in a row stopped for a cardboard box.
MISS_R = 0.95           # closest approach that counts as a conflict, metres
STEP_T = 0.1
CLEAR_R = 1.30          # ...and the room wanted back before moving off again.
                        # Wider than MISS_R on purpose: equal thresholds put
                        # the dog on the boundary, so it creeps, re-conflicts,
                        # stops, and shuffles its way past somebody. Hysteresis
                        # is what makes it one clean stop and one clean start.


def closest_approach(a, va, b, vb, horizon=HORIZON, step=STEP_T):
    """Least distance between two things moving in straight lines."""
    best = math.dist(a, b)
    t = step
    while t <= horizon + 1e-9:
        d = math.hypot((a[0] + va[0] * t) - (b[0] + vb[0] * t),
                       (a[1] + va[1] * t) - (b[1] + vb[1] * t))
        best = min(best, d)
        t += step
    return best


def conflict(xy, vel, tracks, radius=MISS_R, horizon=HORIZON):
    """The moving thing the dog is about to meet, or None.

    Only moving ones. A stationary cluster is a crate and belongs to
    free_carrot, which will go round it; handing it to the stopping logic as
    well is how a dog ends up halted in a gap it was successfully using.
    """
    worst, wd = None, radius
    for t in tracks:
        if not t.moving or math.dist(xy, (t.x, t.y)) < NEAR_R:
            continue
        d = closest_approach(xy, vel, (t.x, t.y), (t.vx, t.vy), horizon)
        # Its own width counts: the track is a centroid, the person is not.
        d -= min(t.r, 0.4)
        if d < wd:
            worst, wd = t, d
    return worst

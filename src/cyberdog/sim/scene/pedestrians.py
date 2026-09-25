"""People in the corridor, who are not in the map either.

`obstacles.py` put crates in the building to give the perception layer a job.
They are the easy half of that job: a crate is where it was last tick, so a
sensor with no memory is still right about it, and the answer -- step around
it -- is a geometry problem.

A person walking across the corridor breaks both of those. Where they are is
only half the question; the other half is where they are going, and a scan
that stands alone cannot answer it. And the right response is not to squeeze
past: it is to stop and let them go. A guide dog that threads a moving gap is
towing a blind person through it.

So these are deliberately the opposite of the crates in every way that
matters, and they are just as deliberately absent from data/building -- no
grid cell, no zone. Nothing here detects anything and nothing here is told to
the robot. The dog sees them with the LiDAR like anything else, and
`sensing/tracking.py` is what turns two scans into a velocity.

Two kinds, because they fail differently:

    along     walks the length of the corridor in a side lane, head-on or
              overtaking. Passes the dog at about 0.9 m. The one that must NOT
              trigger a stop every time, or the dog never finishes a run in a
              building with people in it.
    crossing  walks from one wall to the other, across the route. The one that
              must. It is clear of the dog's line when first seen and in it
              four seconds later, which is exactly the case a single scan
              cannot call.

Reaching the end of a leg is not an exit. They turn and carry on down the
corridor, because a person who walks into a wall and evaporates is not a
person -- and because the tracker reads a body that disappears from four
metres away exactly as it reads one that appears there: as something moving
very fast. So a walker is only ever recycled somewhere else while the dog
cannot see it; in view, it always has somewhere to walk to, and every frame
puts it somewhere it could plausibly have walked from.

They are not allowed to stop and stay stopped, either. A stopped person is a
permanent 0.44 m obstacle, and in a 2.7 m corridor whose route hugs one wall
that is something the dog can only squeeze past -- measured, at 0.09 m, which
is through them. Walking people can be waited for; parked ones just make the
corridor narrower.

Random, with a seed. `Crowd(floor, n, seed)` gives the same people every time
for the same seed, so a run that ends in a collision can be replayed, and a
different seed is a different day in the same building.

Politeness, and its limits. A walker pauses when the dog is right in front of
it (NOTICE below) -- real people look where they are going, and without it an
oblivious crosser walks into a dog that correctly stopped to let it past,
which scores as a collision the robot could not have avoided. That is the
only thing they know about the dog. They do not step round it, they do not
slow down early, and they will happily walk into the side of it if it is not
in front of them. Making them any cleverer would be quietly solving the
robot's problem for it.
"""
import math
import random

from cyberdog.sim.scene import obstacles
from cyberdog.sim.scene.levels import floor_z

# Bodies emitted per floor. Fixed at scene-build time, because mocap bodies
# live in the XML: `--pedestrians N` picks how many of these are walking, and
# the rest are parked underground. Raising it means rebuilding the scene.
POOL = 4

COLOURS = ["0.25 0.45 0.75 1", "0.70 0.30 0.35 1",
           "0.35 0.60 0.40 1", "0.55 0.40 0.68 1"]
PARK_Z = -50.0          # where an unused body waits: far below any floor,
                        # out of the LiDAR's 12 m and under the ground plane

HEIGHT = 1.70           # capsule top, above the floor
RADIUS = 0.22           # a shoulder-width-ish cylinder, seen from any side
FOOT = 0.10             # bottom of the capsule -- above perception's MIN_H

# Where they may walk. The corridor's real walls, inset by a shoulder so a
# person is never inside one.
LANE_Y = (8.15 + RADIUS + 0.05, 10.85 - RADIUS - 0.05)
SIDE_LANES = (8.60, 10.40)      # the two along-corridor lanes
# Clear of the stairwells (x < 2.5) and the lift car (x > 45.6) at both ends.
LANE_X = (4.0, 44.0)

SPEED = (0.85, 1.45)    # m/s, a walking pace
PAUSE = (1.0, 4.0)      # seconds between finishing one leg and starting another
NOTICE = 0.75           # they stop if the dog is this close, ahead of them
# Where a finished walker may reappear. Not next to the dog: a person who
# materialises three metres in front of it is a teleport, and the tracker --
# correctly -- reads a teleport as something moving very fast indeed.
RESPAWN_CLEAR = 9.0     # metres from the dog, minimum
# ...and *when*. The same argument applies to the disappearing end of the
# move: a walker that vanishes off the end of its leg in full view is the same
# teleport seen from the other side.
SEEN_R = 12.0           # lidar.MAX_R: past this the dog reports nothing at
                        # all. Not imported, to keep this module free of
                        # mujoco -- build_scene runs it before there is a
                        # model to scan. Distance is the whole test: the
                        # Mid-360 is a 360-degree sensor, so "behind the dog"
                        # is not unseen.
TURN = (0.3, 1.2)       # seconds spent turning round at the end of a leg
ONWARD = (8.0, 20.0)    # metres of corridor walked after turning


class Walker:
    """One person, walking a there-and-back leg at a constant pace."""

    def __init__(self, a, b, speed, pause, kind, phase=0.0):
        self.reset(a, b, speed, pause, kind, phase)

    def reset(self, a, b, speed, pause, kind, phase=0.0):
        self.a, self.b = a, b           # the two ends of the leg
        self.speed, self.pause, self.kind = speed, pause, kind
        self.t = phase                  # 0..1 along a -> b
        self.waiting = 0.0              # seconds left before this leg starts
        self.done = False               # walked it; wants a new one
        self._place()

    @property
    def length(self):
        return math.dist(self.a, self.b)

    def _place(self):
        self.x = self.a[0] + (self.b[0] - self.a[0]) * self.t
        self.y = self.a[1] + (self.b[1] - self.a[1]) * self.t

    def heading(self):
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        d = math.hypot(dx, dy) or 1.0
        return (dx / d, dy / d)

    def step(self, dt, dog=None):
        """Advance one control tick. `dog` is (x, y), for NOTICE only."""
        if self.waiting > 0.0:
            self.waiting -= dt
            return

        # The one thing they know about the robot: do not walk into the thing
        # directly in front of you.
        if dog is not None:
            hx, hy = self.heading()
            dx, dy = dog[0] - self.x, dog[1] - self.y
            d = math.hypot(dx, dy)
            if d < NOTICE and (dx * hx + dy * hy) > 0:
                return

        self.t += self.speed * dt / (self.length or 1.0)
        if self.t >= 1.0:
            # Walked it. They do NOT turn round and walk back down the same
            # line: a crosser pacing the same two metres for ever is a moving
            # wall, and the dog waiting politely for it never gets down the
            # corridor. `Crowd` either sends them on along the corridor or,
            # if the dog cannot see them, gives them somewhere else to be.
            self.t, self.done = 1.0, True
        self._place()


class Crowd:
    """The people on one floor, and where their bodies are this tick."""

    def __init__(self, floor, n, seed=0):
        self.floor = floor
        self.z = floor_z(floor)
        self.walkers = []
        self._rng = random.Random(seed * 1000 + floor)
        if n <= 0:
            return
        rng = self._rng = random.Random(seed * 1000 + floor)
        boxes = obstacles.boxes(floor)
        for _ in range(min(n, POOL)):
            w = self._make(rng, boxes)
            if w is not None:
                self.walkers.append(w)

    def _make(self, rng, boxes, tries=40):
        """One walker, somewhere that is not inside a crate."""
        for _ in range(tries):
            # Crossings are the interesting case, so they are the common one.
            kind = "crossing" if rng.random() < 0.65 else "along"
            if kind == "crossing":
                x = rng.uniform(*LANE_X)
                if self._blocked(x, boxes):
                    continue
                a, b = (x, LANE_Y[0]), (x, LANE_Y[1])
                if rng.random() < 0.5:
                    a, b = b, a
            else:
                y = rng.choice(SIDE_LANES)
                x0 = rng.uniform(LANE_X[0], LANE_X[1] - 12.0)
                a, b = (x0, y), (x0 + rng.uniform(12.0, 20.0), y)
                if rng.random() < 0.5:
                    a, b = b, a
            return Walker(a, b, rng.uniform(*SPEED), rng.uniform(*PAUSE),
                          kind, phase=rng.random())
        return None

    @staticmethod
    def _blocked(x, boxes, margin=0.5):
        """A crossing lane that runs through a crate: the person would walk
        through it, which looks like a bug and is one."""
        return any(bx0 - margin <= x <= bx1 + margin
                   for bx0, bx1, _y0, _y1, _h, _n in boxes)

    def step(self, dt, dog=None):
        for w in self.walkers:
            w.step(dt, dog)
            if not w.done:
                continue
            if self._unseen(w, dog):
                self._respawn(w, dog)
            else:
                self._onward(w)

    @staticmethod
    def _unseen(w, dog):
        """Is this walker out of the dog's sight right now?

        It has to be the walker's *current* position that is far away, not the
        new leg's start: RESPAWN_CLEAR already covers where they arrive, and
        what was missing was where they left from.
        """
        return dog is None or math.dist((w.x, w.y), dog) > SEEN_R

    def _onward(self, w):
        """Turn and carry on, from exactly where the last leg ended.

        The alternative to vanishing while watched, and the alternative to
        standing there: both put a body somewhere it could not have walked to.
        This starts the new leg at the walker's own feet, so there is no frame
        in which it jumps -- it just turns a corner, which is what somebody who
        has reached a wall does.
        """
        rng = self._rng
        lane = min(SIDE_LANES, key=lambda y: abs(y - w.y))
        span = rng.uniform(*ONWARD)

        # Carry on the way they were already headed. Sending them at the
        # middle of the corridor instead -- which is what picking the
        # direction from their position does -- walks the whole crowd into
        # the centre and leaves them pacing there, which is the moving wall
        # this module exists not to build. A crossing leg has no x direction
        # of its own, so that one tosses a coin.
        dx = w.b[0] - w.a[0]
        way = 1.0 if dx > 0 else -1.0 if dx < 0 else rng.choice((-1.0, 1.0))
        x = w.x + way * span
        if not LANE_X[0] <= x <= LANE_X[1]:     # out of corridor: turn round
            x = w.x - way * span
        x = max(LANE_X[0], min(LANE_X[1], x))

        w.reset((w.x, w.y), (x, lane), rng.uniform(*SPEED),
                rng.uniform(*PAUSE), "along")
        w.waiting = rng.uniform(*TURN)

    def _respawn(self, w, dog, tries=30):
        """Somewhere else in the building, after a pause.

        Only ever called for a walker the dog cannot see, so the move itself
        is never witnessed -- by the camera or by the tracker.

        The pause is what makes the corridor breathe: without it every walker
        is always walking and the dog meets a continuous wall of people, which
        is not a building, it is a turnstile.
        """
        boxes = obstacles.boxes(self.floor)
        for _ in range(tries):
            fresh = self._make(self._rng, boxes)
            if fresh is None:
                continue
            if dog is not None and math.dist((fresh.a[0], fresh.a[1]), dog) < RESPAWN_CLEAR:
                continue
            w.reset(fresh.a, fresh.b, fresh.speed, fresh.pause, fresh.kind)
            w.waiting = self._rng.uniform(*PAUSE)
            return
        # Nowhere far enough away this tick. Send them on down the corridor
        # instead and ask again at the end of that -- rather than appearing in
        # the dog's lap, or snapping back to the start of the leg they have
        # just walked, which is the same teleport in the other direction.
        self._onward(w)

    def poses(self):
        """(body name, (x, y, z)) for every body in this floor's pool.

        Every body, not just the busy ones: an unused body has to be sent
        somewhere, and leaving it where the XML put it would stand a person in
        the corridor who never moves -- a crate with legs.
        """
        out = []
        for k in range(POOL):
            name = f"ped_f{self.floor}_{k}"
            if k < len(self.walkers):
                w = self.walkers[k]
                out.append((name, (w.x, w.y, self.z)))
            else:
                out.append((name, (0.0, 0.0, PARK_Z)))
        return out

    def positions(self):
        """(x, y) of the people actually walking -- ground truth, for scoring."""
        return [(w.x, w.y) for w in self.walkers]


def bodies(floor):
    """MuJoCo mocap bodies for one floor's pool of people.

    Mocap for the same reason the lift car is: they sit outside qpos, so the
    Go2's "home" keyframe still fits the model and MujocoRobot's qpos indices
    are untouched. Unlike `obstacles.geoms`, this takes no floor height: a
    mocap body's XML position is only where it waits, and every one of them is
    put where its walker is on the first tick of a run.
    """
    out = []
    for k in range(POOL):
        out.append(
            f'<body name="ped_f{floor}_{k}" mocap="true" '
            f'pos="0 0 {PARK_Z:.1f}">\n'
            f'      <geom name="pedg_f{floor}_{k}" type="capsule" '
            f'fromto="0 0 {FOOT + RADIUS:.3f} 0 0 {HEIGHT - RADIUS:.3f}" '
            f'size="{RADIUS:.3f}" rgba="{COLOURS[k % len(COLOURS)]}" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>')
    return out

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
              overtaking. Keeps right, as people do here and as the dog does,
              so head-on they pass it on its left at about 0.9 m. The one
              that must NOT trigger a stop every time, or the dog never
              finishes a run in a building with people in it.
    crossing  walks from one wall to the other, across the route -- straight
              over, or on a slant towards a door further along, from the left
              or the right. The one that must. It is clear of the dog's line
              when first seen and in it four seconds later, which is exactly
              the case a single scan cannot call.

Both, all run long. Every walker used to become an `along` one after its first
leg, so a few seconds in nobody near the dog was crossing at all -- measured
over three seeds, not one tick in any of them -- and with lanes picked at
random, up to half the head-on walkers came down the dog's own side.

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
which scores as a collision the robot could not have avoided. And if the dog
stays in their way for GIVE_WAY seconds, they step round it, as anybody
would -- otherwise a crosser and a dog that has correctly stopped for them
wait for each other until the run ends. That is all they know about the dog.
They do not slow down early, and they will happily walk into the side of it
if it is not in front of them. Making them any cleverer would be quietly
solving the robot's problem for it.

That is the default, `Crowd(..., aware=False)`: a stress test, and what every
Gate D campaign before 2026-10-12 was scored on. It is not what a corridor is
like. People see a guide dog and a person with a cane from metres away and
make room, and the oblivious crowd makes situations no real one would: in
`room 101` seed 5 a crosser froze 0.75 m in front of the dog because the dog
was there, the dog went round them, and the turn swung its user into the
wall. `aware=True` is the other end. Within AWARE_R of the party -- the dog
and the person behind it, on one line -- people:

  - crossing: do not step into its path while it is coming. One still at the
    wall waits there until it has gone by; one already across its line
    finishes crossing, a little faster, instead of stopping in it.
  - along, head-on: drift over to leave PASS_ROOM between their line and its.
  - along, the same way: do not tailgate. They stay FOLLOW_GAP behind, or
    pass on the far side if there is PASS_ROOM there.
  - never stand in front of it: caught in its path within NOTICE, they step
    aside at once instead of after GIVE_WAY.

They still know nothing else: not where the dog is going to turn, and not
the crates. Both modes are the same people, seed for seed.
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
SIDE_LANES = (8.60, 10.40)      # the two along-corridor lanes: south, north
CROSS_P = 0.4           # chance a finished leg is followed by a crossing
CROSS_DRIFT = 2.0       # metres a crossing may slant along the corridor
# Clear of the stairwells (x < 2.5) and the lift car (x > 45.6) at both ends.
LANE_X = (4.0, 44.0)

SPEED = (0.85, 1.45)    # m/s, a walking pace
PAUSE = (1.0, 4.0)      # seconds between finishing one leg and starting another
NOTICE = 0.75           # they stop if the dog is this close, ahead of them
GIVE_WAY = 2.0          # ...and after this many seconds of it, step round it.
                        # Without it a crosser who stopped because the dog was
                        # in their way waited for the dog, which was waiting
                        # for them: seed 8 of room 201 ended in that standoff.
                        # Shorter than the dog's own PATIENCE, as it is for
                        # people: someone walking gives way before a guide dog
                        # with a blind person behind it has to.
SIDESTEP = 0.9          # metres off their line, the room people give in passing
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

# aware=True only. The party is the dog and its user, HANDLER_BEHIND (1.1 m,
# run_building) straight behind it: one line, PARTY_BACK long.
AWARE_R = 5.0           # metres: within this of the dog, people take it into account
PASS_ROOM = 1.1         # metres between their line and the party's when passing
PARTY_BACK = 1.6        # metres behind the dog that are still the party (user + 0.5)
FOLLOW_GAP = 2.0        # metres kept behind the user when walking the same way
DRIFT_V = 0.8           # m/s sideways, when moving over to give room
CROSS_HURRY = 1.25      # pace multiplier for a crosser finishing in its path
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
        self.held = 0.0                 # seconds stood behind the dog
        self.off = 0.0                  # aware: sideways shift off the leg's line
        self._place()

    def _step_round(self, dx, dy):
        """Give way: a sidestep off their line, away from the dog at (dx, dy).

        Crowd sends them on from wherever it ends, as after any leg. To the
        other side if the wall leaves no room on this one.
        """
        hx, hy = self.heading()
        left = hx * dy - hy * dx > 0             # the dog is on their left
        for side in ((-1.0, 1.0) if left else (1.0, -1.0)):
            nx, ny = -hy * side, hx * side       # side = +1: their left
            tx = max(LANE_X[0], min(LANE_X[1], self.x + nx * SIDESTEP + hx * 0.3))
            ty = max(LANE_Y[0], min(LANE_Y[1], self.y + ny * SIDESTEP + hy * 0.3))
            if math.dist((tx, ty), (self.x, self.y)) >= SIDESTEP / 2:
                break
        self.a, self.b, self.t = (self.x, self.y), (tx, ty), 0.0
        self.held = 0.0

    @property
    def length(self):
        return math.dist(self.a, self.b)

    def _place(self):
        self.x = self.a[0] + (self.b[0] - self.a[0]) * self.t
        self.y = self.a[1] + (self.b[1] - self.a[1]) * self.t
        if self.off:
            # Only an `along` walker drifts, and its leg runs along x, so the
            # shift is in y. Kept inside the corridor's lanes.
            self.y = max(LANE_Y[0], min(LANE_Y[1], self.y + self.off))

    def heading(self):
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        d = math.hypot(dx, dy) or 1.0
        return (dx / d, dy / d)

    def step(self, dt, dog=None, aware=False):
        """Advance one control tick. `dog` is (x, y) or (x, y, yaw); `aware`
        needs the yaw, and without it falls back to the oblivious rules."""
        if self.waiting > 0.0:
            self.waiting -= dt
            return
        if aware and dog is not None and len(dog) == 3:
            return self._step_aware(dt, dog)

        # The one thing they know about the robot: do not walk into the thing
        # directly in front of you -- and do not stand behind it for ever.
        if dog is not None:
            hx, hy = self.heading()
            dx, dy = dog[0] - self.x, dog[1] - self.y
            d = math.hypot(dx, dy)
            if d < NOTICE and (dx * hx + dy * hy) > 0:
                self.held += dt
                if self.held >= GIVE_WAY:
                    self._step_round(dx, dy)
                return
        self.held = 0.0

        self._advance(self.speed * dt)

    def _advance(self, metres):
        self.t += metres / (self.length or 1.0)
        if self.t >= 1.0:
            # Walked it. They do NOT turn round and walk back down the same
            # line: a crosser pacing the same two metres for ever is a moving
            # wall, and the dog waiting politely for it never gets down the
            # corridor. `Crowd` either sends them on along the corridor or,
            # if the dog cannot see them, gives them somewhere else to be.
            self.t, self.done = 1.0, True
        self._place()

    def _step_aware(self, dt, dog):
        """One tick of somebody who has seen the dog and its user coming.

        In the party's frame: `along` is how far ahead of the dog they are
        (negative: beside or behind the user), `lat` how far off its line.
        """
        x, y, yaw = dog
        ux, uy = math.cos(yaw), math.sin(yaw)
        rx, ry = self.x - x, self.y - y
        along, lat = rx * ux + ry * uy, -rx * uy + ry * ux
        hx, hy = self.heading()
        if math.hypot(rx, ry) > AWARE_R or along < -(PARTY_BACK + FOLLOW_GAP + 1.0):
            self._relax(dt)
            self._advance(self.speed * dt)
            return
        in_path = abs(lat) < PASS_ROOM and -PARTY_BACK < along
        facing = hx * ux + hy * uy            # +1 same way as the party, -1 head-on
        # The party's left, in world y: an `along` walker can only drift in y,
        # and the party's left-hand normal (-uy, ux) has y-component ux.
        left_y = 1.0 if ux >= 0 else -1.0

        if self.kind == "crossing":
            if in_path and along < NOTICE:
                # Caught in front of it: out of the way now, not in 2 s.
                self._step_round(-rx, -ry)
                return
            if not in_path:
                # Would stepping on take them into its path before it has gone?
                nxt_lat = lat + (-hx * uy + hy * ux) * self.speed * dt
                entering = abs(nxt_lat) < abs(lat) and abs(nxt_lat) < PASS_ROOM + 0.2
                if entering and along > -PARTY_BACK:
                    return                    # wait at the side till it has passed
            self._advance(self.speed * (CROSS_HURRY if in_path else 1.0) * dt)
            return

        if facing < 0:
            # Head-on: move over to whichever side of it has room -- their own
            # if it can, across if the wall leaves too little -- slowing until
            # they are clear, and stay over until they are past the user.
            if along > -PARTY_BACK - 0.5:
                if abs(lat) < PASS_ROOM - 0.05:
                    room = self._room_beside(y, lat, left_y)
                    if not room:          # neither side has it: the roomier one
                        room = 1.0 if (LANE_Y[1] - y) > (y - LANE_Y[0]) else -1.0
                    self._drift(room * DRIFT_V * dt)
                    self._advance(self.speed * 0.6 * dt)
                    return
            else:
                self._relax(dt)
            self._advance(self.speed * 0.8 * dt)
            return

        # The same way as the party.
        if along > 1.0:
            self._relax(dt)                   # ahead of it and walking away
        elif abs(lat) >= PASS_ROOM - 0.05:
            pass                              # in a passing lane: keep it till past
        else:
            room = self._room_beside(y, lat, left_y)
            gap = -along - PARTY_BACK         # metres from the user's back
            if room:
                self._drift(room * DRIFT_V * dt)
                self._advance(self.speed * 0.6 * dt)
                return
            if along < 0 and gap < FOLLOW_GAP:
                return                        # no room to pass: hang back, as people do
        self._advance(self.speed * dt)

    @staticmethod
    def _room_beside(party_y, lat, left_y):
        """Which way to drift (world y, +1/-1) to overtake the party with
        PASS_ROOM, the side they are already on first; 0 if neither has it."""
        for side in ((1.0, -1.0) if lat >= 0 else (-1.0, 1.0)):
            dy = side * left_y
            if LANE_Y[0] <= party_y + dy * PASS_ROOM <= LANE_Y[1]:
                return dy
        return 0.0

    def _drift(self, dy):
        lo, hi = LANE_Y[0] - (self.y - self.off), LANE_Y[1] - (self.y - self.off)
        self.off = max(lo, min(hi, self.off + dy))

    def _relax(self, dt):
        """Back towards their own lane once the party is past."""
        if self.off:
            step = min(abs(self.off), DRIFT_V * dt)
            self.off -= math.copysign(step, self.off)


class Crowd:
    """The people on one floor, and where their bodies are this tick."""

    def __init__(self, floor, n, seed=0, aware=False):
        self.floor = floor
        self.aware = aware
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
                x1 = self._slant(rng, x)
                if self._blocked(x, boxes, x1):
                    continue
                a, b = (x, LANE_Y[0]), (x1, LANE_Y[1])
                if rng.random() < 0.5:
                    a, b = (x1, LANE_Y[1]), (x, LANE_Y[0])
            else:
                x0 = rng.uniform(LANE_X[0], LANE_X[1] - 12.0)
                x1 = x0 + rng.uniform(12.0, 20.0)
                if rng.random() < 0.5:
                    x0, x1 = x1, x0
                y = self.keep_right(x1 - x0)
                a, b = (x0, y), (x1, y)
            return Walker(a, b, rng.uniform(*SPEED), rng.uniform(*PAUSE),
                          kind, phase=rng.random())
        return None

    @staticmethod
    def _blocked(x, boxes, x1=None, margin=0.5):
        """A crossing lane that runs through a crate: the person would walk
        through it, which looks like a bug and is one. `x1` is the far end of
        a slanted crossing; the whole span between is tested."""
        lo, hi = (x, x) if x1 is None else (min(x, x1), max(x, x1))
        return any(bx0 - margin <= hi and lo <= bx1 + margin
                   for bx0, bx1, _y0, _y1, _h, _n in boxes)

    @staticmethod
    def _slant(rng, x):
        """Where a crossing starting at `x` ends, along the corridor."""
        return max(LANE_X[0], min(LANE_X[1], x + rng.uniform(-CROSS_DRIFT, CROSS_DRIFT)))

    @staticmethod
    def keep_right(dx):
        """The side lane for walking the corridor in direction `dx`: the one
        on the walker's right. East (+x) has south on its right."""
        return SIDE_LANES[0] if dx > 0 else SIDE_LANES[1]

    def step(self, dt, dog=None):
        """`dog` is (x, y), or (x, y, yaw) -- which an aware crowd needs."""
        for w in self.walkers:
            w.step(dt, dog, self.aware)
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
        return dog is None or math.dist((w.x, w.y), dog[:2]) > SEEN_R

    def _onward(self, w):
        """Turn and carry on, from exactly where the last leg ended.

        The alternative to vanishing while watched, and the alternative to
        standing there: both put a body somewhere it could not have walked to.
        This starts the new leg at the walker's own feet, so there is no frame
        in which it jumps -- it just turns a corner, which is what somebody who
        has reached a wall does.
        """
        rng = self._rng
        # Sometimes over to the other side instead -- to a door opposite, or
        # a little further along. Starting from wherever they stand, so it is
        # still a turn and not a jump, and from whichever wall that is: the
        # dog meets crossers from its left and from its right.
        if rng.random() < CROSS_P:
            x1 = self._slant(rng, w.x)
            if not self._blocked(w.x, obstacles.boxes(self.floor), x1):
                far = LANE_Y[1] if w.y < sum(LANE_Y) / 2 else LANE_Y[0]
                w.reset((w.x, w.y), (x1, far), rng.uniform(*SPEED),
                        rng.uniform(*PAUSE), "crossing")
                w.waiting = rng.uniform(*TURN)
                return
        span = rng.uniform(*ONWARD)

        # Carry on the way they were already headed. Sending them at the
        # middle of the corridor instead -- which is what picking the
        # direction from their position does -- walks the whole crowd into
        # the centre and leaves them pacing there, which is the moving wall
        # this module exists not to build. Off a crossing, which ends at a
        # wall, the way that has that wall on their right: the other way puts
        # their keep-right lane across the corridor, and the 8-20 m diagonal
        # to it is a walker head-on down the dog's side the whole way.
        if w.kind == "crossing":
            way = -1.0 if w.y > sum(LANE_Y) / 2 else 1.0
        else:
            dx = w.b[0] - w.a[0]
            way = 1.0 if dx > 0 else -1.0 if dx < 0 else rng.choice((-1.0, 1.0))
        x = w.x + way * span
        if not LANE_X[0] <= x <= LANE_X[1]:     # out of corridor: turn round
            x = w.x - way * span
        x = max(LANE_X[0], min(LANE_X[1], x))
        lane = self.keep_right(x - w.x)

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
            if dog is not None and math.dist((fresh.a[0], fresh.a[1]), dog[:2]) < RESPAWN_CLEAR:
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

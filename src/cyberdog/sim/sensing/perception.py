"""What the LiDAR found that the map cannot account for.

The twin's safety decisions all read one number: clearance, metres to the
nearest thing the dog must not touch. `clearance.clearance_test` builds it
from the occupancy grid and the Behavior Layer zones, and until now that was
the only source -- which is why the dog walked through `obstacles.py`'s crates
at a reported safety of 0.99. The picture never entered the decision.

This module adds the second source, and only the second source: it takes a
scan, throws away everything the static map already explains, and offers the
rest as the same clearance function everything downstream already consumes.
`Dream` and the VAMOS gate need no changes -- they are handed this instead of
the static closure and cannot tell the difference.

Two jobs, and the second one is the one that actually makes the dog go round.

1. `LiveClearance` -- fused clearance, for the gate and the imagined rollouts.
   This is what makes a candidate path through a crate score 0.00.

2. `free_destination` -- move the goal, when the goal is inside a crate. This is the
   half that was missing, and it is worth being precise about why. VAMOS drives
   at whatever goal pixel it is given. The destination comes from A* on a map with no
   crate in it, so the pixel lands *on* the crate, and the model obligingly
   draws five paths into it -- measured: at 0.8 m the crate fills 60% of the
   frame and all five candidates still go straight through. That is not a model
   that cannot avoid obstacles; it is a model being told to walk into one.
   Rejecting those five and having nothing left is not avoidance either. So the
   map's half of the spec's division of labour -- "map decides WHERE, VLM
   decides HOW" -- has to use what the sensor found, and hand VAMOS a goal on
   open floor. Then the VLM chooses the way there and the gate judges it.

A range sensor sees surfaces, not solids: the crate comes back as the single
face pointing at the dog, and the space behind that face is not free, it is
unknown. Treating unknown as free is how a robot plans confidently into the
inside of a box, so each return also shadows SHADOW metres of the ray behind
it. That is the standard costmap answer and it is deliberately conservative --
it can over-claim when the real object is thinner than SHADOW.

Honest limits. No memory: each scan stands alone, which is fine for a 360-degree
sensor at 12 m and wrong the moment something is occluded. Beyond the local
window and beyond LiDAR range, clearance falls back to the static map, so an
imagined rollout that runs 10 m ahead is scored optimistically at its far end.
And returns the static field already explains are dropped wholesale, so an
obstacle parked against a wall is invisible -- it is inside the wall's own
inflation radius.
"""
import math

import numpy as np
from scipy import ndimage

from cyberdog.sim.clearance import clearance_field, wall_face_field

EXPLAINED = 0.35        # static clearance above this: the map calls it open floor
MIN_H, MAX_H = 0.08, 1.8  # ankle height to head height, above the floor
WINDOW = 8.0            # half-width of the local costmap, metres
SHADOW = 1.0            # how far behind a return to treat as unknown, metres
# Thresholds a displaced goal must satisfy. Deliberately modest: the corridor
# gap beside a crate is about 1 m wide, and a goal needing 0.45 m from the
# crate *and* 0.25 m from the wall leaves a 0.27 m window that any small change
# empties -- which reads as "no way past" in a corridor that plainly has one.
# Wide commitment comes from picking the roomiest offset, not from the floor.
DESTINATION_CLEAR = 0.35     # room a displaced goal must keep from a detected thing
LINE_NEED = 0.20        # ...and the room the way there merely has to survive.
                        # Above ROBOT_R (0.16) so it is still a margin, but not
                        # so far above that rounding an obstacle's corner --
                        # which is the whole manoeuvre -- reads as impossible.
CLUSTER_PAD = 0.25      # margin around a moving thing when taking it out of
                        # the planning field: its track is a centroid and a
                        # radius, and the returns off a coat sleeve are a
                        # little wider than either.
SKIP = 0.30             # metres of the line ignored: where the dog already is
STATIC_MIN = 0.20       # ...and from the walls the map already knew about
MAX_OFFSET = 1.6        # how far sideways the goal may be moved, metres.
                        # Tried at 2.2 -- the corridor's full width, so that a
                        # wall-to-wall crossing is expressible -- and it made
                        # things worse, not better: the dog committed to a goal
                        # on the far side and drove through the obstacle to
                        # reach it, because a straight line that is clear when
                        # the goal is chosen is not clear by the time a long
                        # crossing is half done. Keep it short until something
                        # plans the curve (CE-RRT*, spec L6 s2).
OFFSET_STEP = 0.10
INF = 99.0


class LiveClearance:
    """Static clearance, with whatever the LiDAR found laid over the top.

    Mutable on purpose: `run_building.policy()` caches one VamosPolicy per
    floor, and both its gate and its Dream close over this object, so updating
    it in place is what lets a scan reach them without rebuilding anything.
    """

    def __init__(self, floor):
        self.floor = floor
        self.dist, self.res, self.ox, self.oy = clearance_field(floor)
        self.faces = wall_face_field(floor)[0]   # same grid geometry, real wall faces
        self.H, self.W = self.dist.shape
        self.n = int(2 * WINDOW / self.res)
        self.local = None               # (edt, x0, y0) while something is seen
        self.still = None               # ...the same, minus anything walking
        self.points = np.empty((0, 2))  # the unexplained returns, for drawing
        self._scan = None               # this tick's returns, for `exclude`

    def _static(self, x, y):
        """Vectorised static lookup. Outside the map reads as zero clearance."""
        c = ((x - self.ox) / self.res).astype(int)
        r = (self.H - 1 - (y - self.oy) / self.res).astype(int)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        out = np.zeros(len(x))
        out[ok] = self.dist[r[ok], c[ok]]
        return out

    def update(self, points, floor_z, centre, origin=None):
        """Fold one scan in. `centre` is where the local window sits (the dog),
        `origin` where the rays came from -- needed to know which way is behind."""
        self.local = self.still = None
        self.points = np.empty((0, 2))
        self._scan = None
        if len(points) == 0:
            return

        rel = points[:, 2] - floor_z
        m = (rel > MIN_H) & (rel < MAX_H)
        if not m.any():
            return
        px, py = points[m, 0], points[m, 1]

        # Anything standing where the map already says "not free" is a wall, a
        # stairwell or the lift shaft -- known, and handled by the static field.
        unknown = self._static(px, py) > EXPLAINED
        if not unknown.any():
            return
        px, py = px[unknown], py[unknown]
        self.points = np.column_stack([px, py])
        self._scan = (px, py, centre, origin)
        self.local = self.still = self._field(px, py, centre, origin)

    def exclude(self, movers):
        """Rebuild the planning field with these things left out of it.

        `movers` is (x, y, radius) per thing that is walking. They stay in the
        safety field -- __call__, which the VAMOS gate and the imagined
        rollouts read, still sees every return, and the proximity stop still
        fires on them. What changes is `detected_at`, which is what
        `free_destination` plans detours around, and a person should not be planned
        around: the gap beside them is a gap that is leaving, and a route
        committed to it is committed to where they were.

        Worth one run in six, measured: six seeds of three people, with this
        and without, and the run it saves is one where the dog committed to
        the side of a corridor that was roomier only because somebody was
        standing on the other one, and was still committed to it after they
        had walked away. Waiting is the answer to a person; going round is the
        answer to a crate; mixing the inputs mixes the answers.
        """
        if self._scan is None or not movers:
            self.still = self.local
            return
        px, py, centre, origin = self._scan
        keep = np.ones(len(px), dtype=bool)
        for mx, my, mr in movers:
            keep &= np.hypot(px - mx, py - my) > mr + CLUSTER_PAD
        if keep.all():
            self.still = self.local
            return
        self.still = (self._field(px[keep], py[keep], centre, origin)
                      if keep.any() else None)

    def _field(self, px, py, centre, origin):
        """(edt, x0, y0) for a set of returns -- the local costmap itself."""
        # One small distance transform instead of rebuilding the 385x964 static
        # field, which costs 125 ms and would not survive a 20 Hz control loop.
        x0, y0 = centre[0] - WINDOW, centre[1] - WINDOW
        occ = np.zeros((self.n, self.n), dtype=bool)
        c = ((px - x0) / self.res).astype(int)
        r = ((py - y0) / self.res).astype(int)
        ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
        if not ok.any():
            return None
        occ[r[ok], c[ok]] = True

        # Everything behind a return is unknown, not free. Without this the
        # inside of a crate reads as open floor -- only its near face is ever
        # seen -- and a goal can be placed in the middle of one.
        if origin is not None and SHADOW > 0:
            ox_, oy_ = float(origin[0]), float(origin[1])
            dx, dy = px - ox_, py - oy_
            d = np.hypot(dx, dy)
            live = d > 1e-6
            ux, uy = dx[live] / d[live], dy[live] / d[live]
            bx, by = px[live], py[live]
            for t in np.arange(self.res, SHADOW + 1e-9, self.res):
                sc = ((bx + ux * t - x0) / self.res).astype(int)
                sr = ((by + uy * t - y0) / self.res).astype(int)
                m2 = (sr >= 0) & (sr < self.n) & (sc >= 0) & (sc < self.n)
                occ[sr[m2], sc[m2]] = True

        return (ndimage.distance_transform_edt(~occ) * self.res, x0, y0)

    def detected_at(self, x, y):
        """Distance to the nearest *detected* thing, ignoring the static map.

        The two sources have to be asked separately when deciding whether to
        move a goal. A doorway is tight in the static field and A* routed
        through it anyway -- that is not an obstruction, it is the building,
        and a destination pushed sideways out of a doorway ends up in a wall.
        Only something the map has no record of is a reason to move the goal.

        And only something that is standing still: this reads the field
        `exclude` leaves behind, so a walking person is not in it. See there
        for why.
        """
        if self.still is None:
            return INF
        edt, x0, y0 = self.still
        c = int((x - x0) / self.res)
        r = int((y - y0) / self.res)
        if 0 <= r < self.n and 0 <= c < self.n:
            return float(edt[r, c])
        return INF

    def static_at(self, x, y):
        """Clearance from the map alone -- walls, stairwells, the lift shaft."""
        c = int((x - self.ox) / self.res)
        r = int(self.H - 1 - (y - self.oy) / self.res)
        if 0 <= r < self.H and 0 <= c < self.W:
            return float(self.dist[r, c])
        return 0.0

    def wall_at(self, x, y):
        """Metres to the nearest real wall face -- clearance.wall_face_field."""
        c = int((x - self.ox) / self.res)
        r = int(self.H - 1 - (y - self.oy) / self.res)
        if 0 <= r < self.H and 0 <= c < self.W:
            return float(self.faces[r, c])
        return 0.0

    def __call__(self, x, y):
        """Metres to the nearest no-go -- mapped or seen. Same shape as
        clearance_test()'s closure, which is why nothing downstream changed."""
        c = int((x - self.ox) / self.res)
        r = int(self.H - 1 - (y - self.oy) / self.res)
        s = float(self.dist[r, c]) if (0 <= r < self.H and 0 <= c < self.W) else 0.0
        if self.local is None:
            return s
        edt, x0, y0 = self.local
        lc = int((x - x0) / self.res)
        lr = int((y - y0) / self.res)
        if 0 <= lr < self.n and 0 <= lc < self.n:
            return min(s, float(edt[lr, lc]))
        return s


def line_clear(a, b, clearance, need=DESTINATION_CLEAR, step=0.1, skip=SKIP):
    """Does the straight line from a to b keep `need` metres of room?

    The first `skip` metres are not tested. Where the dog is standing is a
    fact, not a choice: once it has crept to within `need` of something, a test
    that starts at its feet fails for every candidate goal on the map and the
    dog is stuck deciding that nowhere is reachable, including where it came
    from. What matters is whether the way *ahead* is clear.
    """
    d = math.dist(a, b)
    n = max(int(d / step), 1)
    first = int(skip / step) if d > skip else 0
    for k in range(first, n + 1):
        x = a[0] + (b[0] - a[0]) * k / n
        y = a[1] + (b[1] - a[1]) * k / n
        if clearance(x, y) < need:
            return False
    return True


def in_open_floor(a, b, live, need=STATIC_MIN, step=0.1, skip=SKIP):
    """`b`, or the last point on the line a -> b still in mapped open floor.

    Everything below reasons about straight lines, and a straight line is only
    a description of what the dog can do while it stays inside the building.
    A route that turns into a doorway breaks that: `pick_destination` hands
    back a point 2-4 m along the route, and once the dog is within 2 m of the
    turn that point is *past* it -- several metres into the room, behind a
    wall. The tightest spot on the line to it is then the wall, every sideways
    offset from that pinch is inside the building, and free_destination reports
    "no way past" in a corridor with a clear metre down one side. Measured on
    `chemistry lab`: stopped at (42.4, 9.7) with 0.68 m of room all round it.

    So the geometry is done against the part of the line the dog could
    actually drive, which at a turn is the approach to it. The rest of the
    route has not gone anywhere -- the next tick, a metre further on, asks
    again from there.

    Mapped structure only. A crate is exactly what the caller is here to go
    round, and clamping to it would hide it.
    """
    d = math.dist(a, b)
    n = max(int(d / step), 1)
    first = int(skip / step) if d > skip else 0
    for k in range(first, n + 1):
        t = k / n
        p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if live.static_at(*p) < need:
            t = max(k - 1, first) / n
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    return b


def free_destination(destination, xy, live, probe=None, need=DESTINATION_CLEAR,
                     max_offset=MAX_OFFSET, step=OFFSET_STEP, prefer=0.0):
    """The goal to aim at, keeping to the side the dog already committed to.

    Thin wrapper over `_free_destination`, which does the work. A preference
    biases the answer and must never *be* the answer "no way past": the dog
    stopped a stride short of a hallway destination tucked behind a trolley
    because the side it had leant towards had nothing reachable left on it,
    while the other side did. So when the committed side comes back
    empty-handed, both are judged again.
    """
    aim, off = _free_destination(destination, xy, live, probe, need,
                                 max_offset, step, prefer)
    if aim is None and prefer:
        aim, off = _free_destination(destination, xy, live, probe, need,
                                     max_offset, step, 0.0)
    return aim, off


def _free_destination(destination, xy, live, probe=None, need=DESTINATION_CLEAR,
                max_offset=MAX_OFFSET, step=OFFSET_STEP, prefer=0.0):
    """The goal to actually aim at: `destination`, or a point beside it with room.

    Returns (point, offset). Offset is signed and 0.0 when nothing moved, and
    the point is None when there is no way past at all -- the caller's cue to
    crawl and then stop, not to invent one.

    Two questions, answered in two different places, and conflating them is
    what made every earlier version of this fail.

    *Which way round, and how far* is answered at the pinch point: the tightest
    spot on the line from the dog to `probe`, which is where the obstacle
    actually is. Not at the probe itself -- that sits beyond the crate in open
    corridor, where "which side has more room" answers "the middle", and the
    dog is advised to carry straight on into the thing it is trying to avoid.
    Room alone decides it there, no straight line. A straight line from the
    dog to a point beyond a crate cannot both clear the crate and stay inside a
    2.15 m corridor: getting past needs a curve, and demanding a clear straight
    line to the far side just reports "no way past" in a corridor that plainly
    has one. Judging it out there is also what makes the dog start easing
    across while it still has open floor to do it in; judged at the destination,
    nothing looks blocked until the dog is level with the crate and every
    diagonal into the gap shaves the corner.

    *Where to aim now* is answered at the destination, 2-4 m out, where a straight
    line is a fair description of the next few seconds. The sideways shift
    found above is applied there and backed off until the line to it is clear.
    The dog eases over a little each tick and arrives at the gap lined up,
    which is also what the person holding the handle would want: one gentle
    lean, not a swerve at the last moment.

    Sideways, not forwards: the route is still the route and the dog is still
    going to the same place. Both sides are tried and the roomiest wins, so it
    slips past on whichever side has space rather than by convention -- and the
    roomiest rather than the nearest, because the smallest workable shift puts
    the goal hard against the crate and the dog arrives with nowhere to go.

    `prefer` is the side the dog has already committed to, +1 or -1, and it is
    tried alone before both are. Which side is roomiest is judged afresh every
    tick from a scan that changes as the dog closes in, and near a crate on a
    turn the answer flips: level with the floor-2 crate, "roomiest" alternated
    between a point 0.7 m north of the pinch and one 0.9 m south of it, tick
    about. The dog had already leant north, and being sent south every other
    tick walked it into the crate it was going round. Committing is also what
    the person on the handle is owed -- they were told which way it was
    stepping. The commitment is a preference, not an override: if nothing on
    that side is reachable any more, both sides are judged again.
    """
    # Nothing standing in the way, so there is nothing to go round: the
    # planner's route is still the route. `still` rather than `local` -- a
    # person walking across it is not a reason to re-route, it is a reason to
    # wait, and that decision is made in run_building.
    if live.still is None:
        return destination, 0.0

    # Both the far point the detour is judged at and the goal it is placed at
    # have to be inside the building -- see in_open_floor. Without this, a
    # destination around a corner turns every question below into a question
    # about a wall.
    far = in_open_floor(xy, probe if probe is not None else destination, live)
    goal = in_open_floor(xy, destination, live)

    def point_ok(p):
        """Room enough to stand, and still inside the building."""
        return live.detected_at(*p) >= need and live.static_at(*p) >= STATIC_MIN

    def reachable(p):
        return (point_ok(p) and line_clear(xy, p, live.detected_at, LINE_NEED)
                and through_no_wall(xy, p))

    def through_no_wall(a, b, skip=SKIP):
        """The line a -> b never enters a mapped wall. Standing in open floor
        is not enough: a point shifted 1.6 m sideways out of a corridor is
        open floor in the room behind the wall. Seed 8 of `room 201
        --pedestrians 3` aimed at (17.3, 12.1), 1.2 m past the floor-2 north
        wall, and walked into the wall for 6.5 s. Only entering the wall is
        refused, not coming near it, so it is asked of the real wall faces:
        against the inflated ones, a dog 0.3 m off-centre in a doorway found
        a wall on every line into the room and stopped there."""
        return line_clear(a, b, live.wall_at, 1e-6, skip=skip)

    if reachable(far) and reachable(goal):
        return destination, 0.0

    dx, dy = far[0] - xy[0], far[1] - xy[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return None, 0.0

    # Offsets are measured across the ROUTE, not across the dog's current
    # heading. Measured from the dog's own line they shrink as it leans into
    # them: lean 0.3 m and the line now points 0.3 m further over, so the
    # remaining offset reads 0.3 m smaller, and the dog converges on a course
    # that grazes the obstacle instead of one that clears it. Across the route
    # the target is a fixed place in the corridor and the crossing finishes.
    rdx, rdy = far[0] - goal[0], far[1] - goal[1]
    rd = math.hypot(rdx, rdy)
    if rd < 1e-6:
        rdx, rdy, rd = dx, dy, d
    nx, ny = -rdy / rd, rdx / rd        # unit normal to the route ahead

    # The pinch: where along the way the least room is, which is the thing
    # being gone around. Offsets are measured from there.
    steps = max(int(d / 0.1), 1)
    pinch, worst = far, INF
    for k in range(int(SKIP / 0.1), steps + 1):
        t = k / steps
        p = (xy[0] + dx * t, xy[1] + dy * t)
        room = live.detected_at(*p)
        if room < worst:
            pinch, worst = p, room

    # Which side, and how far across: both decided at the pinch, where the
    # obstacle is.
    def roomiest(signs):
        side, want, best_room = None, 0.0, -1.0
        off = step
        while off <= max_offset + 1e-9:
            for sign in signs:
                p = (pinch[0] + nx * off * sign, pinch[1] + ny * off * sign)
                if not point_ok(p) or not through_no_wall(pinch, p, skip=0.0):
                    continue
                room = min(live.detected_at(*p), live.static_at(*p))
                if room > best_room + 1e-9:
                    side, want, best_room = sign, off, room
            off += step
        return side, want, best_room

    side, want, best_room = (roomiest((math.copysign(1.0, prefer),)) if prefer
                             else (None, 0.0, -1.0))
    if side is None:
        side, want, best_room = roomiest((1.0, -1.0))
    if side is None:
        return None, 0.0

    # The destination goes as near that crossing as it can be reached from here.
    # Re-optimising "roomiest" at the destination instead is what made earlier
    # versions creep back: once the destination is clear of the obstacle the
    # roomiest place is the middle of the corridor, so the dog would aim for
    # the centreline, undo the crossing it had started, and arrive at the
    # obstacle exactly where it began. What matters is being across by the time
    # it reaches the pinch, so the pinch sets the distance and the destination only
    # says whether it can be got to.
    tries = sorted((k * step for k in range(1, int(max_offset / step) + 1)),
                   key=lambda o: abs(o - want))

    # Aim where the crossing has to be *finished*, which is the pinch -- not
    # where it would be finished if the dog had the whole way to the
    # destination to do it in. The dog drives at the point it is handed, so a
    # goal offset out at the destination is a shallow diagonal that is only
    # part-way across by the time the dog is level with the obstacle.
    # Measured on `chemistry lab`: told to cross 0.70 m, the dog reached the
    # cartons with 0.24 m in hand against the 0.20 m it insists on, and one
    # tick of noise either way was the difference between squeezing past and
    # reporting no way through. Crossing by the pinch leaves the whole 0.70 m.
    for o in tries:
        p = (pinch[0] + nx * o * side, pinch[1] + ny * o * side)
        if reachable(p):
            return p, o * side

    # Past the pinch, or nothing beside it can be reached from here: fall back
    # to offsetting the goal itself, which is the same manoeuvre judged over a
    # longer run-up.
    for o in tries:
        p = (goal[0] + nx * o * side, goal[1] + ny * o * side)
        if reachable(p):
            return p, o * side

    # Nothing on that side fits yet but the destination itself is still reachable:
    # keep going and ask again a metre later, rather than stopping on a
    # geometry that is about to change.
    return (destination, 0.0) if reachable(goal) else (None, 0.0)

"""CE-RRT*: a chance-constrained RRT* for walking round people (spec L6 s2).

Until this existed the dog stopped for anybody whose path met its own and
waited for them to clear (`run_building.yield_to`). Spec L6 wants a
pedestrian avoided without a full stop, and a goal moved sideways
(`perception.free_destination`) cannot express that: going round somebody
walking towards you is a curve, and the curve has to be where they will be,
not where they are. So this plans in space *and time*.

What it is, in three layers:

  RRT       grow a tree from the dog: sample a point, step towards it from
            the nearest node, keep the step if it is free.
  RRT*      give each new node the parent that makes it cheapest, then hand
            cheaper routes to its neighbours through it (rewiring), so the
            path straightens into a curve instead of a zig-zag.
  chance-   the dog walks at a fixed `v`, so every point on an edge has a
  constrained  time, and each person is checked *at that time*, at where their
            track predicts them (a straight line, tracking.Track.predict).

The chance constraint, and why it only covers half a second. Measured on the
twin (scripts/ce_rrt_calibrate.py, 12 runs, 2852 predictions): the straight
line is out by 0.32 m (99th percentile, along or across their path) half a
second ahead, 1.3 m one second ahead and 3.3 m two seconds ahead. People
pause, turn and step aside; a line cannot know. And the error is not
Gaussian: its tails are heavier, so a Gaussian's 2.33 sigma under-claims
the 1% point -- the margins below are the measured percentiles themselves.

So no plan can promise 1% two seconds out in a corridor, and none needs to.
The plan is replanned every REPLAN_S, and when no plan is safe the dog falls
back to waiting for the person (run_building.yield_to). What it commits to
is the next COMMIT_T: over that, P(closer than CONTACT) < 0.01, per person --
the spec's P_coll < 0.01 -- with the margin Q0 + Q_RATE * t, the measured 99th
percentile. Beyond it, out to PREDICT_T, the plan keeps the separation it
will need when that moment is committed to, from where they are predicted
to be: an intention, not a guarantee, and what stops a plan that is fine
now becoming impossible a tick later. Beyond PREDICT_T nobody is checked.

Everything that is not a person -- the map, the crates, the person on the
handle -- arrives as one callback, `ok(x, y, yaw)`, so the rules that already
keep those safe in `run_building` are the rules here too, unchanged.

Deterministic on purpose: a fixed number of samples from a seeded generator.
Gate D needs a run to repeat tick for tick, and a time budget would make the
plan depend on how busy the machine was.

Warm start: the remainder of the last plan is offered first, ahead of any
random sample, so a plan that is still good is found again at once and the
dog does not change its mind about which side it is passing on.
"""
import math
from typing import NamedTuple

import numpy as np

V = 0.40                # m/s the plan is walked at -- run_building.PASS_V
SPEEDS = (V, 0.25)      # ...tried in turn: slower changes when the dog meets
                        # somebody, and walking slowly past is still not a stop
STEP = 0.50             # metres, longest edge
MAX_TURN = math.radians(50)   # between consecutive edges: a curve, not a pivot.
                        # A pivot swings the person on the handle sideways
                        # (run_building.HANDLER_BEHIND), which is what the
                        # whole turn-easing machinery exists to prevent.
FIRST_TURN = math.radians(70)  # off the dog's own heading, for the first edge.
                        # They bind about half the plans that find nothing, but
                        # 65 and 90 cut corners close enough to graze the
                        # cafeteria door frame twice (0.1 s each) and clip a
                        # crate's corner between checks in the unit tests.
CHECK = 0.10            # metres between collision checks along an edge...
REL_CHECK = 0.05        # ...and closer, where people are: metres the gap to a
                        # person may change between two checks. Spaced by
                        # distance alone, a person closing at 1 m/s moved
                        # 0.25 m between checks and the path slipped inside
                        # the margin between them.
ITERS = 600             # samples per plan: 19 of 20 seeds find a head-on pass
                        # (300: 12 of 20), in under 35 ms -- spec L6, >= 10 Hz
GOAL_BIAS = 0.20        # fraction of samples that are the goal itself
GOAL_TOL = 0.40         # metres from the goal that counts as reaching it
NEIGHBOUR_R = 0.90      # metres: RRT*'s choose-parent and rewire radius
MARGIN = 2.0            # metres the sampling box extends past start and goal

CONTACT = 0.55          # metres, dog centre to a person's: run_building.PED_NEAR,
                        # what Gate D D3 scores as contact.
Q0 = 0.22               # metres: the 99th percentile of a track's error now
Q_RATE = 0.84           # ...growing this much per second of prediction:
                        # 0.64 m at 0.5 s (along their path; across it, 0.45).
                        # Measured, scripts/ce_rrt_calibrate.py. A straight
                        # fit, good to COMMIT_T and not beyond: at 1 s the
                        # data says 1.33 m and the line 1.06 m.
COMMIT_T = 0.5          # seconds held to the chance constraint: the next few
                        # replans (REPLAN_S) and a stop.
REPLAN_S = 0.10         # seconds between plans, for the caller: spec L6 asks
                        # >= 10 Hz, and a plan costs 15-40 ms
PREDICT_T = 3.0         # seconds of prediction looked at at all
HANDLER_SEP = 0.55      # metres from the person on the handle to anybody. No
                        # chance margin: they are behind the dog, so a person
                        # who reaches them has already been passed.
COMFORT = 1.6           # metres: closer than this to anybody costs extra,
W_NEAR = 1.0            # ...this much per metre walked at zero separation


class Person(NamedTuple):
    """A tracked person, as the planner needs them."""
    x: float
    y: float
    vx: float
    vy: float
    r: float = 0.25


def need(t):
    """Separation from a person's predicted centre at time `t`: P(contact)
    < 0.01 up to COMMIT_T, and that separation held from there on."""
    return CONTACT + Q0 + Q_RATE * min(t, COMMIT_T)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Plan(NamedTuple):
    path: list          # [(x, y), ...], starting at the dog
    cost: float
    nodes: int          # tree size, for the log
    side: float         # +1 passed on the left of the straight line, -1 right
    v: float = V        # m/s it was timed at, and must be walked at


class _Planner:
    def __init__(self, start, goal, ok, people, handler_behind, v):
        self.sx, self.sy, self.syaw = start
        self.goal = goal
        self.ok = ok
        self.people = people
        self.behind = handler_behind
        self.v = v
        fastest = max((math.hypot(p.vx, p.vy) for p in self.people), default=0.0)
        self.check = (min(CHECK, REL_CHECK * v / (v + fastest)) if self.people
                      else CHECK)
        self.xy = [(self.sx, self.sy)]
        self.yaw = [self.syaw]
        self.parent = [-1]
        self.cost = [0.0]
        self.length = [0.0]     # metres walked to the node: its time is length / v

    # -- checks -----------------------------------------------------------
    def _point(self, x, y, yaw, t):
        """(free, penalty) for the dog at (x, y) heading `yaw` at time `t`."""
        if not self.ok(x, y, yaw):
            return False, 0.0
        if t > PREDICT_T or not self.people:
            return True, 0.0
        hx, hy = x - self.behind * math.cos(yaw), y - self.behind * math.sin(yaw)
        sep = need(t)
        near = INF
        for p in self.people:
            px, py = p.x + p.vx * t, p.y + p.vy * t
            d = math.hypot(x - px, y - py)
            if d < sep:
                return False, 0.0
            if math.hypot(hx - px, hy - py) < HANDLER_SEP:
                return False, 0.0
            near = min(near, d)
        return True, max(0.0, COMFORT - near) / COMFORT

    def edge(self, a, yaw_a, len_a, b):
        """Can the dog walk the straight line a -> b, leaving a at metre
        `len_a` of its walk with heading `yaw_a`? (free, cost, heading)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return False, 0.0, yaw_a
        yaw = math.atan2(dy, dx)
        limit = FIRST_TURN if len_a == 0.0 else MAX_TURN
        if abs(_wrap(yaw - yaw_a)) > limit:
            return False, 0.0, yaw
        n = max(int(math.ceil(d / self.check)), 1)
        pen = 0.0
        for k in range(1, n + 1):
            s = d * k / n
            free, p = self._point(a[0] + dx * k / n, a[1] + dy * k / n, yaw,
                                  (len_a + s) / self.v)
            if not free:
                return False, 0.0, yaw
            pen += p
        return True, d + W_NEAR * pen * d / n, yaw

    # -- tree -------------------------------------------------------------
    def add(self, q):
        """Try to grow the tree towards `q`. Index of the new node, or None."""
        pts = np.asarray(self.xy)
        dist = np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1])
        near_i = int(np.argmin(dist))
        nx, ny = self.xy[near_i]
        d = float(dist[near_i])
        if d < 1e-6:
            return None
        if d > STEP:
            q = (nx + (q[0] - nx) * STEP / d, ny + (q[1] - ny) * STEP / d)
            dist = np.hypot(pts[:, 0] - q[0], pts[:, 1] - q[1])
        # Choose parent: cheapest first, and stop at the first edge that is
        # free -- the expensive part is the check, not the sort.
        hood = np.nonzero(dist <= NEIGHBOUR_R)[0].tolist()
        if near_i not in hood:
            hood.append(near_i)
        hood.sort(key=lambda j: self.cost[j] + float(dist[j]))
        best = None
        for j in hood:
            free, c, yaw = self.edge(self.xy[j], self.yaw[j], self.length[j], q)
            if free:
                best = (j, self.cost[j] + c, yaw, self.length[j] + float(dist[j]))
                break
        if best is None:
            return None
        j, c, yaw, length = best
        k = len(self.xy)
        self.xy.append(q)
        self.yaw.append(yaw)
        self.parent.append(j)
        self.cost.append(c)
        self.length.append(length)
        # Rewire: route a neighbour through the new node where that is cheaper.
        # Its subtree's times shift earlier with it; `extract` checks the
        # whole path again at its real times before anything is returned.
        for m in hood:
            if m == j or m == 0:
                continue
            free, cm, yaw_m = self.edge(q, yaw, length, self.xy[m])
            if free and c + cm < self.cost[m] - 1e-9:
                self.parent[m] = k
                self.cost[m] = c + cm
                self.yaw[m] = yaw_m
                self.length[m] = length + math.dist(q, self.xy[m])
        return k

    def branch(self, k):
        out = []
        while k != -1:
            out.append(self.xy[k])
            k = self.parent[k]
        return out[::-1]

    def valid(self, path):
        """The whole path, walked from the start at `v`, checked at its real
        times and turns. (free, cost)."""
        yaw, length, total = self.syaw, 0.0, 0.0
        for a, b in zip(path, path[1:]):
            free, c, yaw = self.edge(a, yaw, length, b)
            if not free:
                return False, 0.0
            length += math.dist(a, b)
            total += c
        return True, total

    def shortcut(self, path):
        """Drop every point the path can do without. Greedy and
        deterministic: from each kept point, the furthest one reachable
        straight -- at the time the dog would get there."""
        out = [path[0]]
        yaw, length, i = self.syaw, 0.0, 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                free, _, y2 = self.edge(path[i], yaw, length, path[j])
                if free:
                    break
            else:
                return path                   # cannot happen for a valid path
            length += math.dist(path[i], path[j])
            yaw, i = y2, j
            out.append(path[j])
        return out


INF = 99.0


def plan(start, goal, ok, people=(), warm=None, handler_behind=1.1, v=V, seed=0,
         iters=ITERS):
    """A path from `start` = (x, y, yaw) to within GOAL_TOL of `goal`, or None.

    `ok(x, y, yaw)` is everything static: True where the dog may stand with
    that heading. `people` are `Person`s, predicted in straight lines.
    `warm` is the last plan's path, tried first."""
    P = _Planner(start, goal, ok, list(people), handler_behind, v)
    if not ok(*start):
        # Already somewhere the static rules reject -- squeezed past a crate,
        # inside a doorway's inflated band. Planning from there would find
        # nothing; the caller's ordinary rules know what to do.
        return None
    rng = np.random.default_rng(seed)
    gx, gy = goal
    lo = (min(P.sx, gx) - MARGIN, min(P.sy, gy) - MARGIN)
    hi = (max(P.sx, gx) + MARGIN, max(P.sy, gy) + MARGIN)
    # The last plan, cut to STEP-long pieces: `add` steps at most STEP towards
    # a sample, so a shortcut plan's 2 m edges offered whole were each one
    # half-metre stub, and the plan was not found again.
    queue = []
    for a, b in zip(warm or [], (warm or [])[1:]):
        n = max(int(math.ceil(math.dist(a, b) / STEP)), 1)
        queue += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                  for k in range(1, n + 1)]
    for _ in range(iters):
        if queue:
            q = queue.pop(0)
        elif rng.random() < GOAL_BIAS:
            q = goal
        else:
            q = (rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]))
        P.add(q)
    reached = [k for k, p in enumerate(P.xy) if math.dist(p, goal) <= GOAL_TOL]
    for k in sorted(reached, key=lambda k: P.cost[k]):
        path = P.branch(k)
        free, _ = P.valid(path)
        if not free:
            continue
        path = P.shortcut(path)
        free, cost = P.valid(path)
        if not free:
            continue
        # Which side of the straight line it went: what the person on the
        # handle is told before the dog leans.
        ux, uy = gx - P.sx, gy - P.sy
        lat = max((-(px - P.sx) * uy + (py - P.sy) * ux for px, py in path), key=abs)
        return Plan(path, cost, len(P.xy),
                    math.copysign(1.0, lat) if abs(lat) > 1e-6 else 0.0, v)
    return None


def still_good(start, path, ok, people=(), handler_behind=1.1, v=V):
    """Is the rest of a plan still safe from `start` = (x, y, yaw), against
    where people are predicted to be now? The dog walks from where it is to
    `path[1]` and on. A replan that finds nothing does not mean the plan being
    walked has stopped being safe -- 600 samples miss a narrow gap now and
    then -- and dropping it there made the dog stop, plan, stop again."""
    if len(path) < 2 or not ok(*start):
        return False
    P = _Planner(start, path[-1], ok, list(people), handler_behind, v)
    free, _ = P.valid([(start[0], start[1])] + list(path[1:]))
    return free


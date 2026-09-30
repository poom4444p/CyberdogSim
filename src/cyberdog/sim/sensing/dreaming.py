"""What would happen if we actually followed that path?

VAMOS returns five candidate paths drawn on a camera frame. The old gate asked
one question of each -- does the drawn line stay in free space -- and that is
not the question. The dog does not teleport along the line; it chases it with
a pure-pursuit controller, at a metre a second, with its heading estimate a
few degrees out and its feet slipping. A line that threads a doorway with two
centimetres to spare is free space and is still the wrong path to take.

So before committing to a candidate, the dog imagines driving it. The same
control law the real loop uses is rolled forward from the current pose, many
times, with noise on the things that are actually uncertain -- heading bias,
speed, per-step slip. Some of those imagined runs hit something. The fraction
that do is a collision probability, and that is the safety factor:

    safety = P(no collision) x (how much room it left)

which is the spec's chance constraint (CE-RRT*, P_coll < 0.01) arrived at by
sampling instead of by algebra, and it feeds the same confidence gate the
affordance MLP will feed later.

Two honest limits. The rollouts use the kinematic model the twin is built on,
so they inherit its optimism -- no gait, no contacts, no real dynamics. And
the clearance field they read is built from grids that were already inflated
by the robot's radius, so the distances come out about 0.25 m short of the
truth. Both err towards caution, which is the direction to err in.
"""
import math
import random

# Controller being imagined. These are the real loop's numbers (control.py);
# a dream of a controller the dog does not have would tell us nothing.
from cyberdog.sim.control import FOLLOW_D, K_W, TURN_ONLY

DT = 0.05                       # 20 Hz, the control rate

N_DREAMS = 24                   # rollouts per candidate
ARRIVED_R = 0.2                 # close enough to the path's end to stop early

# How long to imagine. Long enough to walk the whole path, plus an allowance
# for turning onto it -- a fixed horizon lets a rollout that starts by pivoting
# run out of time before it reaches the dangerous end of the path, and a run
# that never arrives anywhere would otherwise score as perfectly safe.
TURN_ALLOWANCE_S = 2.5
HORIZON_SLACK = 1.6             # walking is never as direct as the polyline
HORIZON_CAP_S = 12.0

# What is uncertain, and by how much. Heading bias and speed scale are drawn
# once per rollout (a systematic error that persists); slip is redrawn every
# step (noise that accumulates as a random walk).
YAW_BIAS_SD = 0.06              # rad, ~3.5 deg of heading estimate error
SPEED_SD = 0.10                 # fraction of commanded speed
SLIP_SD = 0.02                  # rad per step

# Thresholds are in the clearance field's own units, which are conservative:
# the grids were already inflated by the robot's radius, so a doorway the dog
# fits through comfortably reads as about 0.25 m and an open corridor as 1.0.
# ROOMY is set so a doorway passes the gate but scores well below a corridor
# -- which is right, a doorway *is* the tighter thing to walk through, and the
# speed scaling should slow the dog down for it.
ROBOT_R = 0.16                  # clearance below this counts as a collision
ROOMY = 0.40                    # clearance at or above this scores full marks
FLOOR = 0.40                    # score of a run that survives but scrapes

# Room is the closest approach, but not to where the dog already is. From a
# pose next to a wall, the closest point of every walk is the first one, which
# every candidate shares -- so the one that steers away from the wall used to
# tie with the one that hugs it. Inside START_R of the pose, only getting
# closer than the start counts; past it, everything does. Collisions are
# still checked on every step.
START_R = 0.5                   # metres, about a body length

# The person on the handle, when the caller says where they are (`person`):
# a rollout that brings them within PERSON_R of a surface -- they are 0.25 m
# across the shoulders, so that is touching -- is a collision, the same as the
# dog's own body. Without this a path could be safe for the dog and still swing
# its user into a trolley: Gate D, `room 101` seed 1, 1.6 s against the floor-1
# trolley on a VAMOS path the gate had passed. As with the dog's start, a person
# who begins closer than that only counts as hit by getting closer still.
PERSON_R = 0.25


class Dream:
    """Imagined rollouts of one candidate path, scored for safety."""

    def __init__(self, clearance, max_v, n=N_DREAMS, seed=0, person=None):
        self.clearance = clearance          # (x, y) -> metres to the nearest no-go
        self.person = person                # (x, y, yaw) -> room around the person
        self.max_v = max_v
        self.n = n
        self.rng = random.Random(seed)      # seeded: the same frame dreams the same

    def factor(self, path, pose):
        """Safety factor in [0, 1] for following `path` from `pose`.

        Also returns the pieces, because a single number nobody can take apart
        is not much use when a path is rejected and someone has to find out why.
        """
        if len(path) < 2:
            return 0.0, {"p_safe": 0.0, "margin": 0.0, "reason": "path too short"}

        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        steps = int(min(TURN_ALLOWANCE_S + HORIZON_SLACK * length / self.max_v,
                        HORIZON_CAP_S) / DT)

        survived, stalled, margins = 0, 0, []
        for _ in range(self.n):
            ok, min_clear, arrived = self._roll(path, pose, steps)
            # Not arriving is not the same as arriving safely. A rollout that
            # was still going when time ran out has not been shown to be safe,
            # so it does not get counted as if it had.
            if ok and arrived:
                survived += 1
                margins.append(min_clear)
            elif ok:
                stalled += 1

        p_safe = survived / self.n
        margin = sum(margins) / len(margins) if margins else 0.0
        room = min(max((margin - ROBOT_R) / (ROOMY - ROBOT_R), 0.0), 1.0)
        reason = ""
        if not survived:
            reason = ("never got to the end of it in time" if stalled
                      else "every imagined run hit something")
        return p_safe * (FLOOR + (1 - FLOOR) * room), {
            "p_safe": p_safe, "margin": margin, "stalled": stalled / self.n,
            "reason": reason,
        }

    def _roll(self, path, pose, steps):
        """One imagined run: (never hit anything, closest approach, got there)."""
        x, y, yaw = pose
        yaw_bias = self.rng.gauss(0.0, YAW_BIAS_SD)
        speed = self.max_v * (1.0 + self.rng.gauss(0.0, SPEED_SD))
        start, start_clear = (x, y), self.clearance(x, y)
        min_clear = float("inf")
        person_floor = (min(PERSON_R, self.person(x, y, yaw)) - 1e-6
                        if self.person else None)

        # Index of the point being chased. It only ever moves forward: a
        # lookahead that re-scans the whole path from the start will, once the
        # dog is more than a lookahead past an early point, turn round and
        # chase it. The live loop never gets there because it replans every
        # second, but a ten-second dream would spend it oscillating.
        i = 0
        for _ in range(steps):
            while i < len(path) - 1 and math.dist(path[i], (x, y)) < FOLLOW_D:
                i += 1
            tx, ty = path[i]
            err = _wrap(math.atan2(ty - y, tx - x) - yaw + yaw_bias)
            # Same law as the real loop: pivot when badly misaligned, else walk.
            v = 0.0 if abs(err) > TURN_ONLY else speed * math.cos(err)
            yaw += (K_W * err) * DT + self.rng.gauss(0.0, SLIP_SD)
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT

            clear = self.clearance(x, y)
            if clear < start_clear or math.dist(start, (x, y)) >= START_R:
                min_clear = min(min_clear, clear)
            if clear < ROBOT_R:
                return False, min_clear, False
            if self.person and self.person(x, y, yaw) < person_floor:
                return False, min_clear, False
            # A walk that ends before leaving START_R without closing in has
            # nothing counted: it kept the room it started with.
            if math.dist((x, y), path[-1]) < ARRIVED_R:
                return True, _or(min_clear, start_clear), True
        return True, _or(min_clear, start_clear), False


def _or(v, default):
    return v if math.isfinite(v) else default


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

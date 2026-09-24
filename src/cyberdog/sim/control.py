"""The control law and the route it follows -- the parts everything shares.

These constants and helpers used to live in `run_demo.py`, and `dreaming.py`,
`run_building.py` and `record_demo.py` all reached into that script to get them
(`from run_demo import K_W, TURN_ONLY`). That made a demo entry point a library
three other modules depended on: running `run_demo.py` and importing the gains
were the same act, and the imagined rollouts in `dreaming.py` could not be read
without first working out which of `run_demo`'s two dozen lines were the
controller and which were the demo around it.

The gains matter more than the tidiness. `dreaming.py` imagines the dog driving
a candidate path, and a dream of a controller the dog does not have tells us
nothing -- so the rollouts and the live loop must read the *same* numbers. Here
they do, by construction.
"""
import math

from cyberdog.planning.building_router import BuildingRouter, NoAccessibleRoute

START_XY = (45.0, 4.0)          # main entrance, floor 1

# --- the control law -------------------------------------------------------
K_W = 2.0                       # rad/s of yaw rate per rad of heading error
TURN_ONLY = 0.8                 # above this error, pivot instead of walking
FOLLOW_D = 1.5                  # pure-pursuit lookahead along a path, metres

# --- when a waypoint or a goal counts as reached ---------------------------
ARRIVE_R = 0.15                 # waypoint reached inside this -- keep it well
                                # under half a door width (doors are 1.2 m) or the
                                # dog turns early and clips the frame
GOAL_R = 0.5


def wrap(a):
    """An angle folded into (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def advance(i, x, y, waypoints):
    """Move to the next waypoint once we are near it or have driven past it.

    Distance alone is not enough: overshoot a waypoint by more than ARRIVE_R
    and the dog turns back for it, overshoots again, and spins forever.
    Projecting onto the segment catches the overshoot.
    """
    while i < len(waypoints) - 1:
        px, py = waypoints[i - 1]
        cx, cy = waypoints[i]
        sx, sy = cx - px, cy - py
        seg = sx * sx + sy * sy
        t = ((x - px) * sx + (y - py) * sy) / seg if seg > 1e-9 else 1.0
        if t >= 1.0 or math.hypot(cx - x, cy - y) < ARRIVE_R:
            i += 1
        else:
            break
    return i


def path_target(path, xy, d=FOLLOW_D):
    """The point `d` metres along `path` to steer at -- pure pursuit's lookahead point.

    Used for the paths VAMOS proposes, which arrive as a handful of points a
    metre or two apart rather than as a waypoint list.
    """
    for p in path:
        if math.dist(p, xy) >= d:
            return p
    return path[-1]


def plan_route(destination, start_xy=START_XY, floor=1):
    """Waypoints to `destination` on this one floor.

    The single-floor scenes (floor1.xml) have no lift and no other storeys,
    so a destination elsewhere is sent to run_building.py rather than having
    its legs from different floors strung together on this one. A refusal
    (the stairs) is the sentence the user would hear, not a traceback.
    """
    router = BuildingRouter()
    try:
        target = router.resolve(destination, floor, start_xy, floor=floor)
    except NoAccessibleRoute as refusal:
        raise SystemExit(f"[voice] {refusal}")
    if target is None:
        floors = router.floors_of(destination)
        if floors:
            raise SystemExit(f"{destination} is on floor(s) {', '.join(map(str, floors))}, "
                             f"not floor {floor}; this demo drives one floor -- "
                             f"use: python -m cyberdog.sim.run_building")
        raise SystemExit(f"unknown destination: {destination}")
    legs = router.plan(floor, start_xy, target)
    if not legs:
        raise SystemExit("no route found")
    return [(float(cp.position[0]), float(cp.position[1]))
            for _, route in legs for cp in route.checkpoints]

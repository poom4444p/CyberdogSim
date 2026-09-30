"""Clearance: metres to the nearest thing the dog must not touch.

Every safety decision in the twin reads one number, and this is where the
static half of it comes from -- the occupancy grid's walls plus the Behavior
Layer's full-stop zones, fused into a single distance field.

This module exists because that field used to live in `record_demo.py`, the
script that renders the MP4. So `sensing/perception.py` -- the live obstacle
layer that decides whether the dog walks into a crate -- began with
`from record_demo import clearance_field`, and the safety-critical path
depended on a video recorder. Importing the clearance field pulled in imageio,
argparse and a `main()`, and anyone tracing how a stop decision gets made had
to read the recorder first.

Who reads what:

    clearance_test(floor)   a closure, for one-point-at-a-time callers
    clearance_field(floor)  the raw array, for perception.py's vectorised
                            thousand-return lookups
    free_space_test(floor)  the closure with the robot's radius thresholded on
                            it -- stands in for VAMOS's affordance MLP

The field is cached per floor: building it is a distance transform over a
385x964 grid and costs about 125 ms, which no 20 Hz loop can afford twice.
"""
from cyberdog import paths
from cyberdog.sim.scene.build_scene import load_grid

ROBOT_RADIUS = 0.16

_BEHAVIOR = None
_CLEARANCE = {}


def _behavior():
    """The building's Behavior Layer, loaded once.

    Same zones.json and same config the planner reads, so the gate and the
    router cannot disagree about where the no-go areas are.
    """
    global _BEHAVIOR
    if _BEHAVIOR is None:
        from cyberdog.mapping.behavior_layer import BehaviorLayer
        _BEHAVIOR = BehaviorLayer.from_config(paths.MAP_CONFIG,
                                              zones_path=paths.ZONES_JSON)
    return _BEHAVIOR


def clearance_field(floor=1):
    """The raw distance field: (dist, res, ox, oy), metres per cell.

    perception.py wants it whole so it can test a thousand LiDAR returns at
    once instead of calling a closure a thousand times; everything else should
    use clearance_test().
    """
    if floor not in _CLEARANCE:
        from scipy import ndimage

        occ, res, ox, oy = load_grid(floor)
        blocked = occ.copy()
        H, W = blocked.shape
        for z in _behavior().zones:
            if z.action_rules.speed_modifier > 0.0:
                continue
            xs = [p[0] for p in z.polygon]
            ys = [p[1] for p in z.polygon]
            for r in range(max(int(H - 1 - (max(ys) - oy) / res), 0),
                           min(int(H - 1 - (min(ys) - oy) / res) + 1, H)):
                for c in range(max(int((min(xs) - ox) / res), 0),
                               min(int((max(xs) - ox) / res) + 1, W)):
                    if z.contains_point(ox + (c + 0.5) * res,
                                        oy + (H - 1 - r + 0.5) * res):
                        blocked[r, c] = True
        # The lift shaft's walls, which the scene has and the grid does not --
        # grown by the grid's own inflation, so they read like every other wall.
        import yaml
        from cyberdog.sim.scene.lift import SHAFT_WALLS
        with open(paths.MAP_CONFIG) as f:
            grow = float(yaml.safe_load(f)["map"].get("grid_inflation", 0.0))
        for x0, x1, y0, y1 in SHAFT_WALLS.values():
            r0 = max(int(H - 1 - (y1 + grow - oy) / res), 0)
            r1 = min(int(H - 1 - (y0 - grow - oy) / res) + 1, H)
            c0 = max(int((x0 - grow - ox) / res), 0)
            c1 = min(int((x1 + grow - ox) / res) + 1, W)
            blocked[r0:r1, c0:c1] = True
        # Distance from every free cell to the nearest blocked one, in metres.
        _CLEARANCE[floor] = (ndimage.distance_transform_edt(~blocked) * res,
                             res, ox, oy)
    return _CLEARANCE[floor]


def clearance_test(floor=1):
    """(x, y) -> metres to the nearest thing the dog must not touch.

    Walls from the grid *and* the Behavior Layer's full-stop zones, in one
    field. The grid is flat -- built from the floor mesh, it knows nothing
    about the stairwell standing on it -- so without the zones a candidate
    path pointing straight into the flight would read as wide open.

    A distance, not a yes/no, because "how close did that come?" is the
    question a safety factor is made of; free_space_test() is this field with
    a threshold on it, and dreaming.py reads the distance directly. Outside
    the map reads as zero clearance.
    """
    dist, res, ox, oy = clearance_field(floor)
    H, W = dist.shape

    def clearance(x, y):
        c = int((x - ox) / res)
        r = int(H - 1 - (y - oy) / res)
        if not (0 <= r < H and 0 <= c < W):
            return 0.0
        return float(dist[r, c])
    return clearance


def free_space_test(floor=1, robot_radius=ROBOT_RADIUS):
    """(x, y) -> can the dog stand here? Stands in for the affordance MLP.

    The clearance field with the robot's own radius as the threshold. The
    trained MLP will see the elevation and reach the same answer about the
    stairs; until then, this is where that answer comes from.
    """
    clearance = clearance_test(floor)
    return lambda x, y: clearance(x, y) >= robot_radius

"""Keep to one side of the corridor, the way the people around the dog do.

A* returns the shortest line, and the shortest line hugs corners: out of the
main entrance the next waypoint was 41 m down the corridor on the edge of the
free space, so the dog walked the length of it with its legs in the wall.
Pedestrians do not walk like that, and a guide that does puts the person it
leads -- who follows behind on a rigid handle -- in the path of everybody
coming the other way. In Denmark, as in most places, you keep right.

So the dense A* path is moved sideways, point by point:

- in a corridor, to `wall_distance` from the wall on the keeping side;
- where there is not room for that twice over -- a doorway, a squeeze -- to
  the middle, which is the furthest from both walls there is;
- anywhere that is not a corridor -- no wall within REACH on either side, or
  walls further apart than `max_width` -- not at all. There is no lane to
  keep in a lobby or across a room, and pinning the route to a room's
  right-hand wall walked the dog two metres out of its way to reach a door.

Distances are measured on a copy of the grid with narrow openings bridged
(DOOR_BRIDGE). Otherwise every room door along the corridor reads as the wall
disappearing, and the lane would swing out towards it and back at each one.
The route still goes through doors; only the measuring pretends they are shut.
"""
import math

import numpy as np
from scipy import ndimage

DOOR_BRIDGE = 1.5   # metres: openings narrower than this are wall, for measuring
REACH = 3.0         # metres: no wall this close on the keeping side is open floor
SMOOTH = 1.0        # metres of path the shift is averaged over
EASE = 1.0          # metres at each end the shift fades in and out over
HEADING = 0.5       # metres either side a point's direction of travel is taken over


def doorway_cells(grid, walls):
    """Cells in a doorway: free in `grid`, but wall once openings are bridged."""
    return walls & (grid.grid == 0)


def _to_door(grid, doors, b, u, reach):
    """Metres from `b` along unit `u` to the first doorway cell within
    `reach`, or infinity."""
    step = grid.resolution / 2
    for k in range(int(reach / step) + 1):
        r, c = grid.world_to_grid(b[0] + u[0] * k * step, b[1] + u[1] * k * step)
        if 0 <= r < doors.shape[0] and 0 <= c < doors.shape[1] and doors[r, c]:
            return k * step
    return math.inf


def door_gates(points, grid, doors):
    """[(entry, exit, axis)] for every doorway the polyline walks through.

    `entry` and `exit` are where the segment crossing the doorway goes in and
    comes out of its cells, `axis` the unit direction of travel through it --
    the door's axis, since keep_side takes doorways through the middle.
    """
    gates = []
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        u = (b - a) / length
        d_in = _to_door(grid, doors, a, u, length)
        if math.isinf(d_in):
            continue
        d_out = _to_door(grid, doors, b, -u, length)
        gates.append((tuple(a + u * d_in), tuple(b - u * d_out), tuple(u)))
    return gates


def round_corners(points, says, grid, radius=1.5, spacing=0.25):
    """The polyline with each corner replaced by an arc, where it fits.

    The person follows on a rigid handle, straight behind the dog's heading,
    so on a turn of radius r they swing sqrt(r^2 + L^2) - r outside the dog's
    path: 1.1 m for a pivot, 0.3 m at r = 1.7 m. A polyline corner is a pivot,
    or close to one. Turning left out of the right-hand lane that put the
    person into the right-hand wall at every door across the corridor.

    Each corner gets an arc of `radius`, shortened to use no more than half
    of either segment next to it (so neighbouring arcs never overlap), and
    tightened, down to a sharp corner, until every arc point and the straight
    lines between them are free in `grid`. A corner's announcements move to
    the arc's first point: "turn left ahead" belongs where the turn starts.
    """
    if len(points) < 3:
        return list(points), [list(s) for s in says]
    out, out_says = [points[0]], [list(says[0])]
    for i in range(1, len(points) - 1):
        a, b, c = (np.asarray(p, float) for p in points[i - 1:i + 2])
        u, v = b - a, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        turn = math.acos(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0)) if lu and lv else 0.0
        arc = None
        if turn > math.radians(5):
            u, v = u / lu, v / lv
            t = min(radius * math.tan(turn / 2), lu / 2, lv / 2)
            for _ in range(4):
                arc = _fillet(b, u, v, t, turn, spacing)
                if all(_free_line(grid, p, q) for p, q in zip([out[-1]] + arc, arc + [tuple(c)])):
                    break
                t /= 2
                arc = None
        if arc is None:
            out.append(points[i])
            out_says.append(list(says[i]))
            continue
        out += arc
        out_says += [list(says[i])] + [[] for _ in arc[1:]]
    out.append(points[-1])
    out_says.append(list(says[-1]))
    return out, out_says


def _fillet(b, u, v, t, turn, spacing):
    """Points on the arc tangent to the lines into and out of corner `b`,
    `t` metres either side of it."""
    r = t / math.tan(turn / 2)
    p1 = b - u * t
    # Centre: from where the arc starts, a radius along the inside normal.
    left = u[0] * v[1] - u[1] * v[0] > 0
    n = np.array([-u[1], u[0]]) if left else np.array([u[1], -u[0]])
    centre = p1 + n * r
    a0 = math.atan2(p1[1] - centre[1], p1[0] - centre[0])
    sweep = turn if left else -turn
    k = max(int(abs(sweep) * r / spacing), 1)
    return [tuple(centre + r * np.array([math.cos(a0 + sweep * j / k),
                                         math.sin(a0 + sweep * j / k)]))
            for j in range(k + 1)]


def _free_line(grid, a, b):
    n = max(int(math.dist(a, b) / (grid.resolution / 2)), 1)
    return all(grid.is_free(*grid.world_to_grid(a[0] + (b[0] - a[0]) * j / n,
                                                 a[1] + (b[1] - a[1]) * j / n))
               for j in range(n + 1))


def bridge_doors(grid):
    """The grid's non-free cells, with openings under DOOR_BRIDGE closed."""
    blocked = grid.grid != 0
    n = int(math.ceil(DOOR_BRIDGE / 2 / grid.resolution))
    return ndimage.binary_closing(blocked, ndimage.generate_binary_structure(2, 2),
                                  iterations=n, border_value=1)


def _march(walls, grid, x, y, nx, ny, reach):
    """Free distance from (x, y) along (nx, ny) before `walls`, or None past `reach`."""
    step = grid.resolution
    for k in range(1, int(reach / step) + 1):
        r, c = grid.world_to_grid(x + nx * k * step, y + ny * k * step)
        if not (0 <= r < walls.shape[0] and 0 <= c < walls.shape[1]) or walls[r, c]:
            return k * step
    return None


def keep_side(path, grid, walls, side="right", wall_distance=0.35, max_width=3.0):
    """`path` (dense (x, y) points, as A* returns them) moved into the lane.

    `grid` is the planning grid and `walls` its bridge_doors(); distances,
    `wall_distance` and `max_width` included, are in that grid's own terms --
    from the edge of its inflated obstacles, not from the wall face. The two
    ends stay where they are: they are where the dog is and where it was asked
    to go.
    """
    if side not in ("right", "left") or len(path) < 3:
        return list(path)
    pts = np.asarray(path, float)
    # Arc length, since A*'s diagonal steps are longer than its straight ones.
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    total = s[-1]
    sign = 1.0 if side == "right" else -1.0

    # NaN where there is no lane to measure (open floor), filled in below.
    shift = np.full(len(pts), np.nan)
    shift[0] = shift[-1] = 0.0
    normal = np.zeros((len(pts), 2))
    for i, (x, y) in enumerate(pts):
        a = pts[np.searchsorted(s, s[i] - HEADING)]
        b = pts[min(np.searchsorted(s, s[i] + HEADING), len(pts) - 1)]
        dx, dy = b - a
        n = math.hypot(dx, dy)
        if n < 1e-9:
            continue
        # Right-hand normal to the direction of travel (left for side="left").
        nx, ny = sign * dy / n, -sign * dx / n
        normal[i] = nx, ny
        r, c = grid.world_to_grid(x, y)
        if walls[r, c]:
            shift[i] = 0.0                # in a doorway: through the middle of it
            continue
        near = _march(walls, grid, x, y, nx, ny, REACH)
        far = _march(walls, grid, x, y, -nx, -ny, REACH)
        if near is None or far is None or near + far > max_width:
            continue                      # not a corridor: no lane to keep
        # From the near wall: wall_distance where there is room, else the middle.
        shift[i] = near - min(wall_distance, (near + far) / 2)

    # Across open floor, blend from whatever comes before to whatever comes
    # after -- a doorway's middle to the lane, say. Dropping back to A*'s own
    # line there instead put an S-bend in the floor-1 lobby, from the entrance
    # door down to A*'s wall-hugging line and back up into the lane, and its
    # left turn swung the person on the handle into the north wall.
    known = ~np.isnan(shift)
    shift = np.interp(s, s[known], shift[known])

    # Averaged along the path, so the lane bends into a door rather than steps.
    shift = np.array([shift[(s >= s[i] - SMOOTH / 2) & (s <= s[i] + SMOOTH / 2)].mean()
                      for i in range(len(pts))])
    shift *= np.clip(np.minimum(s, total - s) / EASE, 0.0, 1.0)

    out = []
    for (x, y), (nx, ny), d in zip(pts, normal, shift):
        # Never into anything: back off towards A*'s own line until free.
        for _ in range(6):
            q = (x + nx * d, y + ny * d)
            if grid.is_free(*grid.world_to_grid(*q)):
                break
            d /= 2
        else:
            q = (x, y)
        out.append(q)
    return out

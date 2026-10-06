"""The elevation patch and the trial record: what the affordance MLP sees and learns.

One sample is one *trial*: the dog stands somewhere, is given a target a
metre or so away, and drives at it with the twin's own control law. What it
saw at the start (the patch), where it was told to go (the target), and what
happened (the outcome) are one record.

The patch
---------
A grid of terrain heights around the dog, aligned with its heading (+x
forward, +y left), RES metres a cell. Each cell holds the *highest* surface
seen in it, in metres above `ground_z` -- the ground the dog is standing on.
Highest, because both producers see the top of things: Isaac's height
scanner casts straight down, and a LiDAR return off a wall or a crate lands
on its face and top. Cells nothing landed in are NaN; the MLP's input layer
decides what to do with them, not this module.

The grid is longer in front than behind, because a target is always ahead
(TARGET_BEARING) and never further than the patch reaches (TARGET_D).

ground_z is passed in rather than taken from the patch, because the two
producers know it differently: in Isaac the scanner sees the ground under
the dog (`ground_under`), and the LiDAR cannot -- MIN_R and the dog's own body
hide it -- but the twin knows which floor it is on.

The label
---------
Recorded raw, labelled late. The collector stores what happened -- did it
arrive, did it fall, how far did the body tilt and rise -- and `classes()`
decides what it means. That is a judgement about a blind person on the
handle, not about the dog: a good policy climbs a flight of stairs without
falling, and the MLP must still say no to it. Keeping the judgement here
means changing a threshold is a retrain, not a recollection on a GPU.

Three answers, not two, because that is how a guide dog works: it does not
refuse every uneven pavement, it slows down and says so, and refuses only
what is dangerous.

    WALKABLE      smooth enough to walk as normal
    CAUTION       walkable, but bumpy: slow down and say "uneven ground ahead"
    NOT_WALKABLE  a step edge, a big drop or climb, a fall, or no way there

The bumpiness is read from the patch itself, along the path (`path_bump`),
because that is all the dog will have at run time: it never knows the
terrain's name, so "rough ground" has to be something it can measure.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np

# --- the patch -------------------------------------------------------------
RES = 0.1                       # metres a cell; the Isaac height scanner's own
X0, NX = -0.4, 21               # x (forward) from -0.4 to +1.6 m
Y0, NY = -1.0, 21               # y (left) from -1.0 to +1.0 m

GROUND_R = 0.25                 # ground_under(): points this close count

# --- the trial -------------------------------------------------------------
# Targets stay inside the patch: at the widest bearing the farthest target is
# 1.4 * sin(45 deg) = 0.99 m to the side, inside the patch's 1.0.
TARGET_D = (0.6, 1.4)           # metres ahead
TARGET_BEARING = np.radians(45.0)
WALK_V = 0.8                    # m/s -- the Go2's MAX_V in sim/robot/mujoco_robot.py

# --- the label (see classes()) ---------------------------------------------
# Tuned against run_20261006_132140 (100k trials, scripts/show_affordance.py).
# TILT_MAX was 8.6 deg, and the Go2's own gait rocks the body 8-10 deg on
# ground with 0.5 cm of bumps: flat ground failed at random, 72% of
# random_rough on tilt alone. 15 deg is clear of the gait and still catches a
# body that really pitches. STEP_MAX is the body rising or dropping by more
# than a kerb over the trial, which is a stair, not a bump.
TILT_MAX = math.radians(15.0)   # rad of roll/pitch combined, worst moment
STEP_MAX = 0.08                 # m of body height gained or lost
# Bumps: the largest height jump between neighbouring cells (10 cm apart) on
# the path. Accessibility codes allow ~1.3 cm of sudden change on a walkway;
# 2 cm leaves room for the scanner. Up to 5 cm is uneven ground to walk with a
# warning; beyond it, an edge to refuse. A 1:12 ramp is 0.8 cm per cell, so
# slopes pass as slopes, not bumps.
BUMP_WALK = 0.02                # m: at or under, walkable
BUMP_CAUTION = 0.05             # m: at or under, caution; over, not walkable
PATH_HALF_W = 0.3               # m either side of the line to the target

WALKABLE, CAUTION, NOT_WALKABLE = 0, 1, 2
CLASS_NAMES = ("walkable", "caution", "not walkable")

# Every field a record has, with its per-trial shape. The collector writes
# exactly these and load_shards() checks them, so the two cannot drift.
FIELDS = {
    "patch": (NY, NX),          # float32, heights above ground_z, NaN = unseen
    "target": (2,),             # float32, (dx, dy) in the dog's frame at the start
    "reached": (),              # bool, got within ARRIVED_R of the target
    "fell": (),                 # bool, the episode ended on a body contact
    "duration": (),             # float32, seconds the trial ran
    "max_tilt": (),             # float32, rad, largest body tilt from upright
    "dz": (),                   # float32, m, body height at the end minus the start
    "z_span": (),               # float32, m, highest minus lowest body height
    "terrain_type": (),         # int16, Isaac terrain column (-1 if unknown)
    "terrain_level": (),        # int16, Isaac terrain row = difficulty (-1 if unknown)
}


def cell_centres():
    """(xs, ys): the patch's cell centres in the dog's frame."""
    return X0 + RES * np.arange(NX), Y0 + RES * np.arange(NY)


def patches_from_points(points, poses, ground_z):
    """(B, NY, NX) patches, one per dog, from (B, M, 3) world points.

    `poses` is (B, 3) of (x, y, yaw); `ground_z` is (B,). Points that are not
    finite (a scanner ray that hit nothing) are ignored. A point lands in the
    cell whose centre it is nearest; each cell keeps the highest.
    """
    points = np.asarray(points, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    B, M, _ = points.shape
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx = points[..., 0] - poses[:, 0:1]
    dy = points[..., 1] - poses[:, 1:2]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    h = points[..., 2] - np.asarray(ground_z, dtype=np.float64)[:, None]
    finite = np.isfinite(h) & np.isfinite(lx) & np.isfinite(ly)
    ix = np.rint((np.where(finite, lx, X0) - X0) / RES).astype(np.int64)
    iy = np.rint((np.where(finite, ly, Y0) - Y0) / RES).astype(np.int64)
    ok = finite & (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)

    flat = np.full(B * NY * NX, -np.inf)
    b = np.broadcast_to(np.arange(B)[:, None], (B, M))
    np.maximum.at(flat, (b * NY * NX + iy * NX + ix)[ok], h[ok])
    flat[np.isneginf(flat)] = np.nan
    return flat.reshape(B, NY, NX).astype(np.float32)


def patch_from_points(points, pose, ground_z):
    """One (NY, NX) patch from (M, 3) world points -- the runtime's call."""
    return patches_from_points(np.asarray(points)[None], np.asarray(pose)[None],
                               np.asarray([ground_z]))[0]


def ground_under(points, poses, r=GROUND_R):
    """(B,) median height of the points within `r` of each dog -- its ground.

    For a producer that can see under the dog (Isaac's height scanner). NaN
    where nothing landed that close.
    """
    points = np.asarray(points, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    d = np.hypot(points[..., 0] - poses[:, 0:1], points[..., 1] - poses[:, 1:2])
    z = np.where((d <= r) & np.isfinite(points[..., 2]), points[..., 2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows
        return np.nanmedian(z, axis=1)


def path_cells(targets, half_w=PATH_HALF_W):
    """(N, NY, NX) bool: the patch cells on the way from the dog to each target.

    Within half_w of the straight line, from the dog to the target. A target
    on top of the dog (zero length) is the cells within half_w of it."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    xs, ys = cell_centres()
    X, Y = np.meshgrid(xs, ys)                  # [iy, ix], like the patch
    L = np.linalg.norm(targets, axis=1)
    u = np.where(L[:, None] > 1e-9, targets / np.maximum(L, 1e-9)[:, None], [1.0, 0.0])
    along = X[None] * u[:, 0, None, None] + Y[None] * u[:, 1, None, None]
    side = np.abs(-X[None] * u[:, 1, None, None] + Y[None] * u[:, 0, None, None])
    # EPS: cell centres are multiples of RES only up to float rounding (y = -0.3
    # is stored as -0.29999999999999993, +0.3 as 0.30000000000000004), and a cell
    # exactly on the edge must count the same on the left and on the right --
    # training mirrors every sample, so a path and its mirror image must match.
    eps = 1e-6
    return (side <= half_w + eps) & (along >= -eps) & (along <= L[:, None, None] + eps) | \
        (np.hypot(X, Y)[None] <= half_w + eps)


def path_bump(records, half_w=PATH_HALF_W, chunk=20_000):
    """(N,) metres: the largest height jump between neighbouring cells on the path.

    Neighbours along x and along y, both cells on the path (path_cells), unseen
    (NaN) cells skipped. 0 where nothing on the path was seen twice in a row.
    """
    patch, target = records["patch"], records["target"]
    out = np.zeros(len(patch), dtype=np.float32)
    for a in range(0, len(patch), chunk):
        p = patch[a:a + chunk].astype(np.float32)
        on = path_cells(target[a:a + chunk], half_w)
        jx = np.abs(np.diff(p, axis=2))
        jy = np.abs(np.diff(p, axis=1))
        jx = np.where(on[:, :, 1:] & on[:, :, :-1] & np.isfinite(jx), jx, 0.0)
        jy = np.where(on[:, 1:, :] & on[:, :-1, :] & np.isfinite(jy), jy, 0.0)
        out[a:a + chunk] = np.maximum(jx.max(axis=(1, 2)), jy.max(axis=(1, 2)))
    return out


def classes(records, tilt_max=TILT_MAX, step_max=STEP_MAX,
            bump_walk=BUMP_WALK, bump_caution=BUMP_CAUTION, bump=None):
    """(N,) int8: WALKABLE, CAUTION or NOT_WALKABLE for each trial.

    NOT_WALKABLE unless it arrived, did not fall, did not tip the body past
    tilt_max and did not climb or drop more than step_max -- the last two over
    the whole trial, not just its ends: a dog that goes up three stairs and
    down three more ends where it started -- and the path's bumps are within
    bump_caution. Then CAUTION if the bumps are over bump_walk, else WALKABLE.
    `bump` is path_bump(records), if already computed.
    """
    if bump is None:
        bump = path_bump(records)
    ok = (records["reached"] & ~records["fell"]
          & (records["max_tilt"] <= tilt_max)
          & (records["z_span"] <= step_max)
          & (bump <= bump_caution))
    out = np.full(len(ok), NOT_WALKABLE, dtype=np.int8)
    out[ok] = np.where(bump[ok] <= bump_walk, WALKABLE, CAUTION)
    return out


def label(records, **thresholds):
    """(N,) bool: could a person on the handle have followed this walk?

    Walkable or caution: CAUTION is a walk with a warning, not a refusal.
    Takes classes()' thresholds by name."""
    return classes(records, **thresholds) != NOT_WALKABLE


def save_shard(path, records, meta):
    """Write one shard: the record arrays plus `meta` as JSON, in one .npz."""
    _check(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **records)


def load_shards(directory):
    """(records, metas): every shard in `directory`, concatenated."""
    parts, metas = [], []
    for f in sorted(Path(directory).glob("*.npz")):
        with np.load(f) as z:
            parts.append({k: z[k] for k in FIELDS})
            metas.append(json.loads(str(z["meta"])))
    if not parts:
        raise FileNotFoundError(f"no .npz shards in {directory}")
    records = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    _check(records)
    return records, metas


def _check(records):
    missing = set(FIELDS) - set(records)
    extra = set(records) - set(FIELDS)
    if missing or extra:
        raise ValueError(f"record fields wrong: missing {sorted(missing)}, "
                         f"unexpected {sorted(extra)}")
    n = {len(records[k]) for k in FIELDS}
    if len(n) != 1:
        raise ValueError(f"record fields have different lengths: {n}")
    for k, shape in FIELDS.items():
        if records[k].shape[1:] != shape:
            raise ValueError(f"{k}: per-trial shape {records[k].shape[1:]}, "
                             f"expected {shape}")

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
arrive, did it fall, how far did the body tilt and rise -- and `label()`
decides what counts as walkable. That is a judgement about a blind person on
the handle, not about the dog: a good policy climbs a flight of stairs
without falling, and the MLP must still say no to it. Keeping the judgement
here means changing a threshold is a retrain, not a recollection on a GPU.
"""
import json
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

# --- the label (see label()) -----------------------------------------------
# Starting points, to be tuned against the data -- they are why the raw
# outcome is stored. TILT_MAX is a little over the 1:12 ramp a wheelchair
# code allows (4.8 deg) plus the gait's own rocking; STEP_MAX is the body
# rising or dropping by more than a kerb, which is a stair, not a bump.
TILT_MAX = 0.15                 # rad, ~8.6 deg of roll/pitch combined
STEP_MAX = 0.08                 # m of body height gained or lost

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


def label(records, tilt_max=TILT_MAX, step_max=STEP_MAX):
    """(N,) bool: was this a walk a person on the handle could have followed?

    Arrived, did not fall, did not tip the body past tilt_max, and did not
    climb or drop more than step_max. Both of the last two use the whole
    trial, not just its ends: a dog that goes up three stairs and down three
    more ends where it started.
    """
    return (records["reached"] & ~records["fell"]
            & (records["max_tilt"] <= tilt_max)
            & (records["z_span"] <= step_max))


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

"""Measure the chance margin CE-RRT* keeps from people (planning/ce_rrt.py).

The planner predicts each person in a straight line (tracking.Track.predict)
and keeps CONTACT + Q0 + Q_RATE * t from where the line says they will be.
Q0 and Q_RATE are the 99th percentile of that line's error, measured here: the
tracker's moving tracks against where every walker really was, t seconds
later, from runs logged with --track-log.

    python scripts/ce_rrt_calibrate.py                      # drive 12 runs, then measure
    python scripts/ce_rrt_calibrate.py --logs DIR           # measure logs already driven

Takes about 30 s to drive. On hardware there is no ground truth to log; the
same measurement wants the people tracked by something better than the dog
(a ceiling camera, a motion-capture room), which is why this stays a script.
"""
import argparse
import glob
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cyberdog.planning import ce_rrt

ROUTES = ["room 101", "room 201", "restroom", "cafeteria"]
SEEDS = [2, 3, 5]
HORIZONS = [0.0, 0.5, 1.0, 1.5, 2.0]
MATCH_R = 0.5           # metres: a track this near a walker is that walker
MOVING = 0.3            # m/s: slower tracks have no direction to split along


def drive(out):
    def one(rs):
        r, s = rs
        subprocess.run([sys.executable, "-m", "cyberdog.sim.run_building", r,
                        "--pedestrians", "3", "--seed", str(s), "--auto-confirm",
                        "--no-video", "--crowd", "aware", "--planner", "map",
                        "--track-log", str(out / f"{r.replace(' ', '_')}_{s}.jsonl")],
                       capture_output=True, check=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, [(r, s) for r in ROUTES for s in SEEDS]))


def errors(logs):
    """{horizon: (along, across)} -- the straight line's error, split along
    the track's own direction and across it."""
    along = {h: [] for h in HORIZONS}
    across = {h: [] for h in HORIZONS}
    for fn in logs:
        rows = [json.loads(line) for line in open(fn)]
        for i, r in enumerate(rows):
            for x, y, vx, vy in r["tracks"]:
                if not r["walkers"]:
                    continue
                d, k = min((math.dist((x, y), w), k) for k, w in enumerate(r["walkers"]))
                sp = math.hypot(vx, vy)
                if d > MATCH_R or sp < MOVING:
                    continue
                ux, uy = vx / sp, vy / sp
                for h in HORIZONS:
                    j = i + int(round(h / 0.05))
                    if (j >= len(rows) or rows[j]["floor"] != r["floor"]
                            or abs(rows[j]["t"] - r["t"] - h) > 0.06):
                        continue
                    wx, wy = rows[j]["walkers"][k]
                    ex, ey = wx - (x + vx * h), wy - (y + vy * h)
                    along[h].append(ex * ux + ey * uy)
                    across[h].append(-ex * uy + ey * ux)
    return {h: (np.array(along[h]), np.array(across[h])) for h in HORIZONS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", metavar="DIR", help="measure these --track-log files")
    args = ap.parse_args()
    if args.logs:
        logs = sorted(glob.glob(str(Path(args.logs) / "*.jsonl")))
    else:
        out = Path(tempfile.mkdtemp(prefix="ce_rrt_calibrate_"))
        print(f"driving {len(ROUTES) * len(SEEDS)} runs into {out} ...")
        drive(out)
        logs = sorted(glob.glob(str(out / "*.jsonl")))
    e = errors(logs)
    print("99th percentile of the straight line's error, metres")
    print(f"{'ahead':>6s} {'n':>6s} {'along':>7s} {'across':>7s} {'worse':>7s}")
    q = {}
    for h in HORIZONS:
        a, c = e[h]
        if not len(a):
            continue
        pa, pc = np.percentile(abs(a), 99), np.percentile(abs(c), 99)
        q[h] = max(pa, pc)
        print(f"{h:5.1f}s {len(a):6d} {pa:7.2f} {pc:7.2f} {q[h]:7.2f}")
    if 0.0 in q and 0.5 in q:
        print(f"\nstraight line through 0 s and 0.5 s: Q0 = {q[0.0]:.2f}, "
              f"Q_RATE = {(q[0.5] - q[0.0]) / 0.5:.2f} per second "
              f"(planning/ce_rrt.py has Q0 = {ce_rrt.Q0}, Q_RATE = {ce_rrt.Q_RATE})")


if __name__ == "__main__":
    main()

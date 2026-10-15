"""The planner's own gate: the same arm with --planner map and --planner cerrt.

Gate D asks whether VAMOS is no worse than the map. It cannot also judge the
planner: with the planner in both arms, a map-only arm that the planner has
made clean (no contact at all) asks the VAMOS arm to be clean on all 70 runs,
and one reshuffled encounter fails it. So the planner is judged the way
Gate D judges VAMOS, against itself switched off -- paired, route by route
and seed by seed, per arm, on the same bars:

    P2  collisions + handle contact      planner <= without
    P3  pedestrian contact               planner <= without
    P4  stair / no-go entries            0
    P5  arrivals on the intended floor   planner >= without
    P6  mean time to goal                planner <= without x 1.10
    P7  stops for people, waiting        report -- what the planner is for

    python scripts/planner_ab.py output/gate_d/<without> output/gate_d/<with>

Both campaigns must be complete Gate D runs with the same routes, seeds,
crowd and VAMOS model; the first without the planner, the second with it.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_d import parse  # noqa: E402

TIME_SLACK = 1.10


def load(d):
    s = json.loads((Path(d) / "summary.json").read_text())
    runs = {}
    for r in s["runs"]:
        if "stops" not in r:
            # A campaign from before gate_d.py kept the stops: read them back
            # out of its own log.
            log = Path(d) / f"{r['arm']}__{r['route'].replace(' ', '_')}__seed{r['seed']}.log"
            r = {**parse(log.read_text()), **r}
        runs[(r["arm"], r["route"], r["seed"])] = r
    return s, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("without", help="Gate D campaign driven with --planner map")
    ap.add_argument("with_", metavar="with", help="Gate D campaign driven with --planner cerrt")
    args = ap.parse_args()
    s0, a = load(args.without)
    s1, b = load(args.with_)
    for key in ("routes", "seeds", "crowd", "model"):
        if s0.get(key) != s1.get(key):
            sys.exit(f"not paired: {key} differs ({s0.get(key)} / {s1.get(key)})")
    if s0.get("planner", "map") != "map" or s1.get("planner") != "cerrt":
        sys.exit("the first campaign must be without the planner, the second with it")

    passed = True
    for arm in ("map", "vamos"):
        keys = sorted(k for k in a if k[0] == arm)
        if set(keys) != {k for k in b if k[0] == arm}:
            sys.exit(f"{arm}: the campaigns did not drive the same runs")
        A, B = [a[k] for k in keys], [b[k] for k in keys]

        def total(R, f):
            return round(sum(r.get(f, 0) or 0 for r in R), 1)

        def mean_time(R):
            ts = [r["time_s"] for r in R if r["arrived"]]
            return round(statistics.mean(ts), 1) if ts else None

        hit = lambda R: round(total(R, "collision_s") + total(R, "handler_s"), 1)
        ta, tb = mean_time(A), mean_time(B)
        rows = [
            ("P2", "Collisions + handle contact (s)", hit(A), hit(B), hit(B) <= hit(A)),
            ("P3", "Pedestrian contact (s)", total(A, "contact_s"), total(B, "contact_s"),
             total(B, "contact_s") <= total(A, "contact_s")),
            ("P4", "Stair / no-go entries", sum(r["halted"] for r in A),
             sum(r["halted"] for r in B), not any(r["halted"] for r in A + B)),
            ("P5", "Arrivals", sum(r["arrived"] for r in A), sum(r["arrived"] for r in B),
             sum(r["arrived"] for r in B) >= sum(r["arrived"] for r in A)),
            ("P6", "Mean time to goal (s)", ta, tb,
             ta is not None and tb is not None and tb <= ta * TIME_SLACK),
            ("P7", "Stops for people", total(A, "stops"), total(B, "stops"), None),
            ("P7", "Waiting for people (s)", total(A, "waiting_s"), total(B, "waiting_s"), None),
            ("P7", "Went round somebody", total(A, "rounds"), total(B, "rounds"), None),
        ]
        print(f"\n{arm} arm, {len(keys)} pairs")
        print(f"{'':4s}{'KPI':34s}{'without':>10s}{'with':>10s}   result")
        for kpi, name, x, y, ok in rows:
            verdict = "report" if ok is None else ("PASS" if ok else "FAIL")
            print(f"{kpi:4s}{name:34s}{str(x):>10s}{str(y):>10s}   {verdict}")
            passed = passed and ok is not False
        moved = [(k, a[k], b[k]) for k in keys
                 if a[k]["contact_s"] != b[k]["contact_s"]
                 or a[k]["collision_s"] + a[k]["handler_s"] != b[k]["collision_s"] + b[k]["handler_s"]
                 or a[k]["arrived"] != b[k]["arrived"]]
        for k, x, y in moved:
            print(f"    {k[1]:28s} seed {k[2]:2d}: contact {x['contact_s']} -> {y['contact_s']}, "
                  f"collision+handle {x['collision_s'] + x['handler_s']:.1f} -> "
                  f"{y['collision_s'] + y['handler_s']:.1f}, "
                  f"arrived {x['arrived']} -> {y['arrived']}")
    print(f"\nPlanner gate: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

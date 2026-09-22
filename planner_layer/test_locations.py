"""Check the synthetic building matches the Input Treating Layer's training set.

Map-only (fast, no model):
    python3 planner_layer/test_locations.py
      - every location the training set can output exists in locations.json
      - every location point is free space on its floor grid
      - a route exists from the main entrance to every location

With the NLU model (slow, needs the vamos_mac env):
    python3 planner_layer/test_locations.py --with-model
      - also runs one generated command per location through the model and
        routes to whatever it parsed

Running on Linux:
    Setup (once), from the project root:
        python3 -m venv .venv && source .venv/bin/activate
        pip install numpy pyyaml pillow scipy
        python3 planner_layer/test_locations.py
    For --with-model, also install the model libraries (same venv is fine):
        pip install torch transformers peft trl datasets accelerate
        python3 planner_layer/test_locations.py --with-model
    Uses cuda automatically if available. Run from the project root or anywhere
    -- paths are resolved relative to this file.
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "input_treating_layer"))
sys.path.insert(0, HERE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.with_model:
        # Load torch before the map stack (numpy/PIL) -- libomp crash on macOS otherwise.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from infer import parse_command
    from generate_dataset import generate_synthetic_dataset
    from building_router import BuildingRouter

    random.seed(args.seed)
    samples = generate_synthetic_dataset(4000)
    by_location = {}
    for s in samples:
        by_location.setdefault(s["location"], s)

    router = BuildingRouter()
    start = router.resolve("main entrance", 1, (0, 0))
    failures = []

    for name in sorted(by_location):
        entries = router.locations.get(name)
        if not entries:
            failures.append(f"{name}: missing from locations.json")
            continue
        for e in entries:
            grid = router.grids[e["floor"]]
            if not grid.is_free(*grid.world_to_grid(*e["xy"])):
                failures.append(f"{name} (floor {e['floor']}): point is not free space")
        target = router.resolve(name, start["floor"], start["xy"])
        if router.plan(start["floor"], start["xy"], target) is None:
            failures.append(f"{name}: no route from main entrance")

    print(f"Map check: {len(by_location) - len({f.split(':')[0] for f in failures})}"
          f"/{len(by_location)} locations OK")

    if args.with_model:
        correct = 0
        for name in sorted(by_location):
            text = by_location[name]["text"]
            parsed = (parse_command(text).get("target_location") or "").strip().lower()
            ok = parsed == name
            routed = router.resolve(parsed, start["floor"], start["xy"]) is not None
            correct += ok
            if not ok or not routed:
                failures.append(f"model: {text!r} -> {parsed!r} (expected {name!r}, routable={routed})")
        print(f"Model check: {correct}/{len(by_location)} commands parsed to the right location")

    for f in failures:
        print("  FAIL", f)
    print("All good." if not failures else f"{len(failures)} failure(s).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

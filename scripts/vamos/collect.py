"""Collect VAMOS LoRA training data (spec L4 step 5) with the auto-labeller.

Map-only runs of `run_building --autolabel`, every named destination that is
not a Gate D route, with and without people. Each frame is labelled with the
path the dog walked from it (sim/autolabel.py), and only clean runs -- arrived,
nothing touched -- go into the dataset.

    python scripts/vamos/collect.py                       # train set
    python scripts/vamos/collect.py --test                # the Gate D routes, held out
    python scripts/vamos/collect.py --seeds 0 1 --pairs 0 # smaller

The Gate D routes are held out on purpose. The adapter is judged on them twice,
offline (scripts/vamos/evaluate.py) and closed-loop (scripts/gate_d.py), and a
model that has seen those exact walks would be judged on its memory. They
share corridors with the training routes -- there is one building -- so this
is a held-out *route*, not a held-out layout, and Gate A's grouped split is
still owed.

Writes datasets/vamos/<split>/<run>/ (frames, samples.jsonl, run.json) and
datasets/vamos/<split>/index.json. No VAMOS server needed.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gate_d import ROUTES  # noqa: E402

from cyberdog import paths  # noqa: E402

OUT = paths.DATASETS_DIR / "vamos"
PEDESTRIANS = 3
# Names that are not somewhere to walk to: the stairs end in a refusal in the
# corridor outside, and the lift and the hallway are the corridor itself.
NOT_PLACES = {"stairs", "staircase", "stairway", "stairwell", "elevator", "lift",
              "hallway", "main entrance"}


def destinations():
    names = json.loads((paths.ROOT / "data" / "building" / "locations.json").read_text())
    return [n for n in names if n not in NOT_PLACES and n not in ROUTES]


def run_id(command, seed):
    return f"{command.replace(', then ', '+').replace(' ', '_')}__seed{seed}"


def drive(command, seed, split_dir):
    rid = run_id(command, seed)
    out = split_dir / rid
    cmd = [sys.executable, "-m", "cyberdog.sim.run_building", command,
           "--auto-confirm", "--no-video", "--autolabel", str(out),
           "--pedestrians", str(PEDESTRIANS if seed else 0), "--seed", str(seed)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.txt").write_text(p.stdout + p.stderr)
    run = out / "run.json"
    r = json.loads(run.read_text()) if run.exists() else {"clean": False, "samples": 0,
                                                         "outcome": f"exit {p.returncode}"}
    return {"run": rid, "command": command, "seed": seed, "clean": r["clean"],
            "samples": r["samples"], "outcome": r["outcome"],
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", action="store_true",
                    help="collect the held-out Gate D routes instead of the train set")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="0 is an empty building; any other is a crowd of 3")
    ap.add_argument("--pairs", type=int, default=20,
                    help="two-stop commands added to the train set, so legs start "
                         "somewhere other than the main entrance")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    if not paths.BUILDING_SCENE.exists():
        raise SystemExit("no building scene yet -- run: "
                         "python -m cyberdog.sim.scene.build_scene --building")
    if args.test:
        split, commands = "test", list(ROUTES)
    else:
        split, places = "train", destinations()
        rng = random.Random(0)
        commands = places + [", then ".join(rng.sample(places, 2)) for _ in range(args.pairs)]
    split_dir = OUT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s) for c in commands for s in args.seeds]
    print(f"{split}: {len(commands)} commands x seeds {args.seeds} = {len(jobs)} runs "
          f"-> {split_dir}")

    t0, results = time.time(), []
    with ThreadPoolExecutor(args.jobs) as pool:
        for k, r in enumerate(pool.map(lambda j: drive(*j, split_dir), jobs), 1):
            results.append(r)
            flag = "" if r["clean"] else f"  NOT CLEAN ({r['outcome']})"
            print(f"  [{k}/{len(jobs)}] {r['run']}: {r['samples']} frames, "
                  f"{r['wall_s']:.0f}s{flag}", flush=True)

    clean = [r for r in results if r["clean"]]
    index = {"split": split, "seeds": args.seeds, "runs": results,
             "clean_runs": [r["run"] for r in clean],
             "samples": sum(r["samples"] for r in clean)}
    (split_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"{len(clean)}/{len(results)} runs clean, {index['samples']} labelled frames "
          f"usable, {time.time() - t0:.0f}s -> {split_dir / 'index.json'}")


if __name__ == "__main__":
    main()

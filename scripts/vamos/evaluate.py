"""Score VAMOS, base or fine-tuned, on the held-out Gate D routes (offline).

Sends each held-out frame to the VAMOS server exactly as the live client does
(5-beam search, temperature 0) and compares the five answers with the path
the dog really walked from that frame, on the floor, in metres:

    path   mean distance from each walked point to the candidate's line
    end    distance between the candidate's end and the walked path's end
    best   the better of the five candidates, by `path`; top is beam 1
    spread how far the five candidates' ends lie from their mean -- the
           README's +/-0.25 m, which a fine-tune is supposed to widen where
           the walk needs it, not everywhere

Reported for all frames and for *detour* frames: those whose walked path
leaves the straight line to its own end by more than DETOUR m -- the frames
that are about going round something, and the reason for the fine-tune.

    conda activate cyberdog_sim                  # server in another terminal
    python scripts/vamos/evaluate.py --name base
    python scripts/vamos/evaluate.py --name lora --load <repo>/models/vamos_lora
    python scripts/vamos/evaluate.py --compare base lora

Results go to output/vamos_eval/<name>.json, every frame's answer included.
This is the offline half of the judgement. The closed-loop half is Gate D,
with the server running the adapter.
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests

from cyberdog import paths
from cyberdog.planning.checkpoint_projector import load_camera_config
from cyberdog.sim.vamos_client import DEFAULT_URL, VamosPolicy, pixel_to_ground

DATA = paths.DATASETS_DIR / "vamos" / "test"
OUT = paths.OUTPUT_DIR / "vamos_eval"
DETOUR = 0.15        # m off the straight line that makes a frame a detour (= train_lora.DETOUR_M)


def seg_dist(p, a, b):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def line_dist(p, line):
    return min(seg_dist(p, a, b) for a, b in zip(line, line[1:]))


def bulge(path):
    """How far the walked path strays from the straight line to its end."""
    start = path[0]
    return max(seg_dist(p, start, path[-1]) for p in path)


def frames(data_dir, limit, seed):
    index = json.loads((data_dir / "index.json").read_text())
    out = []
    for rid in index["clean_runs"]:
        for l in (data_dir / rid / "samples.jsonl").read_text().splitlines():
            s = json.loads(l)
            s["run"], s["image"] = rid, str(data_dir / rid / s["frame"])
            # The walk starts at the lens, not at the first visible point.
            s["detour"] = bulge([s["cam_pose"][:2]] + s["ground"]) > DETOUR
            out.append(s)
    if limit and len(out) > limit:
        # Every detour frame first, up to half: they are rarer and they are
        # what this is for. The rest drawn at random, the same draw each time.
        rng = random.Random(seed)
        det = [s for s in out if s["detour"]]
        rest = [s for s in out if not s["detour"]]
        rng.shuffle(det)
        rng.shuffle(rest)
        det = det[:limit // 2]
        out = det + rest[:limit - len(det)]
    return out


def score(s, answers, cam):
    walked = [tuple(p) for p in s["ground"]]
    cands = []
    for pp in answers:
        g = [pixel_to_ground(u, v, cam, s["cam_pose"]) for u, v in pp]
        g = [p for p in g if p is not None]
        g = [p for k, p in enumerate(g) if k == 0 or math.dist(p, g[k - 1]) > 0.05]
        if len(g) >= 2:
            cands.append(g)
    if not cands:
        return None
    path = [statistics.mean(line_dist(p, c) for p in walked) for c in cands]
    end = [math.dist(c[-1], walked[-1]) for c in cands]
    ends = np.array([c[-1] for c in cands])
    spread = float(np.linalg.norm(ends - ends.mean(0), axis=1).mean())
    b = int(np.argmin(path))
    return {"top_path": path[0], "top_end": end[0], "best_path": path[b],
            "best_end": end[b], "spread": spread, "n": len(cands)}


def summary(rows):
    out = {}
    for name, sel in (("all", rows), ("detour", [r for r in rows if r["detour"]])):
        ok = [r["score"] for r in sel if r["score"]]
        out[name] = {"frames": len(sel), "answered": len(ok), **{
            k: round(statistics.mean(x[k] for x in ok), 3) if ok else None
            for k in ("top_path", "top_end", "best_path", "best_end", "spread")}}
    return out


def table(results):
    keys = ("top_path", "best_path", "top_end", "best_end", "spread")
    for part in ("all", "detour"):
        print(f"\n{part} frames  " + "  ".join(f"{k:>9s}" for k in keys) + "   (metres)")
        for name, r in results.items():
            s = r["summary"][part]
            print(f"  {name:12s}" + "  ".join(
                f"{s[k]:9.3f}" if s[k] is not None else f"{'-':>9s}" for k in keys)
                  + f"   {s['answered']}/{s['frames']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", help="what to call this result (base, lora, ...)")
    ap.add_argument("--load", metavar="MODEL", help="ask the server to load this model first")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--limit", type=int, default=400, help="frames to send (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-prompt", action="store_true",
                    help="ask with the live projector's goal, not the walked path's end")
    ap.add_argument("--compare", nargs="+", metavar="NAME", help="print saved results side by side")
    args = ap.parse_args()

    if args.compare:
        table({n: json.loads((OUT / f"{n}.json").read_text()) for n in args.compare})
        return
    if not args.name:
        ap.error("--name is required (or --compare)")

    if args.load:
        r = requests.post(f"{args.url}/load_model", json={"model_path": args.load}, timeout=900)
        r.raise_for_status()
        if not r.json().get("success"):
            raise SystemExit(r.json().get("message"))
        print(r.json()["message"])

    cam = load_camera_config()
    policy = VamosPolicy(cam, is_free=lambda x, y: True, url=args.url)
    if not policy.available():
        raise SystemExit(f"VAMOS server is not answering on {args.url} (main README, step 6)")
    todo = frames(Path(args.data), args.limit, args.seed)
    print(f"{len(todo)} frames ({sum(s['detour'] for s in todo)} detours) -> {args.url}")

    from PIL import Image
    rows, t0 = [], time.time()
    for k, s in enumerate(todo, 1):
        image = np.asarray(Image.open(s["image"]).convert("RGB"))
        try:
            answers = policy._request(image, s["map_prompt"] if args.map_prompt else s["prompt"])
        except (requests.RequestException, ValueError) as e:
            print(f"  {s['frame']}: {e}", file=sys.stderr)
            answers = []
        rows.append({"run": s["run"], "frame": s["frame"], "detour": s["detour"],
                     "answers": answers, "score": score(s, answers, cam)})
        if k % 25 == 0:
            print(f"  {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    result = {"name": args.name, "load": args.load, "data": args.data,
              "map_prompt": args.map_prompt, "summary": summary(rows), "frames": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(result))
    table({args.name: result})
    print(f"\n-> {OUT / f'{args.name}.json'}")


if __name__ == "__main__":
    main()

"""The auto-labeller (spec L2 step 4): training pairs for the VAMOS LoRA.

A map-only run already does what zero-shot VAMOS cannot: it goes round a
crate by more than the model's +/-0.25 m spread, because `free_destination`
moves the goal into the gap. So the label for a frame is not the A* route --
that runs straight through the crates, which are on no map -- but the path
the dog *actually walked* from that frame on, projected into the frame it
was standing in when it looked. Only runs that arrived with no collision, no
handle contact and no person contact are kept; a label that grazed a box
teaches grazing boxes.

The goal in the prompt is hindsight: the projected end of that walked path,
not the projector's goal pixel. The two are close (the dog was steering at
the goal), and a path that ends where the prompt says it should is the
pairing VAMOS was trained on. The projector's own pixel is kept beside it.

Each run writes one directory:

    <dir>/frames/<n>.png     the onboard frame, exactly what VAMOS is sent
    <dir>/samples.jsonl      one line per labelled frame
    <dir>/run.json           the run's outcome; `clean` decides if it is used

Nothing here imports torch: the twin stays out of the training environment
and the training script stays out of the twin's.
"""
import json
import math
import os

from cyberdog.planning.checkpoint_projector import project_to_pixel

SAMPLE_EVERY = 10     # ticks between frames kept: 0.5 s at 20 Hz
MIN_MOVED = 0.25      # m the dog must have walked since the last kept frame
N_POINTS = 5          # VAMOS answers with 5 points (10 location tokens)
MIN_LENGTH = 1.0      # m of walked path that must be visible for a label
STEP = 0.05           # m between points when walking the track


def prompt(goal_loc):
    """The prompt `vamos_prompt` sends, for a goal in location bins."""
    return f"Navigate to x=<loc{goal_loc[0]:04d}>, y=<loc{goal_loc[1]:04d}>."


def target(locs):
    """The answer VAMOS gives: x then y for each point, no separators."""
    return "".join(f"<loc{x:04d}><loc{y:04d}>" for x, y in locs)


def along(track, length, step=STEP):
    """Points every `step` metres along `track` (a list of (x, y)), up to
    `length` metres. Standing still adds nothing, so a dog that waited for
    somebody walks the same path as one that did not have to."""
    out, walked, nxt = [track[0]], 0.0, step
    for a, b in zip(track, track[1:]):
        seg = math.dist(a, b)
        if seg < 1e-9:
            continue
        while nxt <= walked + seg and nxt <= length + 1e-9:
            t = (nxt - walked) / seg
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            nxt += step
        walked += seg
        if nxt > length + 1e-9:
            break
    return out


def label(track, cam_pose, cam, length, n=N_POINTS, min_length=MIN_LENGTH):
    """The walked path from here, as VAMOS would draw it in this frame.

    `track` is the body's (x, y) from the frame's tick to the end of the leg;
    `length` how far along it to label. The path is cut where it first leaves
    the image -- a turn through a doorway takes it out of view, and a point
    the camera cannot see is not something the model can be asked to draw --
    after skipping the stretch under the lens, then resampled to `n` points evenly by arc length, the first a step away
    from the dog. Returns (pixels, locs, metres) or None if less than
    `min_length` of it is in view.
    """
    pts = along(track, length)
    seen = []
    for p in pts[1:]:
        r = project_to_pixel(p, cam_pose, cam)
        if r["state"] == "TRACK":
            seen.append((p, r["pixel"], r["loc"]))
        elif seen:
            break
        # Else still under the lens: the first ~0.6 m is below the bottom of
        # the frame, and the path starts where the camera can first see it.
    if len(seen) < 2:
        return None
    # Arc length of the visible part, measured from the dog.
    cum, prev = [], pts[0]
    total = 0.0
    for p, _, _ in seen:
        total += math.dist(prev, p)
        cum.append(total)
        prev = p
    if total < min_length:
        return None
    picks = []
    for k in range(1, n + 1):
        s = total * k / n
        j = min(range(len(cum)), key=lambda i: abs(cum[i] - s))
        picks.append(seen[j])
    # The projector's own bins, from the unrounded pixel, so a label and a
    # live prompt for the same point are the same tokens.
    return ([px for _, px, _ in picks], [lc for _, _, lc in picks],
            [p for p, _, _ in picks])


class AutoLabel:
    """Records one run's frames and labels them when the run is over."""

    def __init__(self, out_dir, cam, every=SAMPLE_EVERY):
        self.dir = out_dir
        self.cam = cam
        self.every = every
        os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
        self.legs = []          # [(floor, track, samples)]
        self.kept = 0

    def start_leg(self, floor):
        self.legs.append((floor, [], []))

    def tick(self, n, pose, cam_pose, state, image_fn):
        """Called once per control tick of a leg. `image_fn` renders the
        onboard frame, and is only called for a frame that is kept."""
        floor, track, samples = self.legs[-1]
        track.append((float(pose[0]), float(pose[1])))
        if n % self.every or state.get("state") != "TRACK" or "destination" not in state:
            return
        if samples and math.dist(track[samples[-1]["i"]], track[-1]) < MIN_MOVED:
            return
        name = f"{len(self.legs) - 1:02d}_{n:05d}.png"
        from PIL import Image
        Image.fromarray(image_fn()).save(os.path.join(self.dir, "frames", name))
        dest = state["destination"]
        samples.append({
            "i": len(track) - 1, "tick": n, "frame": f"frames/{name}",
            "pose": [round(float(v), 4) for v in pose],
            "cam_pose": [round(float(v), 4) for v in cam_pose],
            "destination": [round(float(v), 4) for v in dest],
            "length": math.dist(pose[:2], dest),
            "map_loc": list(state["loc"]),
        })

    def finish(self, outcome):
        """Label every frame from the walked track and write it all out.
        `outcome` is the run's verdict; `outcome["clean"]` decides whether a
        training set may use it, but the frames are written either way, so
        a dirty run can still be looked at."""
        lines = []
        for leg, (floor, track, samples) in enumerate(self.legs):
            for s in samples:
                got = label(track[s["i"]:], s["cam_pose"], self.cam, s["length"])
                if got is None:
                    os.remove(os.path.join(self.dir, s["frame"]))
                    continue
                pixels, locs, ground = got
                lines.append({
                    "frame": s["frame"], "floor": floor, "leg": leg, "tick": s["tick"],
                    "prompt": prompt(locs[-1]), "target": target(locs),
                    "pixels": [list(p) for p in pixels],
                    "ground": [[round(x, 3), round(y, 3)] for x, y in ground],
                    "pose": s["pose"], "cam_pose": s["cam_pose"],
                    "map_prompt": prompt(s["map_loc"]),
                })
        with open(os.path.join(self.dir, "samples.jsonl"), "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        with open(os.path.join(self.dir, "run.json"), "w") as f:
            json.dump({**outcome, "samples": len(lines)}, f, indent=2)
        self.kept = len(lines)
        return len(lines)

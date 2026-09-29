"""VAMOS in the loop: image + goal pixel -> candidate paths -> one chosen path.

The VLM runs as a separate process (VAMOS/server/vlm_server.py) because it
needs torch and this environment deliberately does not have it -- the same
split the real robot has, where the VLM is a service rather than an import.

"AI proposes, simple code disposes": VAMOS returns five candidate paths and a
deterministic scorer picks one. On the real system that scorer is the
affordance MLP; here it is the ground-truth map, which is honest for a PoC
and gives the MLP something to be measured against.

Two stages to the disposing. First a cheap geometric test throws out anything
whose drawn line leaves free space -- no point imagining a path that is
already through a wall. What survives is handed to dreaming.py, which rolls
the dog's own controller forward along it with noise and comes back with a
safety factor: the probability of getting down it without hitting anything,
weighted by how much room it left. That factor gates (below GATE, rejected),
ranks (safest first, progress second) and throttles (the chosen path's factor
scales the dog's speed, so a tight path is walked slowly rather than briskly).
"""
import io
import math

import requests
from PIL import Image

DEFAULT_URL = "http://127.0.0.1:8009"

# The spec's confidence gate. A candidate the dog cannot imagine surviving at
# least this well is not offered, however much progress it would have made.
GATE = 0.5
# Safety is the first sort key, but not to three decimal places: two paths
# whose factors differ by a hundredth are equally safe, and progress should
# decide between them.
BAND = 0.05


def pixel_to_ground(u, v, cam, cam_pose):
    """Inverse of checkpoint_projector.project_to_pixel.

    A pixel only maps back to a ground point if it is below the horizon
    (v > cy); anything at or above it is infinitely far away.
    """
    cy, fy, fx, cx = cam["cy"], cam["fy"], cam["fx"], cam["cx"]
    if v <= cy + 1e-6:
        return None
    z_cam = fy * cam["camera_height"] / (v - cy)      # forward distance
    x_cam = (u - cx) * z_cam / fx
    x_fwd, y_left = z_cam, -x_cam

    rx, ry, yaw = cam_pose
    return (rx + math.cos(yaw) * x_fwd - math.sin(yaw) * y_left,
            ry + math.sin(yaw) * x_fwd + math.cos(yaw) * y_left)


class VamosPolicy:
    """Asks the VLM for paths, scores them, returns the winner in map frame."""

    def __init__(self, cam, is_free, dream=None, url=DEFAULT_URL, num_samples=5,
                 timeout=120):
        self.cam = cam
        self.is_free = is_free          # (x, y) -> bool, stands in for the affordance MLP
        self.dream = dream              # dreaming.Dream, or None for geometry only
        self.url = url
        self.num_samples = num_samples
        self.timeout = timeout
        self.last = None
        self.safety = 1.0               # factor of the path currently being followed
        # Last call's candidates, each with what became of it: (path, factor,
        # verdict), verdict "off-map", "gate" or "pass". Factor is None where
        # the dream never ran. Shadow mode logs this; nothing steers by it.
        self.verdicts = []
        self.stats = {"calls": 0, "failures": 0, "rejected": 0,
                      "rejected_by_gate": 0, "safety_sum": 0.0, "chosen": 0}

    def available(self):
        try:
            return requests.get(f"{self.url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    def _request(self, image, prompt):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        buf.seek(0)
        r = requests.post(
            f"{self.url}/predict",
            files={"image": ("frame.png", buf, "image/png")},
            data={"text_prompt": prompt, "num_samples": self.num_samples,
                  "max_tokens": 20},
            timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return d.get("trajectories") or [] if d.get("success") else []

    def traversable(self, path, step=0.1):
        """Does the drawn line itself stay in free space?

        The cheap first pass. Sampling along each segment, not just at the
        returned points: they are 0.5-2 m apart, so a point-only test happily
        passes a path that cuts straight through a wall between two free
        samples.
        """
        if len(path) < 2:
            return False
        for a, b in zip(path, path[1:]):
            n = max(int(math.dist(a, b) / step), 1)
            for k in range(n + 1):
                x = a[0] + (b[0] - a[0]) * k / n
                y = a[1] + (b[1] - a[1]) * k / n
                if not self.is_free(x, y):
                    return False
        return True

    def progress(self, path, destination):
        """How much closer to the map's destination this path's end would get us."""
        return -math.hypot(path[-1][0] - destination[0], path[-1][1] - destination[1])

    def plan(self, image, prompt, cam_pose, destination, pose=None):
        """Returns (chosen_path, all_paths, safety) in map coordinates.

        `pose` is the body pose the imagined rollouts start from; it defaults
        to the camera pose, which is the same heading half a head further on.
        """
        pose = pose if pose is not None else cam_pose
        self.stats["calls"] += 1
        self.verdicts = []
        try:
            pixel_paths = self._request(image, prompt)
        except (requests.RequestException, ValueError):
            self.stats["failures"] += 1
            self.safety = 0.0
            return None, [], 0.0

        paths = []
        for pp in pixel_paths:
            pts = [g for g in (pixel_to_ground(u, v, self.cam, cam_pose) for u, v in pp)
                   if g is not None]
            # The decoder pads with the last token, so drop repeats.
            dedup = [p for k, p in enumerate(pts) if k == 0 or math.dist(p, pts[k - 1]) > 0.05]
            if len(dedup) >= 2:
                paths.append(dedup)

        # Cheap test first: no sense imagining a path already through a wall.
        walkable = [p for p in paths if self.traversable(p)]
        self.stats["rejected"] += len(paths) - len(walkable)
        self.verdicts = [(p, None, "off-map") for p in paths if p not in walkable]

        scored = []
        for p in walkable:
            if self.dream is None:
                scored.append((1.0, p, {}))
                self.verdicts.append((p, None, "pass"))
                continue
            factor, why = self.dream.factor(p, pose)
            if factor < GATE:
                self.stats["rejected_by_gate"] += 1
                self.stats["rejected"] += 1
                self.verdicts.append((p, factor, "gate"))
                continue
            scored.append((factor, p, why))
            self.verdicts.append((p, factor, "pass"))

        if not scored:
            self.safety = 0.0
            return None, paths, 0.0

        # Safest first, progress second -- banded, so a hundredth of safety
        # does not outrank getting somewhere.
        factor, best, _ = max(scored, key=lambda t: (round(t[0] / BAND),
                                                     self.progress(t[1], destination)))
        self.last, self.safety = best, factor
        self.stats["safety_sum"] += factor
        self.stats["chosen"] += 1
        return best, paths, factor

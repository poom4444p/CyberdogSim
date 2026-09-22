"""VAMOS in the loop: image + goal pixel -> candidate paths -> one chosen path.

The VLM runs as a separate process (VAMOS/server/vlm_server.py) because it
needs torch and this environment deliberately does not have it -- the same
split the real robot has, where the VLM is a service rather than an import.

"AI proposes, simple code disposes": VAMOS returns five candidate paths and a
deterministic scorer picks one. On the real system that scorer is the
affordance MLP; here it is the ground-truth map, which is honest for a PoC
and gives the MLP something to be measured against.
"""
import io
import math

import requests
from PIL import Image

DEFAULT_URL = "http://127.0.0.1:8009"


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

    def __init__(self, cam, is_free, url=DEFAULT_URL, num_samples=5, timeout=120):
        self.cam = cam
        self.is_free = is_free          # (x, y) -> bool, stands in for the affordance MLP
        self.url = url
        self.num_samples = num_samples
        self.timeout = timeout
        self.last = None
        self.stats = {"calls": 0, "failures": 0, "rejected": 0}

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

    def score(self, path, carrot, step=0.1):
        """Traversability first, then progress toward the map's carrot.

        Returns None for a path that leaves free space -- the confidence gate.
        Sampling along each segment, not just at the returned points: they are
        0.5-2 m apart, so a point-only test happily passes a path that cuts
        straight through a wall between two free samples.
        """
        if len(path) < 2:
            return None
        for a, b in zip(path, path[1:]):
            n = max(int(math.dist(a, b) / step), 1)
            for k in range(n + 1):
                x = a[0] + (b[0] - a[0]) * k / n
                y = a[1] + (b[1] - a[1]) * k / n
                if not self.is_free(x, y):
                    return None
        end = path[-1]
        return -math.hypot(end[0] - carrot[0], end[1] - carrot[1])

    def plan(self, image, prompt, cam_pose, carrot):
        """Returns (chosen_path, all_paths) in map coordinates; path may be None."""
        self.stats["calls"] += 1
        try:
            pixel_paths = self._request(image, prompt)
        except (requests.RequestException, ValueError):
            self.stats["failures"] += 1
            return None, []

        paths = []
        for pp in pixel_paths:
            pts = [g for g in (pixel_to_ground(u, v, self.cam, cam_pose) for u, v in pp)
                   if g is not None]
            # The decoder pads with the last token, so drop repeats.
            dedup = [p for k, p in enumerate(pts) if k == 0 or math.dist(p, pts[k - 1]) > 0.05]
            if len(dedup) >= 2:
                paths.append(dedup)

        scored = [(self.score(p, carrot), p) for p in paths]
        ok = [(s, p) for s, p in scored if s is not None]
        self.stats["rejected"] += len(scored) - len(ok)
        if not ok:
            return None, paths
        best = max(ok, key=lambda sp: sp[0])[1]
        self.last = best
        return best, paths

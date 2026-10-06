"""The affordance MLP: an elevation patch and a target in, three scores out.

    input   four NY x NX maps -- the patch (heights, metres above the ground
            under the dog), which of its cells were seen at all, which are on
            the way to the target (data.path_cells), and each cell's largest
            height jump to a neighbour (`jumps`) -- and the target (dx, dy):
            4 * NY * NX + 2 = 1766 numbers
    output  one score per class in data.CLASS_NAMES; softmax makes them
            probabilities, the largest is the answer

Unseen cells (NaN) go in as 0 with their "seen" bit off, so the network can
tell "flat" from "don't know". That matters at run time: Isaac's scanner sees
every cell, the twin's LiDAR far fewer (scripts/isaac/README.md, known gaps),
which is why training hides cells at random (`hide`).

The path map is there because the first network, without it, missed 38% of
the edges on the path: it had to work out from two numbers which cells lie
between the dog and the target, and a small MLP does that badly. The dog can
compute the path at run time exactly as the label does, so it is not a hint
the robot will lack -- it is the question, put where the network can see it.

The jump map, for the same reason: an MLP sees 441 heights as 441 unrelated
numbers, and has to learn for every pair of neighbours that a big difference
is an edge. With the path map alone it still called 17% of not-walkable
trials followable; given the jumps, 8%, and 93.8% right on followable-or-not
on tiles it never saw (a small CNN, which has neighbours built in, did about
as well -- 93.3% -- confirming it was the representation, not the size).

Only the ground near the walk is shown (VIEW_HALF_W): everything further
than that from the line to the target is blanked to "unseen" before any map
is made. Isaac's terrain has no walls, so the network learned that tall
things anywhere in view mean stairs or boxes -- and in the twin's corridors,
walking 0.55 m from a wall, it refused 59% of open floor on the first
shadow run. The question is about the walk, so the input is the walk.

Kept apart from data.py so that file stays numpy-only for the Isaac
collector; only training and the runtime need torch.
"""
import numpy as np
import torch
from torch import nn

from cyberdog.affordance import data

H_CLIP = 0.5            # metres: anything higher or lower is "a lot", the same to us
N_CELLS = data.NY * data.NX
N_IN = 4 * N_CELLS + 2

# Refuse when the network gives "not walkable" at least this probability,
# whatever else it rates higher. Chosen for the people this is for: on the
# held-out tiles of run_20261006_132140, 0.5 (plain argmax) called 10.8% of
# not-walkable trials followable; 0.2 calls 5.4%, at the price of refusing
# 10.5% of followable ones (and 92.0% right on followable-or-not, spec > 90%).
# Too careful is a detour; not careful enough is a fall.
REFUSE_P = 0.2
JUMP_SCALE = 10.0       # a 0.1 m jump reads as 1
VIEW_HALF_W = 0.5       # metres either side of the walk the network may see


def jumps(patch):
    """(N, NY, NX): each cell's largest height jump to a seen 4-neighbour, metres.

    Only between two seen cells: next to an unseen one is "don't know", not an
    edge -- the border of what the LiDAR saw would otherwise read as a cliff."""
    p = np.asarray(patch, dtype=np.float32)
    out = np.zeros_like(p)
    with np.errstate(invalid="ignore"):
        for axis in (1, 2):
            d = np.abs(np.diff(p, axis=axis))
            d = np.where(np.isfinite(d), d, 0.0)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            out[tuple(lo)] = np.maximum(out[tuple(lo)], d)
            out[tuple(hi)] = np.maximum(out[tuple(hi)], d)
    return out


def features(patch, target):
    """(N, N_IN) float32 network input from (N, NY, NX) patches and (N, 2) targets."""
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    patch = np.where(data.path_cells(target, VIEW_HALF_W),
                     np.asarray(patch, dtype=np.float32), np.nan)
    seen = np.isfinite(patch)
    h = np.clip(np.where(seen, patch, 0.0), -H_CLIP, H_CLIP) / H_CLIP
    on = data.path_cells(target)
    jump = np.clip(jumps(patch) / H_CLIP, 0.0, 1.0) * (H_CLIP * JUMP_SCALE)
    n = len(patch)
    return np.concatenate([h.reshape(n, -1), seen.reshape(n, -1).astype(np.float32),
                           on.reshape(n, -1).astype(np.float32), jump.reshape(n, -1),
                           target], axis=1)


def hide(x, rng, max_frac=0.5):
    """Training only: hide a random share (0..max_frac) of each sample's cells.

    In place on a (N, N_IN) torch batch: the height goes to 0 and the seen bit
    off, as if the LiDAR had not hit them."""
    n_cells = N_CELLS
    frac = torch.rand(len(x), 1, device=x.device, generator=rng) * max_frac
    gone = torch.rand(len(x), n_cells, device=x.device, generator=rng) < frac
    x[:, :n_cells][gone] = 0.0
    x[:, n_cells:2 * n_cells][gone] = 0.0
    x[:, 3 * n_cells:4 * n_cells][gone] = 0.0
    return x


def mirror(x):
    """The same sample seen left-right mirrored: flip y in the patch and the target.

    The world has no left or right, so the mirrored sample has the same label;
    it doubles what the network sees for free."""
    n_cells = N_CELLS
    out = x.clone()
    for part in (slice(i * n_cells, (i + 1) * n_cells) for i in range(4)):
        grid = x[:, part].reshape(-1, data.NY, data.NX)
        out[:, part] = torch.flip(grid, dims=[1]).reshape(-1, n_cells)
    out[:, -1] = -x[:, -1]                      # target dy
    return out


class AffordanceMLP(nn.Module):
    """1766 -> 256 -> 128 -> 3. Small on purpose: it must run on the robot every tick."""

    def __init__(self, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers, width = [], N_IN
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, len(data.CLASS_NAMES)))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)

    def forward(self, x):
        return self.net(x)


def decide(probs, refuse_p=REFUSE_P):
    """(N,) class per row of (N, 3) probabilities, refusing at refuse_p.

    NOT_WALKABLE when its probability reaches refuse_p; otherwise the more
    likely of walkable and caution."""
    probs = np.asarray(probs)
    out = np.where(probs[:, data.CAUTION] > probs[:, data.WALKABLE],
                   data.CAUTION, data.WALKABLE)
    out[probs[:, data.NOT_WALKABLE] >= refuse_p] = data.NOT_WALKABLE
    return out.astype(np.int8)


@torch.no_grad()
def classify(net, patch, target, refuse_p=REFUSE_P):
    """(classes, probabilities) for raw patches and targets -- the runtime's call."""
    x = torch.as_tensor(features(patch, target), device=next(net.parameters()).device)
    probs = torch.softmax(net(x), dim=1).cpu().numpy()
    return decide(probs, refuse_p), probs


def save(model, path, **info):
    """Weights plus what is needed to rebuild the network and trust it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hidden": model.hidden,
                "classes": data.CLASS_NAMES, **info}, path)


def load(path, device="cpu"):
    """(model in eval mode, saved info)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = AffordanceMLP(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), ckpt

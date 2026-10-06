"""Train the affordance MLP on a run from scripts/isaac/collect_affordance.py.

    python scripts/train_affordance.py                      # newest run
    python scripts/train_affordance.py --epochs 30 --seed 1

Labels are data.classes(): walkable / caution / not walkable. The split is by
terrain *tile*, not by trial: trials on the same tile look nearly identical,
and a test set full of their twins scores far better than the network
really is. So 15% of tiles are locked away for the final test, 15% more
choose when to stop, and the network never sees either while learning.

Reported on the locked-away tiles: accuracy over the three classes, the
spec's yes/no accuracy (followable or not, > 90% wanted), and the mistake
that matters -- not-walkable ground called walkable or caution.
Weights go to models/affordance/mlp.pt (not in git).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cyberdog import paths
from cyberdog.affordance import data, model as M

OUT = paths.MODELS_DIR / "affordance" / "mlp.pt"


def split_by_tile(r, seed, test=0.15, val=0.15):
    """(train, val, test) index arrays, whole tiles to each."""
    tile = r["terrain_level"].astype(np.int64) * 1000 + r["terrain_type"]
    tiles = np.unique(tile)
    rng = np.random.default_rng(seed)
    rng.shuffle(tiles)
    n_test, n_val = int(round(len(tiles) * test)), int(round(len(tiles) * val))
    test_t, val_t = set(tiles[:n_test]), set(tiles[n_test:n_test + n_val])
    which = np.array([2 if t in test_t else 1 if t in val_t else 0 for t in tile])
    return [np.flatnonzero(which == k) for k in range(3)], len(tiles)


def confusion(y, p):
    """3x3 counts: rows the true class, columns the predicted one."""
    c = np.zeros((3, 3), dtype=np.int64)
    np.add.at(c, (y, p), 1)
    return c


@torch.no_grad()
def predict(net, x, device, batch=8192, refuse_p=None):
    """Classes for feature rows: argmax, or M.decide at refuse_p if given."""
    net.eval()
    probs = np.concatenate([torch.softmax(net(torch.as_tensor(x[a:a + batch], device=device)), 1)
                            .cpu().numpy() for a in range(0, len(x), batch)])
    return probs.argmax(1) if refuse_p is None else M.decide(probs, refuse_p)


def report(name, y, p):
    c = confusion(y, p)
    acc = np.trace(c) / c.sum()
    follow_true, follow_pred = y != data.NOT_WALKABLE, p != data.NOT_WALKABLE
    yes_no = (follow_true == follow_pred).mean()
    danger = (~follow_true & follow_pred).sum() / max((~follow_true).sum(), 1)
    nuisance = (follow_true & ~follow_pred).sum() / max(follow_true.sum(), 1)
    print(f"\n{name}: {len(y)} trials")
    print(f"  accuracy, 3 classes            {acc:6.1%}")
    print(f"  accuracy, followable or not    {yes_no:6.1%}   (spec: > 90%)")
    print(f"  DANGER: not walkable -> called followable   {danger:6.1%} of not-walkable trials")
    print(f"  nuisance: followable -> called not walkable {nuisance:6.1%} of followable trials")
    print(f"  confusion (rows = truth, columns = network's answer):")
    print(f"  {'':16s}" + "".join(f"{n:>15s}" for n in data.CLASS_NAMES))
    for i, n in enumerate(data.CLASS_NAMES):
        row = c[i] / max(c[i].sum(), 1)
        print(f"  {n:16s}" + "".join(f"{v:15.1%}" for v in row) + f"   ({c[i].sum()} trials)")
    return {"acc": float(acc), "yes_no": float(yes_no), "danger": float(danger),
            "nuisance": float(nuisance), "confusion": c.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", help="run folder (default: the newest)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hide", type=float, default=0.0,
                    help="largest share of cells hidden per training sample (0 = none). "
                         "Off by default: Isaac sees every cell, and hiding the edge "
                         "a label is about teaches guessing (it cost 1.5 points). "
                         "For the twin's sparse LiDAR, see scripts/isaac/README.md")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    run = Path(args.run) if args.run else sorted((paths.DATASETS_DIR / "affordance").glob("run_*"))[-1]
    r, _ = data.load_shards(run)
    y = data.classes(r).astype(np.int64)
    x = M.features(r["patch"], r["target"])
    (tr, va, te), n_tiles = split_by_tile(r, args.seed)
    print(f"{run.name}: {len(y)} trials on {n_tiles} terrain tiles -> "
          f"train {len(tr)}, val {len(va)}, test {len(te)} (whole tiles each)")
    print("classes in training: " + ", ".join(
        f"{n} {(y[tr] == i).mean():.0%}" for i, n in enumerate(data.CLASS_NAMES)))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    net = M.AffordanceMLP().to(device)
    # Rarer classes count for more, so "always say not walkable" is not a
    # cheap way to be mostly right.
    freq = np.bincount(y[tr], minlength=3) / len(tr)
    weight = torch.as_tensor(1.0 / (3 * freq), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    xt = torch.as_tensor(x[tr], device=device)
    yt = torch.as_tensor(y[tr], device=device)
    best, best_state, t0 = -1.0, None, time.time()
    print(f"\ntraining on {device}: epoch, loss on training, accuracy on val tiles")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.randperm(len(xt), device=device, generator=gen)
        total = 0.0
        for a in range(0, len(order), args.batch):
            idx = order[a:a + args.batch]
            xb = xt[idx].clone()
            flip = torch.rand(len(xb), device=device, generator=gen) < 0.5
            xb[flip] = M.mirror(xb[flip])
            if args.hide > 0:
                xb = M.hide(xb, gen, max_frac=args.hide)
            loss = loss_fn(net(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        acc = (predict(net, x[va], device) == y[va]).mean()
        mark = ""
        if acc > best:
            best, mark = acc, "  <- best so far"
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  {epoch:3d}   {total / len(tr):.3f}   {acc:6.1%}{mark}")
    net.load_state_dict(best_state)
    print(f"trained in {time.time() - t0:.0f}s; keeping the epoch best on val ({best:.1%})")

    report("VALIDATION tiles (used to pick the epoch)", y[va], predict(net, x[va], device))
    report("TEST tiles, plain argmax", y[te], predict(net, x[te], device))
    result = report(f"TEST tiles, as the dog decides (refuse at P >= {M.REFUSE_P})",
                    y[te], predict(net, x[te], device, refuse_p=M.REFUSE_P))

    M.save(net.cpu(), Path(args.out), run=run.name, seed=args.seed, epochs=args.epochs,
           test=result, refuse_p=M.REFUSE_P, thresholds={"tilt_max": data.TILT_MAX, "step_max": data.STEP_MAX,
                                    "bump_walk": data.BUMP_WALK,
                                    "bump_caution": data.BUMP_CAUTION})
    print(f"\nweights -> {args.out}")


if __name__ == "__main__":
    main()

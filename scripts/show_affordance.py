"""Look at the affordance data: what the dog saw, and what happened next.

Draws one walkable and one not-walkable trial for every terrain type in a run
collected by scripts/isaac/collect_affordance.py, and prints why the trials
that were labelled not walkable were labelled that way.

    python scripts/show_affordance.py                       # newest run
    python scripts/show_affordance.py datasets/affordance/run_20261006_132140
    python scripts/show_affordance.py --seed 3              # other examples

Each picture is the elevation patch seen from above, turned so the dog faces
up the page: the triangle is the dog, the cross is where it was told to go.
Colour is height above the ground under the dog -- red above, blue below,
grey level with it.
"""
import argparse
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cyberdog import paths
from cyberdog.affordance import data

# Diverging: blue below the ground, red above, neutral grey for level.
HEIGHT = LinearSegmentedColormap.from_list(
    "height", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])
H_LIM = 0.30                     # metres either way the colour scale covers
INK, INK_2 = "#0b0b0b", "#52514e"


def reason(r, k):
    """Why trial k got the label it got, in words."""
    if r["fell"][k]:
        return "fell over"
    if not r["reached"][k]:
        return "did not arrive"
    if r["z_span"][k] > data.STEP_MAX:
        return f"body rose/dropped {r['z_span'][k] * 100:.0f} cm"
    if r["max_tilt"][k] > data.TILT_MAX:
        return f"tilted {np.degrees(r['max_tilt'][k]):.0f}°"
    return (f"easy: {r['z_span'][k] * 100:.0f} cm, "
            f"{np.degrees(r['max_tilt'][k]):.0f}°")


def why_not(r, k):
    """The first rule a not-walkable trial broke (for the summary table)."""
    if r["fell"][k]:
        return "fell"
    if not r["reached"][k]:
        return "did not arrive"
    if r["z_span"][k] > data.STEP_MAX:
        return "rose/dropped > 8 cm"
    return "tilted > 8.6°"


def draw(ax, r, k, walkable):
    """One trial's patch, dog facing up the page, left on the left."""
    patch = r["patch"][k]                       # [y index, x index], +x forward
    img = patch.T[::-1, ::-1]                   # rows: forward at the top
    half = (data.NY * data.RES) / 2
    extent = (-half, half, data.X0 - data.RES / 2, data.X0 + (data.NX - 0.5) * data.RES)
    ax.imshow(img, cmap=HEIGHT, vmin=-H_LIM, vmax=H_LIM, extent=extent,
              interpolation="nearest")
    ax.plot(0, 0, marker="^", markersize=11, color=INK, markeredgecolor="white",
            markeredgewidth=1.5)
    tx, ty = r["target"][k]
    ax.plot(-ty, tx, marker="X", markersize=11, color=INK, markeredgecolor="white",
            markeredgewidth=1.5)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#0ca30c" if walkable else "#d03b3b")
        side.set_linewidth(3)
    tag = "✓ walkable" if walkable else "✗ not walkable"
    ax.set_title(f"{tag}\n{reason(r, k)}", fontsize=9, color=INK_2, pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", help="run folder (default: the newest)")
    ap.add_argument("--seed", type=int, default=0, help="which examples to pick")
    ap.add_argument("--out", default=str(paths.OUTPUT_DIR / "affordance_examples.png"))
    args = ap.parse_args()

    if args.run:
        run = Path(args.run)
    else:
        runs = sorted((paths.DATASETS_DIR / "affordance").glob("run_*"))
        if not runs:
            raise SystemExit("no runs in datasets/affordance -- see scripts/isaac/README.md")
        run = runs[-1]
    r, metas = data.load_shards(run)
    y = data.label(r)
    names = metas[0]["terrain_columns"]
    rng = np.random.default_rng(args.seed)

    print(f"{run.name}: {len(y)} trials, {y.mean():.0%} labelled walkable")
    print(f"labels: walkable = arrived, did not fall, tilt <= "
          f"{np.degrees(data.TILT_MAX):.1f}°, rise/drop <= {data.STEP_MAX * 100:.0f} cm\n")

    # Group by terrain name: several Isaac columns can share one.
    kind = np.array([names[t] for t in r["terrain_type"]])
    terrains = sorted(set(kind))

    # Why the not-walkable ones are not walkable.
    rules = ["fell", "did not arrive", "rose/dropped > 8 cm", "tilted > 8.6°"]
    print(f"{'terrain':24s} {'trials':>7s} {'walkable':>9s}   " +
          "  ".join(f"{h:>20s}" for h in rules))
    for t in terrains:
        idx = np.flatnonzero(kind == t)
        bad = collections.Counter(why_not(r, k) for k in idx if not y[k])
        print(f"{t:24s} {len(idx):7d} {y[idx].mean():9.0%}   " +
              "  ".join(f"{bad[h] / len(idx):20.0%}" for h in rules))

    fig, axes = plt.subplots(2, len(terrains), figsize=(2.6 * len(terrains), 6.6),
                             facecolor="#fcfcfb")
    for j, t in enumerate(terrains):
        for i, walkable in enumerate((True, False)):
            pool = np.flatnonzero((kind == t) & (y == walkable))
            ax = axes[i, j]
            if not len(pool):
                ax.axis("off")
                continue
            draw(ax, r, rng.choice(pool), walkable)
        axes[0, j].annotate(t.replace("_", " "), (0.5, 1.32), xycoords="axes fraction",
                            ha="center", fontsize=11, color=INK, weight="bold")

    fig.suptitle("What the dog saw before each trial (seen from above, dog facing up)",
                 fontsize=13, color=INK, y=0.995)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=HEIGHT, norm=plt.Normalize(-H_LIM, H_LIM)),
                        ax=axes, orientation="horizontal", fraction=0.04, pad=0.03,
                        aspect=50)
    cbar.set_label("height above the ground under the dog (m)   "
                   "▲ = dog    ✖ = target", color=INK_2)
    cbar.outline.set_visible(False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\npicture -> {out}")


if __name__ == "__main__":
    main()

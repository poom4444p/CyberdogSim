"""Look at the affordance data: what the dog saw, and what label it got.

Draws one walkable, one caution and one not-walkable trial for every terrain
type in a run collected by scripts/isaac/collect_affordance.py, and prints
the share of each class per terrain and why the not-walkable ones are.

    python scripts/show_affordance.py                       # newest run
    python scripts/show_affordance.py datasets/affordance/run_20261006_132140
    python scripts/show_affordance.py --seed 3              # other examples

Each picture is the elevation patch seen from above, turned so the dog faces
up the page: the triangle is the dog, the cross is where it was told to go,
the line its path. Colour is height above the ground under the dog -- red
above, blue below, grey level with it. The rule is data.classes().
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
DEG = "°"

# Each class's row: its tag, and the border colour (status colours, with the
# tag's symbol and words alongside -- never colour alone).
ROW = {data.WALKABLE: ("✓ walkable", "#0ca30c"),
       data.CAUTION: ("⚠ caution", "#c98500"),
       data.NOT_WALKABLE: ("✗ not walkable", "#d03b3b")}
RULES = ["fell", "did not arrive", "rose/dropped", "tilted", "edge on path"]


def why(r, k, bump):
    """The first rule trial k broke, as one of RULES, or None if it broke none."""
    if r["fell"][k]:
        return "fell"
    if not r["reached"][k]:
        return "did not arrive"
    if r["z_span"][k] > data.STEP_MAX:
        return "rose/dropped"
    if r["max_tilt"][k] > data.TILT_MAX:
        return "tilted"
    if bump[k] > data.BUMP_CAUTION:
        return "edge on path"
    return None


def reason(r, k, bump):
    """Why trial k got its class, in words, with the number that decided it."""
    rule = why(r, k, bump)
    if rule == "rose/dropped":
        return f"body rose/dropped {r['z_span'][k] * 100:.0f} cm"
    if rule == "tilted":
        return f"tilted {np.degrees(r['max_tilt'][k]):.0f}{DEG}"
    if rule in ("fell", "did not arrive"):
        return "fell over" if rule == "fell" else rule
    return f"bumps on path {bump[k] * 100:.1f} cm"


def draw(ax, r, k, cls, bump):
    """One trial's patch, dog facing up the page, left on the left, path drawn."""
    patch = r["patch"][k]                       # [y index, x index], +x forward
    img = patch.T[::-1, ::-1]                   # rows: forward at the top
    half = (data.NY * data.RES) / 2
    extent = (-half, half, data.X0 - data.RES / 2, data.X0 + (data.NX - 0.5) * data.RES)
    ax.imshow(img, cmap=HEIGHT, vmin=-H_LIM, vmax=H_LIM, extent=extent,
              interpolation="nearest")
    tx, ty = r["target"][k]
    ax.plot([0, -ty], [0, tx], color=INK, lw=2, alpha=0.7)
    for x, y, m in ((0, 0, "^"), (-ty, tx, "X")):
        ax.plot(x, y, marker=m, markersize=11, color=INK, markeredgecolor="white",
                markeredgewidth=1.5)
    ax.set_xticks([]), ax.set_yticks([])
    tag, colour = ROW[cls]
    for side in ax.spines.values():
        side.set_color(colour)
        side.set_linewidth(3)
    ax.set_title(f"{tag}\n{reason(r, k, bump)}", fontsize=9, color=INK_2, pad=4)


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
    bump = data.path_bump(r)
    c = data.classes(r, bump=bump)
    names = metas[0]["terrain_columns"]
    rng = np.random.default_rng(args.seed)

    print(f"{run.name}: {len(c)} trials -- " + ", ".join(
        f"{(c == i).mean():.0%} {n}" for i, n in enumerate(data.CLASS_NAMES)))
    print(f"rule: arrived, did not fall, tilt <= {np.degrees(data.TILT_MAX):.0f}{DEG}, "
          f"rise/drop <= {data.STEP_MAX * 100:.0f} cm; bumps on the path <= "
          f"{data.BUMP_WALK * 100:.0f} cm walkable, <= {data.BUMP_CAUTION * 100:.0f} cm "
          f"caution\n")

    # Group by terrain name: several Isaac columns can share one.
    kind = np.array([names[t] for t in r["terrain_type"]])
    terrains = sorted(set(kind))

    print(f"{'terrain':22s} {'trials':>7s} {'walk':>6s} {'caution':>8s} {'not':>5s}   "
          "not walkable because: " + ", ".join(RULES))
    for t in terrains:
        idx = np.flatnonzero(kind == t)
        bad = collections.Counter(why(r, k, bump) for k in idx if c[k] == data.NOT_WALKABLE)
        share = [(c[idx] == i).mean() for i in range(3)]
        print(f"{t:22s} {len(idx):7d} {share[0]:6.0%} {share[1]:8.0%} {share[2]:5.0%}   " +
              "  ".join(f"{bad[h] / len(idx):4.0%}" for h in RULES))

    fig, axes = plt.subplots(3, len(terrains), figsize=(2.6 * len(terrains), 9.8),
                             facecolor="#fcfcfb")
    for j, t in enumerate(terrains):
        for i in (data.WALKABLE, data.CAUTION, data.NOT_WALKABLE):
            ax = axes[i, j]
            pool = np.flatnonzero((kind == t) & (c == i))
            if not len(pool):
                ax.text(0.5, 0.5, f"no {data.CLASS_NAMES[i]}\ntrials here",
                        ha="center", va="center", color=INK_2, fontsize=9,
                        transform=ax.transAxes)
                ax.set_xticks([]), ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_color("#d8d7d2")
                continue
            draw(ax, r, rng.choice(pool), i, bump)
        axes[0, j].annotate(t.replace("_", " "), (0.5, 1.32), xycoords="axes fraction",
                            ha="center", fontsize=11, color=INK, weight="bold")

    fig.suptitle("What the dog saw before each trial (seen from above, dog facing up)",
                 fontsize=13, color=INK, y=0.995)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=HEIGHT, norm=plt.Normalize(-H_LIM, H_LIM)),
                        ax=axes, orientation="horizontal", fraction=0.03, pad=0.03,
                        aspect=50)
    cbar.set_label("height above the ground under the dog (m)   "
                   "▲ = dog    ✖ = target    line = its path", color=INK_2)
    cbar.outline.set_visible(False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\npicture -> {out}")


if __name__ == "__main__":
    main()

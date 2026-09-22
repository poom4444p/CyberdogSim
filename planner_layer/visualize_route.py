"""Draw a planned trip on the floor maps, to check routes by eye.

The printed output of main_planner.py can't show whether a route cuts
through a wall. This draws every leg on its floor's occupancy grid, one
panel per floor visited, with the room labels, checkpoints and stops.

A correct route stays in the corridor, enters rooms only through door gaps,
goes via the stairs (west end) to change floor, and ends inside the right room.

Usage (from anywhere):
    # Place names (canonical names from locations.json); "name@N" picks floor N
    python3 planner_layer/visualize_route.py "physics lab"
    python3 planner_layer/visualize_route.py "restroom@2" library cafeteria

    # A full natural-language command, through splitter + floor parser + model
    python3 planner_layer/visualize_route.py --command "I need to pee upstairs, then the caf"

    Options: --start "main entrance" --start-floor 1 --out my_route.png
    Output defaults to planner_layer/route_images/<stops>.png

Running on Linux:
    pip install numpy pyyaml pillow scipy matplotlib
    (--command also needs: pip install torch transformers peft trl datasets accelerate)
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "input_treating_layer"))

LEG_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]


def stops_from_names(names):
    """["restroom@2", "library"] -> [("restroom", 2), ("library", None)]"""
    out = []
    for n in names:
        name, _, floor = n.partition("@")
        out.append((name.strip().lower(), int(floor) if floor else None))
    return out


def stops_from_command(command, start_floor, router_floor_hint):
    """Run the same front-end as main_planner.py. Relative floors ("upstairs")
    are resolved later, once the previous stop's floor is known, so this
    returns (stop_text, None) pairs plus the parse function."""
    from command_splitter import split_destinations
    return split_destinations(command)


def main():
    ap = argparse.ArgumentParser(description="Draw a planned trip on the floor maps")
    ap.add_argument("stops", nargs="*", help='place names, optionally "name@floor"')
    ap.add_argument("--command", help="natural-language command (runs the NLU model)")
    ap.add_argument("--start", default="main entrance")
    ap.add_argument("--start-floor", type=int, default=1)
    ap.add_argument("--out", help="output PNG path")
    args = ap.parse_args()
    if not args.stops and not args.command:
        ap.error("give place names or --command")

    parse_command = None
    if args.command:
        # Load torch before numpy/PIL/matplotlib -- libomp crash on macOS otherwise.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from infer import load_model, parse_command
        load_model()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from building_router import BuildingRouter
    from floor_parser import extract_floor

    router = BuildingRouter()
    start = router.resolve(args.start, args.start_floor, (0, 0), floor=args.start_floor)
    if start is None:
        sys.exit(f"Start '{args.start}' is not on floor {args.start_floor}.")

    here_name, here_floor, here_xy = args.start, start["floor"], start["xy"]
    stops = []      # (label, target entry)
    legs = []       # (stop index, floor, Route)

    if args.command:
        from command_splitter import split_destinations
        texts = split_destinations(args.command)
    else:
        texts = stops_from_names(args.stops)

    for i, item in enumerate(texts, 1):
        if args.command:
            floor, cleaned = extract_floor(item, current_floor=here_floor)
            name = (parse_command(cleaned).get("target_location") or "").strip().lower()
            print(f"Stop {i}: \"{item}\" -> {name!r}" + (f" on floor {floor}" if floor else ""))
        else:
            name, floor = item
        if not router.floors_of(name):
            sys.exit(f"Stop {i}: unknown place '{name}'.")
        target = router.resolve(name, here_floor, here_xy, floor=floor)
        if target is None:
            sys.exit(f"Stop {i}: no {name} on floor {floor} (it is on {router.floors_of(name)}).")
        trip = router.plan(here_floor, here_xy, target)
        if trip is None:
            sys.exit(f"Stop {i}: no route from {here_name} to {name}.")
        stops.append((name, target))
        legs += [(i, f, route) for f, route in trip]
        here_name, here_floor, here_xy = name, target["floor"], target["xy"]

    # One panel per floor visited, in the order first visited
    floors = []
    for _, f, _ in legs:
        if f not in floors:
            floors.append(f)
    if start["floor"] not in floors:
        floors.insert(0, start["floor"])

    fig, axes = plt.subplots(len(floors), 1, figsize=(13, 5.2 * len(floors)), squeeze=False)
    panel = {f: axes[k][0] for k, f in enumerate(floors)}

    for f, ax in panel.items():
        g = router.grids[f]
        extent = [g.origin[0], g.origin[0] + g.width * g.resolution,
                  g.origin[1], g.origin[1] + g.height * g.resolution]
        ax.imshow(g.grid == 100, cmap="gray_r", origin="lower", extent=extent, interpolation="nearest")
        for place, entries in router.locations.items():
            for e in entries:
                if e["floor"] == f and place != "hallway":
                    ax.text(e["xy"][0], e["xy"][1], place.replace(" engineering", "\neng."),
                            ha="center", va="center", fontsize=7, color="0.45")
        ax.set_title(f"Floor {f}")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    for stop_i, f, route in legs:
        ax = panel[f]
        color = LEG_COLORS[(stop_i - 1) % len(LEG_COLORS)]
        xs, ys = zip(*route.raw_path)
        ax.plot(xs, ys, "-", color=color, lw=2.2, label=f"to stop {stop_i}: {stops[stop_i - 1][0]}")
        for cp in route.checkpoints:
            ax.plot(*cp.position, "o", color=color, ms=5, mec="black", mew=0.5)
            if cp.announcements:
                ax.annotate(cp.announcements[0], cp.position, textcoords="offset points",
                            xytext=(6, 6), fontsize=7, color=color)

    ax = panel[start["floor"]]
    ax.plot(*start["xy"], marker="s", color="black", ms=11)
    ax.annotate("START", start["xy"], textcoords="offset points", xytext=(-14, -16), fontsize=8, weight="bold")
    for i, (name, t) in enumerate(stops, 1):
        ax = panel[t["floor"]]
        ax.plot(*t["xy"], marker="*", color=LEG_COLORS[(i - 1) % len(LEG_COLORS)], ms=18, mec="black")
        ax.annotate(str(i), t["xy"], ha="center", va="center", fontsize=7, weight="bold")

    for ax in panel.values():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    title = " → ".join([args.start] + [n for n, _ in stops])
    fig.suptitle(args.command and f'"{args.command}"\n{title}' or title, fontsize=11)
    fig.tight_layout()

    out = args.out
    if not out:
        slug = re.sub(r"[^a-z0-9]+", "_", "_".join(n for n, _ in stops)).strip("_")
        out = os.path.join(HERE, "route_images", f"{slug}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

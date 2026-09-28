import json
import argparse


import math

# The robot starts at the (synthetic) building's main entrance on floor 1.
# See scripts/mapping/generate_building.py -- placeholder layout until the real scan.
START_LOCATION = "main entrance"
START_FLOOR = 1


def print_vamos_steps(legs, cam, project_route, vamos_prompt, step):
    """For each checkpoint, project the destination point ahead into the camera
    image and build the VAMOS prompt. No real robot yet, so we assume it
    stands on each checkpoint facing along the route. Returns the last step number."""
    for floor, route in legs:
        print(f"    -- Floor {floor}: {route}")
        waypoints = [cp.position for cp in route.checkpoints]
        for i, cp in enumerate(route.checkpoints):
            step += 1
            ahead = waypoints[i + 1] if i + 1 < len(waypoints) else cp.position
            yaw = math.atan2(ahead[1] - cp.position[1], ahead[0] - cp.position[0])
            pose = (cp.position[0], cp.position[1], yaw)
            result = project_route(pose, waypoints[i + 1:], cam)

            if result["state"] == "TRACK":
                line = vamos_prompt(result)
            else:
                line = f"[{result['state']}]" + (f" turn {result['turn']}" if "turn" in result else "")
            if cp.announcements:
                line += f" {cp.announcements[0]}."
            print(f"    Step {step}: {line}")
    return step


def main():
    arg_parser = argparse.ArgumentParser(description="CyberDog Planner Layer (PoC)")
    arg_parser.add_argument(
        "command", nargs="?", default="Take me to the library, then the cafeteria.",
        help="Natural-language command to send through the Input Treating Layer",
    )
    args = arg_parser.parse_args()

    print("--------------------------------------------------")
    print(" CYBERDOG PLANNER LAYER (POC) ")
    print("--------------------------------------------------")

    # 1. Split multi-stop commands with rules ("library, then cafeteria" ->
    # two single-stop commands); the model was trained on one stop at a time.
    print(f"\n[1] User Command: \"{args.command}\"")
    from cyberdog.language.command_splitter import split_destinations
    from cyberdog.language.floor_parser import extract_floor
    from cyberdog.language.generate_dataset import UNKNOWN_LOCATION
    stops = split_destinations(args.command)
    if len(stops) > 1:
        print(f"    Split into {len(stops)} stops:")
        for n, s in enumerate(stops, 1):
            print(f"      {n}. \"{s}\"")

    # 2. Load the model first, then the maps: importing the map stack
    # (numpy/PIL) before torch triggers a libomp crash on macOS.
    print("\n[2] Loading language model, building maps and A* planner...")
    try:
        from cyberdog.language.infer import load_model, parse_command
    except ImportError as missing:
        # The twin's environment deliberately has no torch; the LoRA parser is
        # a separate, heavier install. Say so, rather than showing a traceback
        # from three imports down.
        raise SystemExit(
            f"the language layer needs its optional dependencies ({missing.name} "
            f"is missing).\n  pip install -e \".[language]\"\n"
            f"To plan a route without the model, name the destination directly:\n"
            f"  python -m cyberdog.planning.visualize_route \"cafeteria\"")
    load_model()
    from cyberdog.planning.building_router import BuildingRouter, NoAccessibleRoute
    from cyberdog.planning.checkpoint_projector import load_camera_config, project_route, vamos_prompt
    router = BuildingRouter()
    cam = load_camera_config()
    print(f"    Loaded floors {router.floors}, {len(router.locations)} named locations.")

    start = router.resolve(START_LOCATION, START_FLOOR, (0, 0))
    here_name, here_floor, here_xy = START_LOCATION, start["floor"], start["xy"]
    print(f"    Start: {START_LOCATION}, floor {here_floor}, {tuple(here_xy)}")

    # 3. One stop at a time; each stop starts where the previous one ended,
    # so "upstairs" in stop 2 is relative to the floor stop 1 ended on.
    step = 0
    for n, stop_text in enumerate(stops, 1):
        print(f"\n[3.{n}] Stop {n}/{len(stops)}: \"{stop_text}\"")

        # The model wasn't trained on floor phrases, so strip them out with
        # rules first ("pee on the 2nd floor" -> floor=2, "pee").
        # "upstairs"/"downstairs" use the robot's current floor; on the real
        # robot that comes from localization.
        requested_floor, command = extract_floor(stop_text, current_floor=here_floor)
        if requested_floor is not None:
            print(f"    Floor phrase found: floor {requested_floor} -> sending \"{command}\" to the model")

        # Rules before the model: a no-go place named out loud is taken from
        # the raw text, because the parser was fine-tuned on destinations and
        # maps an unknown word onto the nearest one it knows -- "the stairs"
        # comes back as "hallway", and the stairs are never mentioned again.
        named = router.hazard_named(stop_text)
        if named is not None:
            target_name, nlu_output = named[0], {"target_location": named[0],
                                                 "task": "navigation", "query": None}
            print(f"    No-go place named in the command: {target_name} "
                  f"(rules, not the model)")
        else:
            nlu_output = parse_command(command)
            print("    Parsed by Input Treating Layer: "
                  + json.dumps(nlu_output, ensure_ascii=False))
            target_name = nlu_output.get("target_location") or ""
        if target_name.strip().lower() == UNKNOWN_LOCATION:
            print(f"    I don't know where that is -- \"{stop_text}\" names no place "
                  f"in this building. Staying here.")
            return
        if requested_floor is not None and requested_floor not in router.floors:
            print(f"    Error: there is no floor {requested_floor} (you are on floor {here_floor}; "
                  f"the building has floors {router.floors[0]}-{router.floors[-1]}). Stopping.")
            return
        floors = router.floors_of(target_name)
        if not floors:
            print(f"    Error: unknown target location '{target_name}' "
                  f"(not in data/building/locations.json). Stopping.")
            return
        # A refusal is not an error -- it is the answer, and it is the one the
        # user hears. A floor the lift does not serve has no route at all; a
        # place inside a no-go zone has one to the corridor outside it.
        try:
            approached = (router.approach(target_name, here_floor, here_xy,
                                          floor=requested_floor)
                          if router.hazard_place(target_name) else None)
            if approached is not None:
                target, line = approached
                print(f"    {line}")
            else:
                target = router.resolve(target_name, here_floor, here_xy,
                                        floor=requested_floor)
            if target is None:
                print(f"    Error: there is no {target_name} on floor {requested_floor} "
                      f"(it is on floor(s) {', '.join(map(str, floors))}). Stopping.")
                return

            print(f"    Route: {here_name} (floor {here_floor}) -> {target_name} "
                  f"(floor {target['floor']}, {tuple(target['xy'])})")
            legs = router.plan(here_floor, here_xy, target)
        except NoAccessibleRoute as refusal:
            print(f"    Refused: {refusal}")
            return
        if legs is None:
            print("    Error: Route not found. Obstacles blocked the way. Stopping.")
            return
        if approached is not None:
            # Not "destination reached": the dog is outside what was asked
            # for, and the arrival line is where that gets said.
            legs[-1][1].checkpoints[-1].announcements = [line]

        print("    Passing Instructions to VAMOS Layer:")
        step = print_vamos_steps(legs, cam, project_route, vamos_prompt, step)

        if nlu_output.get("task") == "visual_qa":
            print(f"    Note: question \"{nlu_output.get('query')}\" is not answered yet "
                  f"(visual QA isn't wired up); navigation only.")

        here_name, here_floor, here_xy = target_name, target["floor"], target["xy"]

    print(f"\nPipeline simulation complete: {len(stops)} stop(s), {step} step(s).\n")


if __name__ == '__main__':
    main()

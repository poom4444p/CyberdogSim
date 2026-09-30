"""Drive the dog through the whole three-floor building, not one floor.

run_demo.py and record_demo.py each load floor1.xml and stop there,
so a route that crosses floors had nowhere to go. This runs the same stack
against output/scene_cache/building.xml -- all three floors and the lift between
them in one model -- so "take me to the server room" is one continuous run:
corridor, lift, corridor, lift, corridor.

The lift, never the stairs. The person holding the handle cannot see the steps
and has no free hand for a rail, so the stairwells are stop zones the planner
refuses to route through and this driver halts in if the dog ever drifts into
one. At the lift the dog stops and says so, because it cannot press a call
button -- a human does that, and presses continue.

    python -m cyberdog.sim.scene.build_scene --building          # once, to make the scene
    python -m cyberdog.sim.run_building "take me to the chemistry lab"
    python -m cyberdog.sim.run_building "go to the library on the 2nd floor, then the server room"
    python -m cyberdog.sim.run_building "restroom on floor 3" --no-video   # fast, headless
    python -m cyberdog.sim.run_building "server room" --nlu                # through the Gemma layer
    python -m cyberdog.sim.run_building "server room" --vamos              # VLM in the steering loop
    python -m cyberdog.sim.run_building "server room" --shadow             # VLM + dream, map still drives
    python -m cyberdog.sim.run_building "server room" --auto-confirm       # don't wait at the lift

Without --nlu the destination is matched against locations.json by name, which
keeps torch out of the process; --nlu runs the real Input Treating Layer.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np


from cyberdog import paths
from cyberdog.planning.building_router import BuildingRouter, NoAccessibleRoute
from cyberdog.planning.checkpoint_projector import (load_camera_config,
                                                    pick_destination, project_route,
                                                    project_to_pixel,
                                                    vamos_prompt)
from cyberdog.sim.control import (GOAL_R, K_W, TURN_ONLY, advance, path_target,
                                  wrap)
from cyberdog.sim.overlay import (ChaseCam, draw_marker, draw_path,
                                  draw_points, label_places, safety_bar)
from cyberdog.sim.robot.mujoco_robot import MujocoRobot
from cyberdog.sim.scene import levels, lift, obstacles, pedestrians
from cyberdog.sim.sensing import perception
from cyberdog.sim.sensing.dreaming import ROBOT_R
from cyberdog.sim.sensing.lidar import MOUNT_H, Lidar
from cyberdog.sim.sensing.perception import (INF, LiveClearance, free_destination,
                                             line_clear)
from cyberdog.sim.sensing.tracking import (CLEAR_R, MAX_RADIUS, MIN_RADIUS, MISS_R,
                                           Tracker, closest_approach, conflict)
from cyberdog.sim.vamos_client import ahead

START_LOCATION = "main entrance"
START_FLOOR = 1
FPS = 20                  # equals the control rate, so the video is real time
WAIT_S = 1.5              # held at the lift doors, and while auto-confirming
ANNOUNCE_R = 2.5          # metres out that a checkpoint's line is spoken
STEP_V = 0.4              # m/s in and out of the car -- slower than corridor pace
REPLAN_EVERY = 20         # ticks between VLM calls when --vamos is on
REPLAN_BLOCKED = 10       # ...and while nothing VAMOS offered clears an obstacle
# Shadow mode: a proposal whose heading is further than this off the map's
# steering target is logged as the dream wanting to go somewhere else.
SHADOW_DISAGREE = math.radians(15)
# Flatter and closer than the single-floor demo: every storey now has a slab
# over it, and the old -28 degree chase camera sat inside the ceiling, which
# rendered as solid grey.
CHASE_D, CHASE_EL = 3.2, -11
# How far a door sign sits off the corridor centreline, towards its own room.
# The corridor is 2.7 m wide, so this puts it essentially on that room's wall --
# and, more to the point, on the correct side of a corridor whose two facing
# rooms share a single door_xy in locations.json.
SIGN_OFF = 1.25
# Azimuth, distance, elevation for the lift -- see frame(). Azimuth is the
# direction the camera looks along, so 0 puts it west of the shaft looking in
# through the open face.
LIFT_CAM = (0, 3.4, -6)
MAX_TICKS = 6000          # per leg

# A leg is done anywhere within GOAL_R of its last waypoint, which is half a
# metre of slop -- fine outside a room, too much in front of a lift, where it
# is the difference between stopping at the doors and stopping in them. Lift
# legs are held to a tighter radius, and the standoff has to beat it.
LIFT_GOAL_R = 0.25
assert lift.STANDOFF > LIFT_GOAL_R, \
    f"lift.STANDOFF ({lift.STANDOFF}) must exceed LIFT_GOAL_R ({LIFT_GOAL_R})"

# What to do when the LiDAR has found something and VAMOS has nothing that
# clears it. Not one rule but three, because "stop" and "carry on" are both
# wrong: crawl while asking again, and only stop once it is close and the
# answer has not changed. The spec wants a pedestrian avoided *without* full
# stops (L6 acceptance); a full stop is what is left when avoidance has failed.
CRAWL = 0.30              # fraction of top speed while looking for a way round
DETOUR = 0.70             # ...and while actually stepping around something
# There is deliberately no "clearance ahead" stop threshold. free_destination is
# the single authority on whether a way past exists -- it validates the line to
# the goal itself -- and a second test of the same line against a different
# number is how the dog ended up halting in gaps it was successfully using.
# What is left is the honest emergency: ROBOT_R, the distance at which the dog
# is touching something rather than approaching it.
BLOCKED_S = 8.0           # seconds stopped with no way past before giving up
LOOKAHEAD = 3.0           # metres of the route ahead tested for obstructions
BLOCKED_TICKS = 5         # consecutive ticks with no clear goal before believing it
PROBE_D = 6.0             # metres ahead the blockage is judged at, not the destination's 2-4

# People, who are not crates. A crate is gone round; a person is waited for --
# a guide dog that threads a moving gap is towing someone through it. The
# decision is tracking.conflict(), which asks whether the two paths meet
# rather than whether anyone is near; this is only what happens afterwards.
PATIENCE = 3.0            # seconds a person may stand still before the dog
                          # stops waiting for them and treats them as the
                          # obstacle they have become. Long enough to cover
                          # somebody breaking stride to let the dog past --
                          # that pause is about a second -- and short enough
                          # that a person who has genuinely stopped does not
                          # end the run.
PASS_NEED = 0.60          # going round someone who has stopped: room kept from
                          # what the LiDAR sees, instead of the crates' 0.35.
                          # About 0.8 m centre to centre -- the 0.9 m people
                          # give each other in passing. A crate's 0.35 took
                          # the dog past a standing person at 0.50 m, at full
                          # speed, 9 cm from their shoulder.
PASS_V = 0.40             # m/s while within PASS_R of them: walking past
PASS_R = 2.0              # somebody, not overtaking them
# No stepping back from them first. It was tried: reversing pushes the rigid
# handle into the person holding it, and nothing the dog senses knows they
# are there -- seed 8 of room 201 backed them into a wall for 9 s. A standoff
# with a pedestrian is broken on their side instead (pedestrians.GIVE_WAY).
YIELD_MAX = 30.0          # the backstop under all of it: still waiting after
                          # this long, creep and keep asking. Reachable only
                          # by a queue of people arriving one after another.
PED_NEAR = 0.55           # ground truth: nearer than this to a person counts
                          # as a contact. Scoring only, never shown the robot.

# The person being guided. They walk behind the dog holding a rigid handle
# (RDog's), so they are always straight behind it along its heading: the
# handle does not swivel. Not in the scene -- nothing the robot senses knows
# they are there -- and scored the way the dog's body is: touching a wall, a
# crate, the lift or the stairs counts. A route the dog clears can still walk
# them into something: on a pivot the handle swings them sideways through
# whatever is beside the dog.
HANDLER_BEHIND = 1.1      # metres, dog centre to theirs: the Go2's 0.41 m of
                          # body behind its centre, ~0.4 m of handle, an arm
HANDLER_R = 0.25          # metres, half a person's shoulder width
HANDLER_H = (0.4, 1.0)    # heights tested: crate height, and hip/hand height
HANDLER_RAYS = 12

# Easing a turn that would swing them into something. The dog knows the
# handle is rigid and how long it is, so it knows where the person will be
# after any turn -- and every handler contact measured was a turn: a pivot
# beside the crate it had just gone round (`electrical engineering lab`, the
# cart on `room 201` with people), a left turn past the lift (`cafeteria`),
# and a sharp left into the floor-2 restroom from the north lane after going
# round somebody, which put the person 1.1 m behind into the north wall. So
# each command is rolled forward SWING_T and, where it would take the person
# closer than HANDLER_ROOM to a wall or anything the LiDAR has found, the turn
# is eased and the dog keeps walking: a wider arc drags them round instead of
# swinging them sideways.
HANDLER_ROOM = 0.35       # metres, their centre to a surface: HANDLER_R + 0.10
SWING_T = 0.8             # seconds a command is rolled forward over
SWING_V = 0.30            # m/s walked while easing a turn a pivot would have made
SWING_EASE = (0.5, 0.25, 0.0)   # fractions of the turn rate tried, in order
SWING_NOSE = 0.30         # metres ahead of the dog's centre its front legs reach.
                          # An eased command walks on while the turn waits, so
                          # the front of the body has to stay clear too, not
                          # only the centre: seed 8 of room 201 --pedestrians 3
                          # kept 0.18 m at the centre and put its front legs on
                          # the floor-2 cart's corner.
NOSE_MIN = 0.15           # metres from the dog's front (SWING_NOSE ahead) to
                          # anything the LiDAR found, below which it turns
                          # before it walks on. The proximity stop reads the
                          # centre, and the Go2 is twice as long as it is wide:
                          # seed 8 again, centre 0.25 m clear of the cart, front
                          # 0.10 m, walking on at 0.6 m/s while turning.
NOSE_HALF_W = 0.15        # metres either side of the centreline, the front legs
SWING_PEOPLE = 0.70       # metres, dog centre to a person's predicted centre,
                          # kept by any step easing adds. PED_NEAR plus a margin.

# Doorways. The route goes through each one's middle, but the dog is not always
# on the route when it gets there -- stepped round a person, or on a VAMOS
# path, which cuts corners. From
# FUNNEL_D out it steers at a point FUNNEL_LEAD ahead of itself on the door's
# axis, so it closes on the line before the frame instead of arriving at the
# frame and pivoting to find it. That pivot was the floor-2 restroom: 0.5 m
# east of the axis at the threshold, turning to line up, with the crate beside
# the door on the person's side.
FUNNEL_D = 2.0            # metres before a doorway the funnel takes over
FUNNEL_LEAD = 0.8         # metres ahead on the axis it steers at
FUNNEL_PAST = 0.4         # metres beyond the doorway's far side it lets go
DOOR_MAP_D = 4.0          # metres from a doorway on the route inside which the
                          # map's route steers, not a VAMOS path. The route
                          # starts crossing the corridor for a door about this
                          # far out; a VAMOS path kept the dog in its lane past
                          # that point, left a sharp turn at the door, and the
                          # turn-easing for the person on the handle made it
                          # overshoot: chemistry lab, +14 s over map-only.
DOOR_NEAR = 1.0           # metres before a doorway where it lines up before walking:
DOOR_ERR = 0.25           # rad of heading error it will walk with there, against
                          # TURN_ONLY's 0.8 anywhere else. Walking while turning
                          # at the threshold is the body's corner meeting the frame.

# Backing out of a graze. The proximity stop below sets the speed to zero, and
# that used to be all it did -- which is a deadlock, because clearance cannot
# improve while the dog does not move, so every graze ran the full BLOCKED_S
# and ended the leg. Squeezing past the floor-2 cart leaves 0.18 m against a
# 0.16 m threshold, so which side of that a run lands on is decided by two
# centimetres: the empty building cleared it, the same route with people in it
# did not, and the people were nowhere near -- they had simply perturbed the
# dog's line. A two-centimetre difference should cost a step backwards, not a
# failed run.
BACK_V = 0.15             # m/s, backwards -- slow enough to feel deliberate on
                          # the handle rather than like a flinch
BACK_STEP = 0.45          # how far behind is checked before reversing into it


def resolve_stops(router, text, use_nlu=False):
    """Command -> [(destination name, floor or None)], one entry per stop.

    Splitting and floor phrases are rules either way (that is where they live
    in the pipeline); only the destination itself needs the model, and with
    --nlu off it is matched against the map's own names instead.
    """
    from cyberdog.language.command_splitter import split_destinations
    from cyberdog.language.floor_parser import extract_floor
    from cyberdog.language.generate_dataset import UNKNOWN_LOCATION

    parse = None
    if use_nlu:
        from cyberdog.language.infer import load_model, parse_command      # torch first, then the maps
        load_model()
        parse = parse_command

    stops, floor = [], START_FLOOR
    for text_i in split_destinations(text):
        requested, command = extract_floor(text_i, current_floor=floor)

        # Rules before the model: a no-go place named out loud is taken from
        # the raw text, because the parser was fine-tuned on destinations and
        # maps an unknown word onto the nearest one it knows -- "the stairs"
        # comes back as "hallway", and the stairs are never mentioned again.
        # plan_stops turns it into an approach point, not a refusal.
        named = router.hazard_named(text_i)
        if named is not None:
            stops.append((named[0], requested))
            if requested is not None:
                floor = requested
            continue

        if parse is not None:
            name = (parse(command).get("target_location") or "").strip().lower()
            if name == UNKNOWN_LOCATION:
                raise SystemExit(f"I don't know where that is: {text_i!r} names "
                                 f"no place in this building. Staying here.")
        else:
            # Longest name first, so the most specific match wins when one
            # location's name contains another's.
            name = next((k for k in sorted(router.locations, key=len, reverse=True)
                         if k in command.lower()), "")
        if not name:
            raise SystemExit(f"no destination found in {text_i!r} "
                             f"(try --nlu, or name a place from locations.json)")
        stops.append((name, requested))
        if requested is not None:
            floor = requested
    return stops


def lift_approach(pts, says):
    """A leg planned to the lift, cut back to the waiting point outside it.

    The router hands over at lift.LIFT_XY, which is inside the car, so the
    velocity controller used to drive the dog through the open face and park it
    in the shaft -- before the lift had been called, and with a scripted "walk
    in" afterwards that had nowhere left to walk. Drop the checkpoints in the
    doorway and end at the standoff; the handover happens there and lift_ride
    walks the last metre in. The final announcement ("take the lift") comes
    with it, because it is still the last thing said on this leg.
    """
    keep = [k for k, p in enumerate(pts) if not lift.in_doorway(p)]
    if keep and keep[-1] == len(pts) - 1:      # handed over outside already
        return pts, says
    return ([pts[k] for k in keep] + [lift.WAIT_XY],
            [says[k] for k in keep] + [says[-1]])


def lift_departure(pts, says):
    """The leg out of the lift, picked up where the dog actually stands.

    lift_ride steps back out to the same waiting point, so starting this leg at
    LIFT_XY meant the first waypoint was behind the dog and the first thing the
    controller did was turn round towards the shaft.
    """
    keep = [k for k, p in enumerate(pts) if not lift.in_doorway(p)]
    if keep and keep[0] == 0:                  # picked up outside already
        return pts, says
    return ([lift.WAIT_XY] + [pts[k] for k in keep],
            [says[0]] + [says[k] for k in keep])


def plan_stops(router, stops):
    """Stops -> [(floor, destination, [(x, y), ...], [[announcement], ...])]
    legs, following the planner's own floor handoffs.

    Each leg is one floor: a cross-floor stop comes back as a leg to the lift
    and a leg out of it, with the ride in between. The announcements come
    along one list per waypoint -- they are the whole point of the checkpoint
    on a robot whose user cannot see where it is about to turn.

    A stop inside a no-go zone becomes a leg to the corridor outside it, with
    the arrival announcement saying exactly that. Those legs also come back in
    {leg index: point to face on arrival}.
    """
    legs, faces = [], {}
    floor, xy = START_FLOOR, tuple(router.resolve(START_LOCATION, START_FLOOR, (0, 0))["xy"])
    start_xy = xy
    for name, requested in stops:
        if requested is not None and requested not in router.floors:
            raise SystemExit(f"there is no floor {requested}; the building has "
                             f"floors {router.floors[0]}-{router.floors[-1]}")
        if not router.floors_of(name):
            raise SystemExit(f"unknown destination: {name}")

        # Somewhere we will not go: walk to the corridor outside it instead,
        # and say so. Asking for the stairs is a fair thing to ask -- it is
        # going up them that is not on offer.
        hazard = router.hazard_place(name)
        if hazard is not None:
            approached = router.approach(name, floor, xy, floor=requested)
            if approached is None:
                raise SystemExit(f"there is no {name} on floor {requested}")
            target, line = approached
        else:
            target = router.resolve(name, floor, xy, floor=requested)
        if target is None:
            raise SystemExit(f"there is no {name} on floor {requested} (it is on "
                             f"floor(s) {', '.join(map(str, router.floors_of(name)))})")
        planned = router.plan(floor, xy, target, lift_stop=lift.WAIT_XY)
        if planned is None:
            raise SystemExit(f"no route to {name}")
        for j, (f, route) in enumerate(planned):
            pts, says = router.walk_line(f, route)
            # A cross-floor stop comes back as a leg to the lift and a leg out
            # of it; neither should be driven into the car itself.
            if len(planned) > 1 and j == 0:
                pts, says = lift_approach(pts, says)
            elif len(planned) > 1 and j == 1:
                pts, says = lift_departure(pts, says)
            legs.append((f, name, pts, says))
        if hazard is not None:
            # Replaces "destination reached" on the last leg of this stop:
            # the dog has not reached what was asked for, and saying so is the
            # announcement. Anything else would be a small lie to someone who
            # cannot check.
            legs[-1][3][-1] = [line]
            # And turn to face the thing at the end of it. The person asked to
            # be taken to the stairs; standing beside them facing down the
            # corridor answers half the question.
            faces[len(legs) - 1] = tuple(router.resolve(
                name, floor, xy, floor=requested, allow_hazard=True)["xy"])
        floor, xy = target["floor"], tuple(target["xy"])
    return start_xy, legs, faces


class HazardStop(Exception):
    """The dog entered a zone it must not move in. Nothing resumes after this.

    The planner will not route through a stairwell, so this firing means
    something upstream is wrong -- which is exactly when a guide robot should
    stop moving and say so rather than carry on.
    """


class Run:
    """One drive through the building, with the video panels attached."""

    def __init__(self, scene, start_xy, start_yaw, router, out=None, vamos=False,
                 auto_confirm=False, speed=1, crowd=0, seed=0, shadow=None,
                 vamos_url=None, vlm_latency=None, vamos_sample=False):
        self.robot = MujocoRobot(scene, start_xy=start_xy, start_yaw=start_yaw,
                                 start_z=levels.floor_z(START_FLOOR))
        self.cam = load_camera_config()
        self.router = router
        self._doors = {}          # floor -> [(name, door_xy)], for the captions
        self.floor = START_FLOOR
        self.in_lift = False
        self.behind = False
        self.auto_confirm = auto_confirm
        # Write one frame in `speed`. The building is 48 m end to end and the
        # stairwell and the lift are at opposite ends of it, so a run that
        # visits both spends a minute walking down a corridor that looks the
        # same the whole way. Dropping frames plays that back faster without
        # touching the control loop, which still runs at 20 Hz.
        self.speed = max(int(speed), 1)
        self.tick = 0
        self.writer = self.chase = self.outside = None
        self.policies, self.vamos = {}, vamos
        # Shadow mode (Gate C in docs/dreaming_safety_kpis.md): VAMOS is asked
        # and the dream scores its answer on the same schedule as --vamos, but
        # the map route alone drives. `shadow` is the log file, one JSON line
        # per call; None is off. --vamos and --shadow are exclusive.
        if shadow:
            os.makedirs(os.path.dirname(os.path.abspath(shadow)), exist_ok=True)
        self.shadow = open(shadow, "w") if shadow else None
        self.shadow_n = {"calls": 0, "steer": 0, "none": 0}
        self.vamos_url = vamos_url
        self.vamos_sample = vamos_sample
        # How long a VAMOS answer takes to arrive, in seconds of sim time; None
        # is the call's own wall-clock time. See ask_vamos.
        self.vlm_latency = vlm_latency
        # The request in flight, if any: (tick it lands on, its paths, tick it
        # was asked on, body xy then). One at a time, as a GPU serves them.
        self.pending = None
        self.vlm = {"asked": 0, "answers": 0, "dropped": 0, "age": 0.0, "moved": 0.0}
        # Every velocity command sent, fingerprinted. Two runs of the same
        # route and seed with equal fingerprints commanded the dog identically,
        # tick for tick -- which is how a shadow run proves it touched nothing.
        self.commands = hashlib.sha256()
        self.n_commands = 0
        self.lidar = Lidar(self.robot.model, self.robot.data)
        self.live = {}            # floor -> LiveClearance, one per storey
        self.seen = np.empty((0, 2))   # this tick's unexplained returns
        # Ground truth, for scoring only -- never shown to the robot. The dog
        # finds these with the LiDAR or not at all; this is how we check.
        self.hits = {"ticks": 0, "boxes": set()}
        self.handler_hits = {"ticks": 0, "boxes": set()}
        self.dog_geoms, self.touchable = self.body_contacts()
        self.obs = {"crawl": 0, "stopped": 0, "seen": 0, "min_clear": 99.0, "eased": 0}
        # People. Not in any grid either, and unlike the crates they move, so
        # one scan cannot describe them -- `tracker` is what two scans give.
        self.crowds = {n: pedestrians.Crowd(n, crowd, seed) for n in (1, 2, 3)}
        if crowd and not self.robot.has_body("ped_f1_0"):
            # A mocap body cannot be added to a loaded model, so people only
            # exist if the scene was built with them. Silently walking an
            # empty building instead is the worst of both: the run looks fine
            # and proves nothing.
            raise SystemExit("this scene has no pedestrian bodies -- rebuild it: "
                             "python -m cyberdog.sim.scene.build_scene --building")
        self.tracker = Tracker(self.robot.CONTROL_DT)
        self.waiting_for = None        # the track being yielded to, if any
        self.passing = None            # where the person being gone round stands
        self.waited = 0                # ticks spent yielding to it
        self.held_still = 0            # ...of which it has not moved at all
        self.movers = np.empty((0, 2))  # this tick's moving things, for the video
        self.ped = {"yields": 0, "waited": 0, "near": 0, "min_d": 99.0}
        self.place_crowd()

        if out is not None:
            import imageio
            import mujoco
            self.writer = imageio.get_writer(out, fps=FPS, macro_block_size=1)
            self.chase = mujoco.MjvCamera()
            self.chase.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.chase.distance, self.chase.elevation = CHASE_D, CHASE_EL
            self.outside = mujoco.Renderer(self.robot.model, 480, 640)

    # -- video ----------------------------------------------------------
    def frame(self, state=None, chosen=None, candidates=(), safety=None):
        if self.writer is None:
            return
        self.tick += 1
        if self.tick % self.speed:
            return

        dog = self.robot.get_image().copy()
        pose = self.robot.camera_pose()
        # Under the paths: what it found that the map had no record of. A dog
        # swerving for nothing visible reads as a bug, not as avoidance.
        if len(self.seen):
            draw_points(dog, self.seen, pose, self.cam)
        # And which of them are walking, in a colour of their own. A dog that
        # stops dead in an empty-looking corridor reads as a fault; these are
        # the reason, and they are the only thing on screen that tells the
        # difference between the crate it went round and the person it did not.
        if len(self.movers):
            draw_points(dog, self.movers, pose, self.cam, colour=(70, 160, 255), r=3)
        for c in candidates:
            draw_path(dog, c, pose, self.cam, (120, 120, 130))
        if chosen:
            draw_path(dog, chosen, pose, self.cam, (90, 220, 120), thick=2)
        if state and state["state"] == "TRACK":
            draw_marker(dog, *state["pixel"])
        if safety is not None:
            safety_bar(dog, safety)

        x, y, yaw = self.robot.get_pose()
        self.chase.lookat[:] = [x, y, self.robot.z + 0.3]
        if self.in_lift:
            # Chasing from behind puts the camera inside the shaft wall. The
            # car is only open to the west, so watch from out in the corridor
            # and let it rise past the lens.
            self.chase.azimuth, self.chase.distance, self.chase.elevation = LIFT_CAM
        else:
            # Azimuth is the direction the camera looks along, so the usual
            # +180 sits the camera ahead of the dog, walking towards the lens.
            # For the closing hold, drop the 180 and watch from behind: the
            # dog ends a run facing whatever it was sent to -- a wall, or a
            # stairwell -- and a camera in front of it is inside that.
            self.chase.azimuth, self.chase.distance, self.chase.elevation = \
                math.degrees(yaw) + (0 if self.behind else 180), CHASE_D, CHASE_EL
        self.outside.update_scene(self.robot.data, self.chase)
        left = np.ascontiguousarray(self.outside.render())
        self.floor_tag(left)
        self.name_tag(left, x, y)
        self.writer.append_data(np.hstack([left, dog]))

    def doors(self, floor):
        """(name, sign_xy) for every named place on one floor, cached.

        Doors rather than room centres: the door is what the dog walks past,
        and a room centre is behind a wall. `hallway` is skipped -- it is the
        corridor itself, so its label would sit on top of every other one.

        Two collisions have to be undone first, and both come from
        `locations.json` being a routing table rather than a signboard:

        `door_xy` is the corridor centreline, so the room on the north side
        and the room on the south share one to the centimetre -- `room 101`
        and `room 106` are both [3.0, 9.5]. Signed as-is, one of every pair is
        dropped as an overlap and half the building is unlabelled. So the sign
        slides SIGN_OFF off the centreline towards its own room, which puts it
        on that room's wall, which is where signage lives anyway.

        And a place can have several names for the parser's benefit -- stairs,
        staircase, stairway and stairwell are one stairwell at [1.0, 9.5].
        A door has one sign, so the shortest name wins and the synonyms go.
        """
        if floor not in self._doors:
            seen, out = {}, []
            for name, entries in self.router.locations.items():
                if name == "hallway":
                    continue
                for e in entries:
                    if e["floor"] != floor:
                        continue
                    door = tuple(e.get("door_xy") or e["xy"])
                    room = tuple(e["xy"])
                    key = (round(door[0], 2), round(door[1], 2),
                           round(room[0], 2), round(room[1], 2))
                    if key in seen and len(seen[key]) <= len(name):
                        continue        # a synonym of something already signed
                    seen[key] = name
            for (dx, dy, rx, ry), name in seen.items():
                vx, vy = rx - dx, ry - dy
                n = math.hypot(vx, vy)
                if n:
                    dx, dy = dx + vx / n * SIGN_OFF, dy + vy / n * SIGN_OFF
                out.append((name, (dx, dy)))
            self._doors[floor] = out
        return self._doors[floor]

    def name_tag(self, img, x, y):
        """Room names on the chase panel, for the rooms the dog is passing.

        The same complaint `floor_tag` answers, one level finer: the pips say
        which storey, and nothing said where along it. Every doorway in this
        building renders identically, so a run to `room 201` and a run to
        `room 206` are the same video.

        Caption only. These are drawn onto the rendered chase frame, not built
        into the scene, so the dog's camera -- the one VAMOS is handed -- has
        no text in it. That boundary is the same one `obstacles.py` and
        `pedestrians.py` keep: the robot is not told anything here.
        """
        cam = ChaseCam(self.outside.scene, img.shape[1], img.shape[0])
        label_places(img, cam, self.doors(self.floor), (x, y),
                     levels.floor_z(self.floor))

    def floor_tag(self, img, n=3):
        """Three stacked pips down the left edge, the current floor lit.

        The chase view of a corridor looks the same on every storey, so
        without this you cannot tell from the video which floor you are on.
        """
        for k in range(n):
            top = 14 + (n - 1 - k) * 22          # floor 1 at the bottom
            lit = (k + 1) == self.floor
            img[top:top + 16, 14:30] = (250, 205, 90) if lit else (70, 72, 78)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    # -- driving --------------------------------------------------------
    def clearance(self, floor):
        """The floor's clearance field: the static map, plus whatever the LiDAR
        found that the map cannot account for.

        One object per storey, updated in place. The cached VamosPolicy's gate
        and its Dream both close over it, so a scan reaches both of them
        without rebuilding either.
        """
        if floor not in self.live:
            self.live[floor] = LiveClearance(floor)
        return self.live[floor]

    def policy(self, floor):
        """VAMOS policy for a floor -- its gate and its dreams read that floor.

        Both come off the same clearance field, so the cheap geometric test
        and the imagined rollouts cannot disagree about where the walls, the
        stairwells, or the crate that is not on any map are.
        """
        if floor not in self.policies:
            from cyberdog.sim.sensing.dreaming import Dream
            from cyberdog.sim.vamos_client import VamosPolicy
            from cyberdog.sim.vamos_client import DEFAULT_URL
            live = self.clearance(floor)
            url = self.vamos_url or DEFAULT_URL
            p = VamosPolicy(self.cam, lambda x, y: live(x, y) >= ROBOT_R,
                            dream=Dream(live, self.robot.MAX_V,
                                        person=lambda x, y, yaw: self.handler_room(
                                            live, x, y, yaw)), url=url,
                            sample=self.vamos_sample)
            if not p.available():
                raise SystemExit(f"VAMOS server is not answering on {url} -- "
                                 "start vendor/VAMOS/server/vlm_server.py first")
            self.policies[floor] = p
        return self.policies[floor]

    def log_shadow(self, floor, n, pose, target, proposed, safety):
        """One shadow-mode call: what VAMOS offered, what the dream made of
        each candidate, and what it would have steered at against what the map
        actually did.

        "steer" is a proposal pointing more than SHADOW_DISAGREE off the map's
        target; "none" is nothing passing the gate, which under --vamos means
        crawling and asking again. Both are what Gate D is about to measure.
        """
        x, y, yaw = pose
        verdicts = self.policies[floor].verdicts
        if proposed is None:
            kind, angle = "none", None
        else:
            px, py = path_target(proposed, (x, y))
            angle = abs(wrap(math.atan2(py - y, px - x)
                             - math.atan2(target[1] - y, target[0] - x)))
            kind = "steer" if angle > SHADOW_DISAGREE else "agree"
        self.shadow_n["calls"] += 1
        if kind != "agree":
            self.shadow_n[kind] += 1
        rnd = lambda pts: [[round(a, 3), round(b, 3)] for a, b in pts]
        self.shadow.write(json.dumps({
            "floor": floor, "tick": n, "t": round(self.robot.sim_time, 2),
            "pose": [round(x, 3), round(y, 3), round(yaw, 4)],
            "map_target": [round(target[0], 3), round(target[1], 3)],
            "candidates": [{"path": rnd(p), "factor": None if f is None else round(f, 4),
                            "verdict": v} for p, f, v in verdicts],
            "would_choose": next((k for k, (p, _, _) in enumerate(verdicts)
                                  if p is proposed), None),
            "safety": round(safety, 4),
            "disagree": kind,
            "angle_deg": None if angle is None else round(math.degrees(angle), 1),
        }) + "\n")

    def ask_vamos(self, floor, n, state):
        """Send VAMOS this tick's frame; the answer lands some ticks later.

        The VLM takes about 1.8 s and the control loop has 50 ms, so the loop
        cannot wait for it: plan() used to be called here and block, and the
        twin paused physics for the duration, which hid that a real dog would
        walk on for 1.8 s on its last command with no LiDAR, no stop and no
        yield. Now the loop never waits. The request is made at tick n, from
        this tick's image, and its answer is released at tick n + latency --
        in sim time, so a run with --vlm-latency fixed is repeatable where a
        real thread would land its answer on a different tick every time. On
        the robot this is a worker thread; the timing is the same.

        The wall-clock time the HTTP call really took is the default latency:
        the sim is paused while it runs, then charged for it afterwards.
        """
        t0 = time.perf_counter()
        cam_pose = self.robot.camera_pose()
        paths = self.policy(floor).request(self.robot.get_image(),
                                           vamos_prompt(state), cam_pose)
        wait = (time.perf_counter() - t0 if self.vlm_latency is None
                else self.vlm_latency)
        self.pending = (n + int(round(wait / self.robot.CONTROL_DT)), paths,
                        n, self.robot.get_pose()[:2])
        self.vlm["asked"] += 1

    def take_answer(self, floor, n, pose, destination):
        """The pending answer, if it has landed: judged from where the dog is
        now, not from where it was asked. Returns plan()'s triple, or None
        while the answer is still on its way.
        """
        if self.pending is None or n < self.pending[0]:
            return None
        _, paths, asked_at, xy0 = self.pending
        self.pending = None
        self.vlm["answers"] += 1
        self.vlm["age"] += (n - asked_at) * self.robot.CONTROL_DT
        self.vlm["moved"] += math.dist(xy0, pose[:2])
        return self.policy(floor).deliver(paths, pose, destination)

    def say(self, line):
        """One line of the Preemptive Voice Engine. Printed here, spoken later."""
        print(f"    [voice] {line}")

    @staticmethod
    def clear_ahead(live, xy, target, d=LOOKAHEAD, skip=perception.SKIP):
        """Least clearance along the next `d` metres towards `target`.

        Clearance under the dog is not the question -- by the time it is low
        the dog is already in the thing. This is what is coming, and only from
        things the map had no record of.

        The first `skip` metres are left out for the same reason line_clear
        leaves them out: a dog squeezing through a gap beside a crate is
        legitimately 0.16 m from it, and a test that starts at its feet reads
        that as "blocked" and halts it in the middle of the gap it is using.
        """
        dx, dy = target[0] - xy[0], target[1] - xy[1]
        span = math.hypot(dx, dy)
        if span < 1e-6:
            return live.detected_at(*xy)
        d = min(d, span)
        steps = max(int(d / 0.1), 1)
        first = int(skip / 0.1) if d > skip else 0
        # Detected things only. The static field is tight in every doorway on
        # the map and A* already approved those; stopping for one would mean
        # never leaving a room.
        return min(live.detected_at(xy[0] + dx * (d * k / steps) / span,
                                    xy[1] + dy * (d * k / steps) / span)
                   for k in range(first, steps + 1))

    def body_contacts(self):
        """(the dog's geoms, {scene geom: what it is}) for collision scoring.

        The dog's geoms are everything under the body with the free joint:
        trunk, hips, legs, feet. The scene's are everything the body must not
        touch -- walls, crates, the lift shaft and roof, stair steps. Left out:
        what it stands on (`ground`, the `slab`s, the lift car's `lift_plate`),
        where contact is walking, and the people, whose geoms do not collide at
        all and are scored by distance instead (PED_NEAR).
        """
        import mujoco
        m = self.robot.model
        root = next(m.jnt_bodyid[j] for j in range(m.njnt)
                    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)

        def ours(b):
            while b:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False

        dog, scene = set(), {}
        for g in range(m.ngeom):
            if ours(m.geom_bodyid[g]):
                dog.add(g)
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if (name == "ground" or name.startswith("slab") or name == "lift_plate"
                    or not m.geom_contype[g] and not m.geom_conaffinity[g]):
                continue
            scene[g] = (name.split("_", 1)[1] if name.startswith("obs")
                        else "stairs" if name.startswith("step")
                        else "lift" if name.startswith("lift")
                        else "wall")
        return dog, scene

    def score_collision(self, floor, x, y):
        """Ground truth: is any part of the dog touching a wall or an obstacle?

        MuJoCo's own contacts between the Go2's geoms and the scene's, so the
        whole body counts -- trunk, hips, swinging legs -- not just the point
        at its centre. The point test this replaced scored a trunk scraping a
        crate as clean as long as the centre stayed outside the box, and never
        looked at walls at all.

        Scoring only, and never fed back into the robot -- the dog finds these
        with the LiDAR or it does not find them. A run that reports arrival
        while this counter climbed is a run that walked into something.
        """
        d = self.robot.data
        touched = set()
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 in self.dog_geoms else g1 if g2 in self.dog_geoms else None
            if other in self.touchable:
                touched.add(self.touchable[other])
        if touched:
            self.hits["ticks"] += 1
            self.hits["boxes"].update(f"floor {floor} {what}" for what in touched)

    def handler_at(self):
        """Where the person on the handle is: straight behind, along the heading."""
        x, y, yaw = self.robot.get_pose()
        return x - HANDLER_BEHIND * math.cos(yaw), y - HANDLER_BEHIND * math.sin(yaw)

    @staticmethod
    def door_near(gates, xy, reach):
        """Is a doorway on the route, not yet walked through, within `reach`?"""
        for entry, exit_, (ux, uy) in gates:
            depth = (exit_[0] - entry[0]) * ux + (exit_[1] - entry[1]) * uy
            if (xy[0] - entry[0]) * ux + (xy[1] - entry[1]) * uy > depth + FUNNEL_PAST:
                continue                          # through this one already
            return math.dist(xy, entry) < reach
        return False

    @staticmethod
    def funnel(gates, xy, target):
        """(`target`, or a point on the axis of the doorway just ahead; whether
        the dog is within DOOR_NEAR of that doorway). See FUNNEL_D."""
        for entry, exit_, (ux, uy) in gates:
            s = (xy[0] - entry[0]) * ux + (xy[1] - entry[1]) * uy
            depth = (exit_[0] - entry[0]) * ux + (exit_[1] - entry[1]) * uy
            if s > depth + FUNNEL_PAST:
                continue                          # through this one already
            if s < -FUNNEL_D:
                break                             # not there yet
            # Not a door the dog is walking past on its way somewhere else:
            # only when it is within a funnel's width of the axis.
            lateral = (xy[0] - entry[0]) * -uy + (xy[1] - entry[1]) * ux
            if abs(lateral) > FUNNEL_D:
                break
            ahead = min(s + FUNNEL_LEAD, depth + FUNNEL_PAST)
            return (entry[0] + ux * ahead, entry[1] + uy * ahead), -DOOR_NEAR < s < 0.0
        return target, False

    def handler_room(self, live, x, y, yaw):
        """What the dog can know of the room around the person, for a dog pose:
        walls from the map, measured to their real faces rather than the
        planner's inflated ones (clearance.wall_face_field -- the person's
        shoulder lives in the inflation band), and crates from the LiDAR's
        standing field."""
        px, py = x - HANDLER_BEHIND * math.cos(yaw), y - HANDLER_BEHIND * math.sin(yaw)
        return min(live.wall_at(px, py), live.detected_at(px, py))

    @staticmethod
    def nose_closing(live, pose, cmd, dt=0.1):
        """Is a front corner of the dog within NOSE_MIN of something, and would
        this command bring that corner closer still? Walking away is fine.

        Corners, not the middle of the front: turning away from a crate swings
        the middle clear while the corner on the crate's side is still closing
        -- seed 8's front-left leg, on the cart, with the dog turning right."""
        def corners(x, y, yaw):
            fx, fy = x + SWING_NOSE * math.cos(yaw), y + SWING_NOSE * math.sin(yaw)
            return [live.detected_at(fx + side * -math.sin(yaw), fy + side * math.cos(yaw))
                    for side in (-NOSE_HALF_W, 0.0, NOSE_HALF_W)]
        x, y, yaw = pose
        now = corners(x, y, yaw)
        if min(now) >= NOSE_MIN:
            return False
        yaw += cmd[2] * dt
        x += cmd[0] * math.cos(yaw) * dt
        y += cmd[0] * math.sin(yaw) * dt
        return any(a < NOSE_MIN and b < a for a, b in zip(now, corners(x, y, yaw)))

    def swing(self, live, pose, cmd, people=(), dt=0.1):
        """Roll `cmd` forward SWING_T: (least room around the person, whether
        the dog itself stays clear -- of everything by ROBOT_R, and of where
        each of `people` is heading by SWING_PEOPLE)."""
        x, y, yaw = pose
        vx, _, w = cmd
        worst, clear = INF, True
        for k in range(1, int(round(SWING_T / dt)) + 1):
            yaw += w * dt
            x += vx * math.cos(yaw) * dt
            y += vx * math.sin(yaw) * dt
            worst = min(worst, self.handler_room(live, x, y, yaw))
            clear = (clear and live(x, y) >= ROBOT_R
                     and live.detected_at(x + SWING_NOSE * math.cos(yaw),
                                          y + SWING_NOSE * math.sin(yaw)) >= ROBOT_R
                     and all(math.dist((x, y), p.predict(k * dt)) >= SWING_PEOPLE
                             for p in people))
        return worst, clear

    def spare_handler(self, live, pose, cmd, limit, people=()):
        """`cmd`, with its turn eased where it would swing the person into
        something. See HANDLER_ROOM. `limit` is the speed every other rule
        has already allowed; easing never walks faster than it.

        `people` are the person-sized tracks. Easing walks on where the
        command would have pivoted, and on seeds 4 and 12 of `room 201
        --pedestrians 3` that walked the dog to 0.42 m of somebody standing
        beside it. So an eased command that walks further than the original
        must keep SWING_PEOPLE from where each of them is heading, or the
        turn goes ahead as commanded. Only then: holding every eased command
        to it, whether or not it added a step, turned the dog's easing off
        wherever anyone was near, and put the person on the handle back
        against walls on 5 seeds of 12."""
        now = self.handler_room(live, *pose)
        worst, _ = self.swing(live, pose, cmd)
        if worst >= HANDLER_ROOM or worst >= now:
            return cmd
        vx, vy, w = cmd
        v = max(vx, min(SWING_V, limit))
        best = None
        for f in SWING_EASE:
            alt = (v, vy, w * f)
            room, clear = self.swing(live, pose, alt, people if v > vx else ())
            if not clear:
                continue
            if room >= HANDLER_ROOM or room >= now:
                best = (room, alt)
                break
            if best is None or room > best[0]:
                best = (room, alt)
        if best is None or best[0] <= worst:
            return cmd
        self.obs["eased"] += 1
        return best[1]

    def score_handler(self, floor):
        """Ground truth for the person: is their body touching anything?

        They are not in the MuJoCo scene, so this asks the scene with rays: a
        ring of HANDLER_RAYS at each of HANDLER_H, and a hit within HANDLER_R
        of their centre is a touch. A ray starting inside a solid box sees
        nothing of it, so one more straight down from above the crates: landing
        on something before the floor means they are standing in it. Scoring
        only, like score_collision.
        """
        import mujoco
        m, d = self.robot.model, self.robot.data
        px, py = self.handler_at()
        z0 = levels.floor_z(floor)
        gid = np.zeros(1, np.int32)
        touched = set()

        def ray(p, v, reach):
            dist = mujoco.mj_ray(m, d, np.array(p, float), np.array(v, float),
                                 None, 1, -1, gid)
            if 0 <= dist <= reach and int(gid[0]) in self.touchable:
                touched.add(self.touchable[int(gid[0])])

        for h in HANDLER_H:
            for k in range(HANDLER_RAYS):
                a = 2 * math.pi * k / HANDLER_RAYS
                ray((px, py, z0 + h), (math.cos(a), math.sin(a), 0.0), HANDLER_R)
        ray((px, py, z0 + 1.5), (0.0, 0.0, -1.0), 1.45)
        if touched:
            self.handler_hits["ticks"] += 1
            self.handler_hits["boxes"].update(f"floor {floor} {what}" for what in touched)

    # -- the crowd ------------------------------------------------------
    def place_crowd(self):
        """Put every pedestrian body where its walker is, this tick.

        Every floor's, not just this one's: the dog can see up a stairwell and
        out of the lift, and a person who only exists while the dog is on
        their storey pops into being mid-corridor.
        """
        poses = [p for crowd in self.crowds.values() for p in crowd.poses()]
        self.robot.move_mocaps(poses)

    def step_crowd(self):
        """One control tick of walking, for everyone.

        Called once per tick from every loop that also steps the robot --
        follow, and the scripted lift moves. A crowd that freezes while the
        dog rides the lift would have the whole building hold still for it.
        """
        dog = self.robot.get_pose()[:2]
        for floor, crowd in self.crowds.items():
            crowd.step(self.robot.CONTROL_DT, dog if floor == self.floor else None)
        self.place_crowd()

    def score_people(self, floor, x, y):
        """Ground truth again: how near the dog actually got to a person.

        Same rule as score_collision -- never fed back into the robot. The
        dog stops for people because it tracked them, or it does not stop.
        """
        for px, py in self.crowds[floor].positions():
            d = math.hypot(px - x, py - y)
            self.ped["min_d"] = min(self.ped["min_d"], d)
            if d < PED_NEAR:
                self.ped["near"] += 1

    def yield_to(self, tracks, xy, yaw, speed):
        """The person to wait for, or None. Holds the decision between ticks.

        Two thresholds, not one. Starting to wait asks MISS_R; carrying on
        waiting asks CLEAR_R, which is wider. With a single number the dog
        moves off the instant it is a millimetre clear, immediately conflicts
        again, and shuffles its way past somebody in a series of twitches.
        The gap between the two is what makes it one stop and one start.

        And once it is waiting for somebody it keeps waiting for *them*, not
        for whoever happens to look threatening this tick. That fixes a nasty
        little deadlock: a person who has stopped -- because the dog is in
        front of them, which is why they stopped -- has no velocity, so the
        moving test calls them a crate, so the dog drives at them, so they
        stay stopped. Measured without the latch: 43 stop-starts in one run
        and an ending nose-first against the floor-2 cart.

        PATIENCE is the other end of it, and it is not a tidy-up. Waiting is
        only the right answer while they are actually going somewhere. Someone
        who has stopped and stayed stopped -- reading a noticeboard, holding a
        door -- is furniture now, and furniture is `free_destination`'s job: go
        round it. Without the release the dog waits out the whole run for a
        person who has no intention of moving, which it did, for 267 seconds,
        four metres from a lift it never reached.
        """
        vel = (speed * math.cos(yaw), speed * math.sin(yaw))

        held = self.waiting_for if self.waiting_for in tracks else None
        if held is not None:
            d = closest_approach(xy, vel, (held.x, held.y), (held.vx, held.vy))
            self.held_still = 0 if held.moving else self.held_still + 1
            if d - min(held.r, 0.4) < CLEAR_R and self.held_still < PATIENCE * FPS:
                self.waited += 1
                self.ped["waited"] += 1
                return held
            if self.held_still >= PATIENCE * FPS:
                self.say("They have stopped. I will go around them.")
                self.passing = (held.x, held.y)
                self.waiting_for, self.waited, self.held_still = None, 0, 0
                return None

        person = conflict(xy, vel, tracks, radius=MISS_R)

        if person is None:
            if self.waiting_for is not None:
                self.say("Thank you. Carrying on.")
            self.waiting_for, self.waited, self.held_still = None, 0, 0
            return None

        if self.waiting_for is None:
            self.ped["yields"] += 1
            if os.environ.get("CYBERDOG_TRACE"):
                # Every yield in a building with nobody in it is a crate being
                # mistaken for a person. This is how to see which crate.
                print(f"    [trace] yield to ({person.x:.1f},{person.y:.1f}) "
                      f"v=({person.vx:.2f},{person.vy:.2f}) |v|={person.speed:.2f} "
                      f"r={person.r:.2f} from ({xy[0]:.1f},{xy[1]:.1f})")
            # Which way they are going, because the person on the handle can
            # hear that and cannot see it.
            side = "right" if (person.vx * -math.sin(yaw)
                               + person.vy * math.cos(yaw)) < 0 else "left"
            self.say(f"Someone crossing from the {side}. Waiting.")
        self.waiting_for = person
        self.waited += 1
        self.ped["waited"] += 1
        return person

    def follow(self, waypoints, floor, announcements=None, verbose=True,
               goal_r=GOAL_R):
        """Walk one floor's leg. Same controller as run_demo, plus the frames.

        `goal_r` is how near the last waypoint counts as arrived -- tightened
        for the legs that end at the lift, where the remaining half metre is
        walked scripted and has to start outside the car.
        """
        self.floor = floor
        self.robot.set_height(levels.floor_z(floor))
        # An answer still on its way from the last leg is about a corridor the
        # dog has left -- another floor, or the far side of a lift ride.
        if self.pending is not None:
            self.pending = None
            self.vlm["dropped"] += 1
        self.passing = None
        if not waypoints:
            # Asked for where it already is. A* has no checkpoints to give for
            # a route of zero length, and every line below indexes into them --
            # `waypoints[-1]` on an empty list ended the run in an IndexError
            # rather than an answer, for "take me to the main entrance" said at
            # the main entrance.
            self.say("We are already there.")
            return True, 0
        i = 1 if len(waypoints) > 1 else 0
        said = set()
        gates = self.router.gates(floor, waypoints)
        # `proposed` is what VAMOS last offered and the gate let through;
        # `chosen` is what steers -- the same thing under --vamos, never
        # anything under --shadow.
        proposed, chosen, candidates, safety = None, None, [], 1.0
        hunting = halted = False
        stalled = no_goal = 0
        lean = 0.0                # the side of a detour once it is committed

        for n in range(MAX_TICKS):
            x, y, yaw = self.robot.get_pose()

            # Before anything else: is this a place the dog must not be? The
            # planner routes around the stairwells, so reaching one means a
            # layer below it went wrong, and the safe answer is to stop dead.
            hazard = self.router.hazard_at((x, y))
            if hazard is not None:
                self.robot.set_velocity(0.0, 0.0, 0.0)
                self.say(hazard)
                raise HazardStop(f"stopped on floor {floor} at ({x:.1f}, {y:.1f}): {hazard}")

            i = advance(i, x, y, waypoints)
            # Preemptive, which is the whole point: the line comes while the
            # turn is still ahead, not as the dog is already in it. The last
            # checkpoint is the exception -- "destination reached" is said on
            # arrival, below, not a couple of metres out.
            if announcements and i not in said and i < len(waypoints) - 1 \
                    and math.hypot(waypoints[i][0] - x, waypoints[i][1] - y) < ANNOUNCE_R:
                said.add(i)
                for line in announcements[i]:
                    self.say(line)
            # What the sensor found, folded into the field the gate and the
            # imagined rollouts read. Everything below that reacts to a crate
            # on no map reacts because of this one call.
            self.step_crowd()
            live = self.clearance(floor)
            live.update(self.lidar.scan((x, y), self.robot.z), self.robot.z,
                        (x, y), origin=(x, y, self.robot.z + MOUNT_H))
            self.seen = live.points
            self.obs["seen"] = max(self.obs["seen"], len(live.points))
            # Same returns, asked a different question: which of them moved
            # since last tick. A crate answers "none of me".
            tracks = self.tracker.update(live.points)
            walking = self.tracker.movers()
            # Out of the planning field, still in the safety one. A person is
            # waited for; only a crate is gone round. See LiveClearance.exclude.
            live.exclude([(t.x, t.y, t.r) for t in walking])
            self.movers = np.array([(t.x, t.y) for t in walking]).reshape(-1, 2)
            self.score_collision(floor, x, y)
            self.score_handler(floor)
            self.score_people(floor, x, y)

            # Picked from the body, projected from the lens -- see
            # project_route. The probe below already measures from the body.
            state = project_route(self.robot.camera_pose(), waypoints[i:], self.cam,
                                  pick_from=(x, y))

            # "Map decides WHERE": the same destination, moved sideways when it or
            # the line to it is blocked by something the map never had. VAMOS
            # drives at whatever goal pixel it is given, so a destination inside a
            # crate is five candidate paths into the crate -- measured, before
            # this existed. Move the goal and the model has something to solve.
            aim, offset, blocked = None, 0.0, False
            no_way = False            # the map has a route destination and no clear aim
            # Whenever the route has a destination -- not only when the camera
            # can see it. This used to read `if state["state"] == "TRACK"`, and
            # that is a camera test standing in for a safety one: ALIGN means
            # the goal pixel is outside the image, which happens exactly when
            # the dog is turning into a doorway, and it says nothing whatever
            # about what is in front of it. The LiDAR is a 360-degree sensor.
            # With avoidance gated on the projection, the dog steered at the
            # raw A* waypoint for the whole turn and drove straight through
            # anything the map did not know about: `chemistry lab` spent 0.4 s
            # inside the floor-3 cartons, every tick of it in ALIGN, on a route
            # whose own clearance figures looked fine either side of the turn.
            tracking = state["state"] == "TRACK"
            if state.get("destination") is not None:
                # Judge the blockage further out than the destination -- see
                # free_destination. PROBE_D is far enough to start moving across
                # while there is still open corridor to do it in.
                #
                # But not around a corner. Arc length along the route runs on
                # past turns, and 6 m along a leg that turns into a doorway is
                # a point the straight line reaches only by crossing a wall --
                # so the tightest spot on that line is the wall, and the dog
                # concludes the building is in its way. Take the furthest probe
                # still in open floor on a straight line, and settle for a
                # nearer one at a turn.
                probe = None
                for reach in (PROBE_D, 4.5, 3.0):
                    q = pick_destination((x, y), waypoints[i:], reach, reach + 2.0)
                    if q is not None and line_clear((x, y), q, live.static_at, ROBOT_R):
                        probe = q
                        break
                # And not across a corner either. The destination is 2-4 m along
                # the route, so near a turn it is round the corner, and the
                # straight line to it cuts across whatever the route goes past.
                # `chemistry lab` turns into a south door 1.2 m after the
                # cartons: the line from the lane to the far side of the turn
                # crossed them, avoidance sent the dog round their east end,
                # and it pivoted beside them with its body in them. If the
                # route itself -- the dog to the corner, the corner to the
                # destination -- is clear of everything the LiDAR has found,
                # there is nothing to go round: follow it.
                #
                # Round a person who has stopped, the room is a person's, not a
                # crate's. Where there is not that much, the crate's will do --
                # slowly, below -- rather than no way past at all.
                need = PASS_NEED if self.passing else perception.DESTINATION_CLEAR
                dest, corner = state["destination"], waypoints[i]
                around = (math.dist((x, y), corner) + math.dist(corner, dest)
                          > math.dist((x, y), dest) + 0.1)
                if (around and line_clear((x, y), corner, live.detected_at, need)
                        and line_clear(corner, dest, live.detected_at, need, skip=0.0)):
                    aim, offset = dest, 0.0
                else:
                    aim, offset = free_destination(dest, (x, y), live, probe=probe,
                                                   prefer=lean, need=need)
                    if aim is None and self.passing:
                        aim, offset = free_destination(dest, (x, y), live,
                                                       probe=probe, prefer=lean)
                # Which way round it went, kept until the thing is out of
                # sight: a detour that changes its mind halfway is a swerve,
                # and the announcement below has already told the person which
                # way they are about to be led.
                if offset:
                    lean = math.copysign(1.0, offset)
                # One tick with no clear goal is noise -- the scan is rebuilt
                # from scratch every tick and a single ray landing awkwardly
                # should not start the stopping sequence.
                no_goal = no_goal + 1 if aim is None else 0
                no_way = aim is None
                blocked = no_goal >= BLOCKED_TICKS
                # Re-aiming the VLM is still a TRACK-only affair: a goal pixel
                # is what VAMOS consumes, and during an alignment turn there is
                # not one. Steering by the displaced goal, below, needs no
                # pixel at all.
                if tracking and aim is not None and offset:
                    moved = project_to_pixel(aim, self.robot.camera_pose(), self.cam)
                    if moved["state"] == "TRACK":
                        moved["destination"] = aim
                        state = moved

            # Ask again sooner while nothing has cleared: the view changes as
            # the dog closes in, and a candidate that was not there at 4 m
            # often is at 2 m.
            # Asked on schedule, answered whenever the answer lands; the loop
            # carries on either way. `asked` is an answer arriving this tick.
            due = n % (REPLAN_BLOCKED if proposed is None else REPLAN_EVERY) == 0
            if ((self.vamos or self.shadow) and state["state"] == "TRACK" and due
                    and self.pending is None):
                self.ask_vamos(floor, n, state)
            answer = self.take_answer(floor, n, (x, y, yaw),
                                      state.get("destination") or waypoints[-1])
            asked = answer is not None
            if asked:
                proposed, candidates, safety = answer
            if self.vamos:
                chosen = proposed

            self.frame(state, proposed, candidates, safety if proposed else None)

            if math.hypot(waypoints[-1][0] - x, waypoints[-1][1] - y) < goal_r:
                for line in (announcements[-1] if announcements else []):
                    self.say(line)
                return True, n

            # VAMOS decides HOW when it has a usable path; the map still
            # decides WHERE, and alignment turns stay with the rules.
            # While a detour is on, the displaced goal steers and VAMOS keeps
            # proposing and being judged (its candidates and the gate's
            # rejections are in the video). Not a demotion -- a measurement.
            # Zero-shot VAMOS spreads its five candidates about +/-0.25 m over a
            # 2 m path, and getting round a crate in this corridor takes about
            # 0.65 m, so its paths are safe over their own length, pass the
            # gate, and still walk the dog into the thing. The gate cannot
            # reject a path that is fine for 2 m and leads nowhere at 4 m.
            # Spec L4 step 5 -- the LoRA fine-tune -- is what would let the
            # model make this turn itself; until then the map makes it.
            #
            # Only what is left of the path in front of the dog. It is held
            # until the next answer lands, which can now be a couple of metres
            # of walking later, and a path walked to its end is spent, not a
            # reason to turn round for its first point.
            rest = ahead(chosen, (x, y)) if chosen else []
            on_vamos = (len(rest) >= 2 and state["state"] == "TRACK" and not offset
                        and not self.door_near(gates, (x, y), DOOR_MAP_D))
            if on_vamos:
                tx, ty = path_target(rest, (x, y))
            elif offset:
                # Nothing VAMOS offered survived the gate, but the goal has been
                # moved clear of what the sensor found, so steer at that rather
                # than at the A* waypoint, which may be inside a crate. Only
                # when it actually moved: with nothing detected this is the
                # ordinary waypoint-following the rest of the run depends on.
                tx, ty = aim
            else:
                tx, ty = waypoints[i]
            at_door = False
            if not offset:
                (tx, ty), at_door = self.funnel(gates, (x, y), (tx, ty))
            if asked and self.shadow:
                self.log_shadow(floor, n, (x, y, yaw), (tx, ty), proposed, safety)

            # Reported, not acted on: free_destination already decided whether
            # there is a way through, and this is how much room it left.
            self.obs["min_clear"] = min(self.obs["min_clear"],
                                        self.clear_ahead(live, (x, y), (tx, ty)))
            # Stepping round something is not being stuck: only count it as
            # searching when there is no detour to follow either -- and no
            # clear way on the route itself. This used to crawl whenever VAMOS
            # had nothing and the LiDAR saw anything at all, a crate 8 m off or
            # somebody at the far end of the corridor, while free_destination
            # had already found the route clear and the map-only dog walked it
            # at full pace. Gate D, 21 pairs: 266.7 s of crawling under
            # --vamos against 0.0 s map-only, and the candidate slower on every
            # pair (chemistry lab +33 s). VAMOS offering nothing is not a
            # reason to slow down; the map having no way past is.
            searching = (self.vamos and chosen is None and not offset
                         and len(live.points) > 0 and no_way)
            touching = live.detected_at(x, y) < ROBOT_R

            err = wrap(math.atan2(ty - y, tx - x) - yaw)
            # Zones slow the dog down as well as stopping it: "grass ahead" is
            # a speed modifier, not just a sentence.
            speed = self.robot.MAX_V * self.router.behavior.query_actions(x, y)["speed_modifier"]
            # People first, and before every other speed rule below: none of
            # them can see that the thing ahead is walking. free_destination would
            # happily route the dog through the gap behind somebody, and the
            # gap moves.
            person = self.yield_to(tracks, (x, y), yaw, speed)
            # And so does the safety factor: a path the dog only just believes
            # it can walk is walked at half pace.
            if not len(live.points):
                # Nothing in sight any more: next time is a new obstacle and
                # deserves to be announced again, and gets its own side.
                hunting = halted = False
                stalled = lean = 0
            if offset and not hunting:
                # Said before the dog leans, not during: the person's arm is
                # attached to the handle, and which way it is about to go is
                # the one thing they cannot see coming. Positive offset is the
                # left-hand normal to the direction of travel.
                hunting = True
                self.say("Something ahead. Stepping around it to your "
                         f"{'left' if offset > 0 else 'right'}.")
            if touching:
                # Unconditional, and above everything else. This used to live
                # only in the branch below, so a dog on a detour had no
                # proximity stop at all and would drive through the thing it
                # was going round. Nothing that moves the dog gets to skip it.
                speed = 0.0
                stalled += 1
                if not halted:
                    halted = True
                    self.say("Stopping. That is too close.")
                    if os.environ.get("CYBERDOG_TRACE"):
                        peeps = [(round(math.hypot(px - x, py - y), 2), round(px, 1), round(py, 1))
                                 for px, py in self.crowds[floor].positions()]
                        print(f"    [trace] too close at ({x:.2f},{y:.2f}) "
                              f"detected={live.detected_at(x, y):.2f} "
                              f"all={live(x, y):.2f} offset={offset:.2f} aim={aim} "
                              f"people={sorted(peeps)[:2]}")
                # Then get out of it. Standing still is not a recovery from
                # being too close to something -- see BACK_V -- so back off
                # the way the dog came, which is the one direction it has
                # already been. Only if that is still clear: reversing blind
                # into a corridor is how a guide dog trips somebody up.
                behind = (x - BACK_STEP * math.cos(yaw),
                          y - BACK_STEP * math.sin(yaw))
                if (live.detected_at(*behind) > ROBOT_R
                        and live.static_at(*behind) > perception.STATIC_MIN):
                    speed, err = -BACK_V, 0.0
                if stalled > BLOCKED_S * FPS:
                    self.say("I cannot find a way past this. Stopping here.")
                    return False, n
            elif on_vamos:
                speed *= safety
            elif offset:
                speed *= DETOUR          # deliberate, not a swerve
            elif searching or blocked:
                # Something is there and nothing has been approved past it.
                # Crawl while asking again; stop once it is close and the
                # answer has not changed; give up out loud rather than
                # grinding to MAX_TICKS in silence.
                if not hunting:
                    hunting = True
                    self.say("Something ahead. Finding a way round.")
                speed *= CRAWL
                self.obs["crawl"] += 1
                if blocked or touching:
                    speed, stalled = 0.0, stalled + 1
                    self.obs["stopped"] += 1
                    if not halted:
                        halted = True
                        self.say("Stopping. I have no clear way past.")
                    if stalled > BLOCKED_S * FPS:
                        self.say("I cannot find a way past this. Stopping here.")
                        return False, n
                else:
                    stalled, halted = 0, False
            if self.passing is not None:
                # Walking past somebody, not striding: the person on the handle
                # passes them at the same distance a moment later.
                px, py = self.passing
                if math.dist((x, y), (px, py)) < PASS_R:
                    speed = min(speed, PASS_V)
                elif (px - x) * math.cos(yaw) + (py - y) * math.sin(yaw) < 0:
                    self.passing = None          # behind the dog, and clear
            if person is not None:
                # Waiting is not being stuck, and the give-up timer must not
                # think it is. Without this the dog announces that it cannot
                # find a way past a corridor it is standing in politely.
                stalled = 0
                # Stopped, and not turning either: swinging round to track
                # somebody walking past is what a dog does and is not what a
                # handle attached to a person's arm should do.
                if self.waited < YIELD_MAX * FPS:
                    speed, err = 0.0, 0.0
                else:
                    # They are not going anywhere. Neither can the dog stand
                    # here for ever, so creep and keep asking.
                    if self.waited == int(YIELD_MAX * FPS):
                        self.say("They are not moving. Going slowly.")
                    speed *= CRAWL

            turn_only = DOOR_ERR if at_door else TURN_ONLY
            cmd = (0.0 if abs(err) > turn_only else speed * math.cos(err), 0.0, K_W * err)
            if cmd[0] > 0.0 and self.nose_closing(live, (x, y, yaw), cmd):
                cmd = (0.0, 0.0, cmd[2])
            if speed > 0.0:
                people = [t for t in tracks if t.moving or MIN_RADIUS <= t.r <= MAX_RADIUS]
                cmd = self.spare_handler(live, (x, y, yaw), cmd, speed, people)
            self.commands.update(struct.pack("3d", *cmd))
            self.n_commands += 1
            self.robot.set_velocity(*cmd)
            self.robot.step()

            if verbose and n % 60 == 0:
                print(f"    t={self.robot.sim_time:6.1f}s  floor {floor}  "
                      f"pos=({x:5.1f},{y:5.1f})  wp={i}/{len(waypoints) - 1}  "
                      f"{state['state']}")
        return False, MAX_TICKS

    def confirm(self, prompt):
        """Stop, say something, and wait for a human to act on it.

        The robot cannot press a call button, and pretending otherwise is the
        kind of shortcut that turns into a person standing in front of a lift
        that was never called. So the handover is explicit: it asks, a human
        does the part that needs hands, and the run continues.

        --auto-confirm (and any non-interactive stdin, which is every batch
        run) answers for them, after a visible pause, so the video still shows
        the stop.
        """
        self.say(prompt)
        if self.auto_confirm or not sys.stdin.isatty():
            for _ in range(int(WAIT_S * FPS)):
                self.step_crowd()
                self.frame()
            print("    (auto-confirmed)")
            return
        input("    press Enter once the lift is here and the doors are open... ")

    def glide(self, to_xy, yaw, z):
        """Walk a straight short line with the pose set directly.

        Used only for the few metres in and out of the car, where the velocity
        controller would turn on the spot inside a shaft it cannot see. The
        real Go2 walks this under its own controller; here it is scripted, the
        way the whole kinematic twin is.
        """
        x, y, yaw0 = self.robot.get_pose()
        # Turned over the walk, not snapped on the first frame: the dog reaches
        # the lift on whatever heading the corridor left it with, and setting
        # the target yaw outright made the board a teleported quarter-turn.
        dyaw = wrap(yaw - yaw0)
        ticks = max(int(math.dist((x, y), to_xy) / (STEP_V * self.robot.CONTROL_DT)), 1)
        for k in range(1, ticks + 1):
            t = k / ticks
            self.robot.place((x + (to_xy[0] - x) * t, y + (to_xy[1] - y) * t),
                             yaw0 + dyaw * t, z=z)
            self.step_crowd()
            self.frame()

    def pivot(self, from_yaw, to_yaw, z, seconds=1.0):
        """Turn on the spot, scripted, inside the car."""
        d = wrap(to_yaw - from_yaw)
        xy = self.robot.get_pose()[:2]
        for k in range(1, int(seconds / self.robot.CONTROL_DT) + 1):
            self.robot.place(xy, from_yaw + d * k / (seconds / self.robot.CONTROL_DT), z=z)
            self.step_crowd()
            self.frame()

    def hold(self, seconds):
        """Stand still, still filming."""
        for _ in range(int(seconds / self.robot.CONTROL_DT)):
            self.robot.place(self.robot.get_pose()[:2], self.robot.get_pose()[2])
            self.step_crowd()
            self.frame()

    def face(self, point):
        """Turn to look at a point on the map, walking the turn."""
        x, y, yaw = self.robot.get_pose()
        self.pivot(yaw, math.atan2(point[1] - y, point[0] - x), self.robot.z, seconds=1.2)

    def lift_ride(self, from_floor, to_floor):
        """Change floor the only way this robot will: in the lift.

        The leg in stops at lift.WAIT_XY, outside the open face (see
        lift_approach), so the dog is standing in front of the doors when it
        gets here rather than already in the shaft. This is the bit in between:
        ask, walk in, ride, announce, step back out.
        """
        if to_floor not in lift.FLOORS_SERVED:
            raise HazardStop(f"the lift does not serve floor {to_floor}")

        self.in_lift = True
        z_from = levels.floor_z(from_floor)
        # Asked from outside, and nothing moves until it is answered: a dog
        # that boards and then asks for the button has already committed the
        # person it is leading to a car that may not be there.
        self.confirm(f"Lift ahead. Press the call button for floor {to_floor}, "
                     f"then press continue.")
        # Turn to face the doors, then walk straight in through the open west
        # face. Turn first, walk second: rolling the two together -- gliding to
        # the car while the yaw caught up -- walked the dog in sideways, which
        # is not a thing a dog does and looked exactly as wrong as it was.
        self.face(lift.LIFT_XY)
        self.glide(lift.LIFT_XY, 0.0, z_from)
        self.pivot(self.robot.get_pose()[2], math.pi, z_from)
        # Facing the doors, in the car, waiting to go: that is the shot.

        z0, z1 = z_from, levels.floor_z(to_floor)
        yaw = math.pi                      # facing the doors for the ride
        self.say(f"Doors closing. Going to floor {to_floor}.")
        ticks = max(int(abs(z1 - z0) / (lift.RIDE_V * self.robot.CONTROL_DT)), 1)
        for k in range(1, ticks + 1):
            z = z0 + (z1 - z0) * k / ticks
            self.robot.place(lift.LIFT_XY, yaw, z=z)
            self.robot.move_mocap("lift_car", lift.car_pos(z))
            # Round to the nearest storey so the pip in the video flips as the
            # car passes each slab, not only when it arrives.
            self.floor = min(max(round(z / levels.FLOOR_HEIGHT) + 1,
                                 lift.FLOORS_SERVED[0]), lift.FLOORS_SERVED[-1])
            self.frame()

        self.robot.set_height(z1)
        self.floor = to_floor
        self.say(f"Floor {to_floor}. Doors opening.")
        for _ in range(int(WAIT_S * FPS)):
            self.frame()
        # Back out west into the corridor before the velocity loop takes over,
        # so the first thing it does is not a turn into the shaft wall. To the
        # same waiting point the leg out of the lift now starts from.
        self.glide(lift.WAIT_XY, math.pi, z1)
        self.in_lift = False


def main():
    ap = argparse.ArgumentParser(description="CyberDog whole-building simulation")
    ap.add_argument("command", nargs="?", default="take me to the server room",
                    help="natural-language command; multi-stop is split by rules")
    ap.add_argument("--out", default=str(paths.SCENE_CACHE / "building.mp4"))
    ap.add_argument("--no-video", action="store_true", help="drive without rendering")
    ap.add_argument("--nlu", action="store_true", help="parse through the Gemma layer")
    ap.add_argument("--vamos", action="store_true", help="VLM in the steering loop")
    ap.add_argument("--shadow", action="store_true",
                    help="ask VAMOS and score its paths with the dream, but let the map "
                         "route drive; every call is logged to --shadow-log")
    ap.add_argument("--shadow-log", default=str(paths.OUTPUT_DIR / "shadow_log.jsonl"),
                    metavar="PATH", help="where --shadow writes one JSON line per VAMOS call")
    ap.add_argument("--vamos-url", default=None, metavar="URL",
                    help="VAMOS server (default http://127.0.0.1:8009)")
    ap.add_argument("--vlm-latency", type=float, default=None, metavar="S",
                    help="with --vamos or --shadow: seconds of sim time before a VAMOS "
                         "answer arrives; the control loop runs on meanwhile (default: "
                         "the call's measured time; fix it for repeatable runs)")
    ap.add_argument("--vamos-sample", action="store_true",
                    help="with --vamos or --shadow: let VAMOS sample its paths "
                         "(temperature 1.0, different every run) instead of beam "
                         "search, which gives the same paths for the same frame")
    ap.add_argument("--auto-confirm", action="store_true",
                    help="answer the lift handover prompt instead of waiting for a human")
    ap.add_argument("--pedestrians", type=int, default=0, metavar="N",
                    help=f"people walking the corridors on each floor (0-{pedestrians.POOL}); "
                         "they are on no map, and the dog stops for them")
    ap.add_argument("--seed", type=int, default=0,
                    help="which crowd -- the same seed is the same people every run")
    ap.add_argument("--speed", type=int, default=1, metavar="N",
                    help="play the video back N times faster (control still runs at 20 Hz)")
    args = ap.parse_args()
    if args.vamos and args.shadow:
        ap.error("--vamos steers with VAMOS and --shadow never does; pick one")

    scene = str(paths.BUILDING_SCENE)
    if not os.path.exists(scene):
        raise SystemExit("no building scene yet -- run: python -m cyberdog.sim.scene.build_scene --building")

    router = BuildingRouter()
    # A refusal -- stairs as a destination, or a floor the lift misses -- is an
    # answer, not a crash. It is the sentence the user would hear.
    try:
        stops = resolve_stops(router, args.command, use_nlu=args.nlu)
        print(f'command: "{args.command}"')
        for k, (name, floor) in enumerate(stops, 1):
            print(f"  stop {k}: {name}" + (f" (floor {floor})" if floor else ""))

        start_xy, legs, faces = plan_stops(router, stops)
    except NoAccessibleRoute as refusal:
        raise SystemExit(f"[voice] {refusal}")

    print(f"{len(legs)} leg(s) across floors "
          f"{sorted({f for f, _, _, _ in legs})}, starting at {START_LOCATION}")

    first = legs[0][2]
    yaw0 = math.atan2(first[1][1] - first[0][1], first[1][0] - first[0][0]) \
        if len(first) > 1 else 0.0
    run = Run(scene, start_xy, yaw0, router, out=None if args.no_video else args.out,
              vamos=args.vamos, auto_confirm=args.auto_confirm, speed=args.speed,
              crowd=args.pedestrians, seed=args.seed,
              shadow=args.shadow_log if args.shadow else None, vamos_url=args.vamos_url,
              vlm_latency=args.vlm_latency, vamos_sample=args.vamos_sample)

    ok, halted = True, None
    try:
        for k, (floor, name, waypoints, announcements) in enumerate(legs):
            changing = k + 1 < len(legs) and legs[k + 1][0] != floor
            print(f"  leg {k + 1}/{len(legs)}: floor {floor}, {len(waypoints)} checkpoints "
                  f"-> {'the lift' if changing else name}")
            ok, ticks = run.follow(waypoints, floor, announcements,
                                   goal_r=LIFT_GOAL_R if changing else GOAL_R)
            if not ok:
                print(f"  leg {k + 1} timed out after {ticks} ticks")
                break
            if k in faces:
                run.face(faces[k])
            if changing:
                run.lift_ride(floor, legs[k + 1][0])
            else:
                # Arrived somewhere the person asked for. Stand still long
                # enough for that to register -- a guide dog that announces a
                # destination and immediately walks off has not delivered it.
                run.hold(WAIT_S * 2)
    except HazardStop as stop:
        ok, halted = False, stop
    if ok:
        run.behind = True
        for _ in range(FPS * 2):      # hold the arrival, seen over its shoulder
            run.frame()
    run.close()

    x, y, _ = run.robot.get_pose()
    if halted is not None:
        print(f"HALTED {halted}")
    print(f"{'ARRIVED' if ok else 'FAILED'} on floor {run.floor} at ({x:.1f}, {y:.1f}) "
          f"after {run.robot.sim_time:.0f}s of sim time")
    for floor, p in sorted(run.policies.items()):
        st = p.stats
        mean = st["safety_sum"] / st["chosen"] if st["chosen"] else 0.0
        print(f"VAMOS floor {floor}: {st['calls']} calls, {st['failures']} failed, "
              f"{st['rejected']} candidates rejected "
              f"({st['rejected_by_gate']} of them by the safety gate), "
              f"mean safety of the paths it followed {mean:.2f}")
    if args.vamos or args.shadow:
        v = run.vlm
        k = max(v["answers"], 1)
        dropped = v["dropped"] + (run.pending is not None)
        print(f"VLM ASYNC: {v['asked']} asked, {v['answers']} answered, {dropped} "
              f"dropped at a leg's end; answers were {v['age'] / k:.1f}s old on "
              f"arrival, the dog {v['moved'] / k:.2f} m on from where it asked; "
              f"the control loop never waited")
    if run.shadow is not None:
        run.shadow.close()
        sn = run.shadow_n
        print(f"SHADOW: {sn['calls']} VAMOS calls logged, {sn['steer']} would have "
              f"steered more than {math.degrees(SHADOW_DISAGREE):.0f} deg off the map "
              f"route, {sn['none']} had nothing pass the gate -> {args.shadow_log}")
    # Tick for tick what the dog was told to do; compare across runs.
    print(f"commands: {run.n_commands} ticks, sha256 {run.commands.hexdigest()[:16]}")
    o, h = run.obs, run.hits
    print(f"obstacles: up to {o['seen']} returns the map could not explain, "
          f"{o['crawl'] / FPS:.1f}s crawling, {o['stopped'] / FPS:.1f}s stopped, "
          f"least clearance ahead {o['min_clear']:.2f} m")
    # Ground truth, and the only line here the robot cannot flatter itself on.
    if h["ticks"]:
        print(f"COLLISIONS: {h['ticks'] / FPS:.1f}s touching {', '.join(sorted(h['boxes']))}")
    else:
        print("collisions: none -- no part of the dog touched a wall or an obstacle")
    hh = run.handler_hits
    if hh["ticks"]:
        print(f"HANDLER: {hh['ticks'] / FPS:.1f}s with the person on the handle touching "
              f"{', '.join(sorted(hh['boxes']))}")
    else:
        print("handler: none -- the person on the handle never touched a wall or an obstacle")
    print(f"turns eased to keep the person on the handle clear: {o['eased'] / FPS:.1f}s")
    # Also when there are none: a yield in an empty building is a false
    # positive, and it should be as visible as a collision is.
    if args.pedestrians or run.ped["yields"]:
        pd = run.ped
        print(f"people: {pd['yields']} times it stopped to let someone past, "
              f"{pd['waited'] / FPS:.1f}s waiting in total")
        if pd["near"]:
            print(f"CONTACT: {pd['near'] / FPS:.1f}s within {PED_NEAR} m of a person "
                  f"(closest {pd['min_d']:.2f} m)")
        else:
            print(f"contact: none -- closest it came to anybody was {pd['min_d']:.2f} m")
    if not args.no_video:
        pace = "real time" if args.speed == 1 else f"{args.speed}x real time"
        print(f"video -> {args.out}  ({run.robot.sim_time / args.speed:.0f}s, {pace})")


if __name__ == "__main__":
    main()

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
    python -m cyberdog.sim.run_building "server room" --auto-confirm       # don't wait at the lift

Without --nlu the destination is matched against locations.json by name, which
keeps torch out of the process; --nlu runs the real Input Treating Layer.
"""
import argparse
import math
import os
import sys

import numpy as np


from cyberdog import paths
from cyberdog.planning.building_router import BuildingRouter, NoAccessibleRoute
from cyberdog.planning.checkpoint_projector import (load_camera_config,
                                                    pick_carrot, project_route,
                                                    project_to_pixel,
                                                    vamos_prompt)
from cyberdog.sim.control import (GOAL_R, K_W, TURN_ONLY, advance, path_target,
                                  wrap)
from cyberdog.sim.overlay import (draw_marker, draw_path, draw_points,
                                  safety_bar)
from cyberdog.sim.robot.mujoco_robot import MujocoRobot
from cyberdog.sim.scene import levels, lift, obstacles
from cyberdog.sim.sensing import perception
from cyberdog.sim.sensing.dreaming import ROBOT_R
from cyberdog.sim.sensing.lidar import MOUNT_H, Lidar
from cyberdog.sim.sensing.perception import (LiveClearance, free_carrot,
                                             line_clear)

START_LOCATION = "main entrance"
START_FLOOR = 1
FPS = 20                  # equals the control rate, so the video is real time
WAIT_S = 1.5              # held at the lift doors, and while auto-confirming
ANNOUNCE_R = 2.5          # metres out that a checkpoint's line is spoken
STEP_V = 0.4              # m/s in and out of the car -- slower than corridor pace
REPLAN_EVERY = 20         # ticks between VLM calls when --vamos is on
REPLAN_BLOCKED = 10       # ...and while nothing VAMOS offered clears an obstacle
# Flatter and closer than the single-floor demo: every storey now has a slab
# over it, and the old -28 degree chase camera sat inside the ceiling, which
# rendered as solid grey.
CHASE_D, CHASE_EL = 3.2, -11
# Azimuth, distance, elevation for the lift -- see frame(). Azimuth is the
# direction the camera looks along, so 0 puts it west of the shaft looking in
# through the open face.
LIFT_CAM = (0, 3.4, -6)
MAX_TICKS = 6000          # per leg

# What to do when the LiDAR has found something and VAMOS has nothing that
# clears it. Not one rule but three, because "stop" and "carry on" are both
# wrong: crawl while asking again, and only stop once it is close and the
# answer has not changed. The spec wants a pedestrian avoided *without* full
# stops (L6 acceptance); a full stop is what is left when avoidance has failed.
CRAWL = 0.30              # fraction of top speed while looking for a way round
DETOUR = 0.70             # ...and while actually stepping around something
# There is deliberately no "clearance ahead" stop threshold. free_carrot is
# the single authority on whether a way past exists -- it validates the line to
# the goal itself -- and a second test of the same line against a different
# number is how the dog ended up halting in gaps it was successfully using.
# What is left is the honest emergency: ROBOT_R, the distance at which the dog
# is touching something rather than approaching it.
BLOCKED_S = 8.0           # seconds stopped with no way past before giving up
LOOKAHEAD = 3.0           # metres of the route ahead tested for obstructions
BLOCKED_TICKS = 5         # consecutive ticks with no clear goal before believing it
PROBE_D = 6.0             # metres ahead the blockage is judged at, not the carrot's 2-4


def resolve_stops(router, text, use_nlu=False):
    """Command -> [(destination name, floor or None)], one entry per stop.

    Splitting and floor phrases are rules either way (that is where they live
    in the pipeline); only the destination itself needs the model, and with
    --nlu off it is matched against the map's own names instead.
    """
    from cyberdog.language.command_splitter import split_destinations
    from cyberdog.language.floor_parser import extract_floor

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
        planned = router.plan(floor, xy, target)
        if planned is None:
            raise SystemExit(f"no route to {name}")
        for f, route in planned:
            legs.append((f, name,
                         [(float(c.position[0]), float(c.position[1]))
                          for c in route.checkpoints],
                         [list(c.announcements) for c in route.checkpoints]))
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
                 auto_confirm=False, speed=1):
        self.robot = MujocoRobot(scene, start_xy=start_xy, start_yaw=start_yaw,
                                 start_z=levels.floor_z(START_FLOOR))
        self.cam = load_camera_config()
        self.router = router
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
        self.lidar = Lidar(self.robot.model, self.robot.data)
        self.live = {}            # floor -> LiveClearance, one per storey
        self.seen = np.empty((0, 2))   # this tick's unexplained returns
        # Ground truth, for scoring only -- never shown to the robot. The dog
        # finds these with the LiDAR or not at all; this is how we check.
        self.hits = {"ticks": 0, "boxes": set()}
        self.obs = {"crawl": 0, "stopped": 0, "seen": 0, "min_clear": 99.0}

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
        self.writer.append_data(np.hstack([left, dog]))

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
            live = self.clearance(floor)
            p = VamosPolicy(self.cam, lambda x, y: live(x, y) >= ROBOT_R,
                            dream=Dream(live, self.robot.MAX_V))
            if not p.available():
                raise SystemExit("VAMOS server is not answering on 127.0.0.1:8009 -- "
                                 "start vendor/VAMOS/server/vlm_server.py first")
            self.policies[floor] = p
        return self.policies[floor]

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

    def score_collision(self, floor, x, y):
        """Ground truth: is the dog standing inside an obstacle right now?

        Scoring only, and never fed back into the robot -- the dog finds these
        with the LiDAR or it does not find them. A run that reports arrival
        while this counter climbed is a run that walked through a crate, which
        is precisely the failure that was invisible before.
        """
        for bx0, bx1, by0, by1, _h, name in obstacles.boxes(floor):
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                self.hits["ticks"] += 1
                self.hits["boxes"].add(f"floor {floor} {name}")

    def follow(self, waypoints, floor, announcements=None, verbose=True):
        """Walk one floor's leg. Same controller as run_demo, plus the frames."""
        self.floor = floor
        self.robot.set_height(levels.floor_z(floor))
        i = 1 if len(waypoints) > 1 else 0
        said = set()
        chosen, candidates, safety = None, [], 1.0
        hunting = halted = False
        stalled = no_goal = 0

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
            live = self.clearance(floor)
            live.update(self.lidar.scan((x, y), self.robot.z), self.robot.z,
                        (x, y), origin=(x, y, self.robot.z + MOUNT_H))
            self.seen = live.points
            self.obs["seen"] = max(self.obs["seen"], len(live.points))
            self.score_collision(floor, x, y)

            state = project_route(self.robot.camera_pose(), waypoints[i:], self.cam)

            # "Map decides WHERE": the same carrot, moved sideways when it or
            # the line to it is blocked by something the map never had. VAMOS
            # drives at whatever goal pixel it is given, so a carrot inside a
            # crate is five candidate paths into the crate -- measured, before
            # this existed. Move the goal and the model has something to solve.
            aim, offset, blocked = None, 0.0, False
            if state["state"] == "TRACK":
                # Judge the blockage further out than the carrot -- see
                # free_carrot. PROBE_D is far enough to start moving across
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
                    q = pick_carrot((x, y), waypoints[i:], reach, reach + 2.0)
                    if q is not None and line_clear((x, y), q, live.static_at, ROBOT_R):
                        probe = q
                        break
                aim, offset = free_carrot(state["carrot"], (x, y), live, probe=probe)
                # One tick with no clear goal is noise -- the scan is rebuilt
                # from scratch every tick and a single ray landing awkwardly
                # should not start the stopping sequence.
                no_goal = no_goal + 1 if aim is None else 0
                blocked = no_goal >= BLOCKED_TICKS
                if aim is not None and offset:
                    moved = project_to_pixel(aim, self.robot.camera_pose(), self.cam)
                    if moved["state"] == "TRACK":
                        moved["carrot"] = aim
                        state = moved

            # Ask again sooner while nothing has cleared: the view changes as
            # the dog closes in, and a candidate that was not there at 4 m
            # often is at 2 m.
            due = n % (REPLAN_BLOCKED if chosen is None else REPLAN_EVERY) == 0
            if self.vamos and state["state"] == "TRACK" and due:
                chosen, candidates, safety = self.policy(floor).plan(
                    self.robot.get_image(), vamos_prompt(state),
                    self.robot.camera_pose(), state["carrot"], pose=(x, y, yaw))

            self.frame(state, chosen, candidates, safety if chosen else None)

            if math.hypot(waypoints[-1][0] - x, waypoints[-1][1] - y) < GOAL_R:
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
            on_vamos = bool(chosen) and state["state"] == "TRACK" and not offset
            if on_vamos:
                tx, ty = path_target(chosen, (x, y))
            elif offset:
                # Nothing VAMOS offered survived the gate, but the goal has been
                # moved clear of what the sensor found, so steer at that rather
                # than at the A* waypoint, which may be inside a crate. Only
                # when it actually moved: with nothing detected this is the
                # ordinary waypoint-following the rest of the run depends on.
                tx, ty = aim
            else:
                tx, ty = waypoints[i]

            # Reported, not acted on: free_carrot already decided whether
            # there is a way through, and this is how much room it left.
            self.obs["min_clear"] = min(self.obs["min_clear"],
                                        self.clear_ahead(live, (x, y), (tx, ty)))
            # Stepping round something is not being stuck: only count it as
            # searching when there is no detour to follow either.
            searching = (self.vamos and chosen is None and not offset
                         and len(live.points) > 0)
            touching = live.detected_at(x, y) < ROBOT_R

            err = wrap(math.atan2(ty - y, tx - x) - yaw)
            # Zones slow the dog down as well as stopping it: "grass ahead" is
            # a speed modifier, not just a sentence.
            speed = self.robot.MAX_V * self.router.behavior.query_actions(x, y)["speed_modifier"]
            # And so does the safety factor: a path the dog only just believes
            # it can walk is walked at half pace.
            if not len(live.points):
                # Nothing in sight any more: next time is a new obstacle and
                # deserves to be announced again.
                hunting = halted = False
                stalled = 0
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
            self.robot.set_velocity(
                0.0 if abs(err) > TURN_ONLY else speed * math.cos(err),
                0.0, K_W * err)
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
        x, y, _ = self.robot.get_pose()
        ticks = max(int(math.dist((x, y), to_xy) / (STEP_V * self.robot.CONTROL_DT)), 1)
        for k in range(1, ticks + 1):
            t = k / ticks
            self.robot.place((x + (to_xy[0] - x) * t, y + (to_xy[1] - y) * t), yaw, z=z)
            self.frame()

    def pivot(self, from_yaw, to_yaw, z, seconds=1.0):
        """Turn on the spot, scripted, inside the car."""
        d = wrap(to_yaw - from_yaw)
        xy = self.robot.get_pose()[:2]
        for k in range(1, int(seconds / self.robot.CONTROL_DT) + 1):
            self.robot.place(xy, from_yaw + d * k / (seconds / self.robot.CONTROL_DT), z=z)
            self.frame()

    def hold(self, seconds):
        """Stand still, still filming."""
        for _ in range(int(seconds / self.robot.CONTROL_DT)):
            self.robot.place(self.robot.get_pose()[:2], self.robot.get_pose()[2])
            self.frame()

    def face(self, point):
        """Turn to look at a point on the map, walking the turn."""
        x, y, yaw = self.robot.get_pose()
        self.pivot(yaw, math.atan2(point[1] - y, point[0] - x), self.robot.z, seconds=1.2)

    def lift_ride(self, from_floor, to_floor):
        """Change floor the only way this robot will: in the lift.

        The planner hands over at the lift point and picks up at the same xy
        on the other floor, and the dog is already standing in the car when it
        gets here -- that point is inside the car's footprint. This is the bit
        in between: ask, wait, ride, announce.
        """
        if to_floor not in lift.FLOORS_SERVED:
            raise HazardStop(f"the lift does not serve floor {to_floor}")

        self.in_lift = True
        z_from = levels.floor_z(from_floor)
        self.confirm(f"Lift ahead. Press the call button for floor {to_floor}, "
                     f"then press continue.")
        # East into the car through its open west face, then turn to face the
        # doors: that is where the person is, and it is what the dog's own
        # camera should be looking at on the way up.
        self.glide(lift.LIFT_XY, 0.0, z_from)
        self.pivot(0.0, math.pi, z_from)

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
        # so the first thing it does is not a turn into the shaft wall.
        self.glide((lift.CAR[0] - 0.4, lift.LIFT_XY[1]), math.pi, z1)
        self.in_lift = False


def main():
    ap = argparse.ArgumentParser(description="CyberDog whole-building simulation")
    ap.add_argument("command", nargs="?", default="take me to the server room",
                    help="natural-language command; multi-stop is split by rules")
    ap.add_argument("--out", default=str(paths.SCENE_CACHE / "building.mp4"))
    ap.add_argument("--no-video", action="store_true", help="drive without rendering")
    ap.add_argument("--nlu", action="store_true", help="parse through the Gemma layer")
    ap.add_argument("--vamos", action="store_true", help="VLM in the steering loop")
    ap.add_argument("--auto-confirm", action="store_true",
                    help="answer the lift handover prompt instead of waiting for a human")
    ap.add_argument("--speed", type=int, default=1, metavar="N",
                    help="play the video back N times faster (control still runs at 20 Hz)")
    args = ap.parse_args()

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
              vamos=args.vamos, auto_confirm=args.auto_confirm, speed=args.speed)

    ok, halted = True, None
    try:
        for k, (floor, name, waypoints, announcements) in enumerate(legs):
            changing = k + 1 < len(legs) and legs[k + 1][0] != floor
            print(f"  leg {k + 1}/{len(legs)}: floor {floor}, {len(waypoints)} checkpoints "
                  f"-> {'the lift' if changing else name}")
            ok, ticks = run.follow(waypoints, floor, announcements)
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
    o, h = run.obs, run.hits
    print(f"obstacles: up to {o['seen']} returns the map could not explain, "
          f"{o['crawl'] / FPS:.1f}s crawling, {o['stopped'] / FPS:.1f}s stopped, "
          f"least clearance ahead {o['min_clear']:.2f} m")
    # Ground truth, and the only line here the robot cannot flatter itself on.
    if h["ticks"]:
        print(f"COLLISIONS: {h['ticks'] / FPS:.1f}s inside {', '.join(sorted(h['boxes']))}")
    else:
        print("collisions: none -- the dog never entered an obstacle's footprint")
    if not args.no_video:
        pace = "real time" if args.speed == 1 else f"{args.speed}x real time"
        print(f"video -> {args.out}  ({run.robot.sim_time / args.speed:.0f}s, {pace})")


if __name__ == "__main__":
    main()

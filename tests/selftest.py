"""Check the twin a piece at a time, so a failure names its own layer.

`run_building.py` is the whole stack at once: map, sensor, perception, VLM,
controller, video. When it misbehaves that is six suspects and a two-minute
render per guess. Each stage here exercises one layer and prints PASS or FAIL
with the number it judged on, so the question "which part is wrong?" is one
command and a few seconds.

    python tests/selftest.py            # every stage
    python tests/selftest.py lidar      # just one

Stages run bottom-up: scene, lidar, perception, destination, crowd, latency,
run. The first failure is usually the real one -- a bad scene fails everything
above it.
"""
import math
import os
import sys
import time


import numpy as np

from cyberdog import paths

SCENE = str(paths.BUILDING_SCENE)
_fails = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        _fails.append(name)


# -- stages ---------------------------------------------------------------

def scene():
    """Is the building the right size, and is the dog the size it claims?

    This is the one that was wrong for a long time: the walls were drawn from
    the planner's inflated grid, so the corridor came out 0.6 m too narrow and
    the dog rendered half inside them.
    """
    import mujoco
    from cyberdog.sim.scene import levels
    from cyberdog.sim.scene.build_scene import INFLATION
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot

    # Through MujocoRobot, so the dog is actually standing in the corridor:
    # the raw keyframe leaves it at the model origin and every ray below
    # misses it, which reads as a 9 m wide robot.
    r = MujocoRobot(SCENE, start_xy=(24.0, 9.5), start_yaw=0.0,
                    start_z=levels.floor_z(1))
    m, d = r.model, r.data
    gid = np.zeros(1, np.int32)

    def ray(p, v, statics=True):
        return mujoco.mj_ray(m, d, np.array(p, float), np.array(v, float),
                             None, 1 if statics else 0, -1, gid)

    z = levels.floor_z(1) + 1.0
    widths = []
    for x in (12.0, 24.0, 30.0):
        s, n = ray((x, 9.5, z), (0, -1, 0)), ray((x, 9.5, z), (0, 1, 0))
        widths.append(s + n)
    w = sum(widths) / len(widths)
    # generate_building.py: corridor 8.0-11.0, walls 0.2 thick -> 2.8 m free.
    # A cell is lost each side in the inflate/erode round trip, so allow 0.15.
    check("corridor width", abs(w - 2.8) < 0.15,
          f"{w:.2f} m drawn, 2.80 m intended (inflation {INFLATION} m removed)")

    zt = float(d.qpos[2])
    lo = ray((24.0, 6.0, zt), (0, 1, 0), False)
    hi = ray((24.0, 13.0, zt), (0, -1, 0), False)
    trunk = (13.0 - hi) - (6.0 + lo) if lo > 0 and hi > 0 else -1.0
    check("dog fits", 0 < trunk < w / 2,
          f"trunk {trunk:.2f} m wide in a {w:.2f} m corridor" if trunk > 0
          else "rays missed the dog -- it is not where the test put it")
    check("floors present", m.ngeom > 300, f"{m.ngeom} geoms")


def lidar():
    """Does the sensor see a thing that is not on the map, and how fast?"""
    from cyberdog.sim.scene import levels
    from cyberdog.sim.sensing import lidar as L
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot

    fz = levels.floor_z(2)
    r = MujocoRobot(SCENE, start_xy=(21.0, 9.8), start_yaw=math.pi, start_z=fz)
    sensor = L.Lidar(r.model, r.data)

    pts = sensor.scan((21.0, 9.8), fz)
    t = time.perf_counter()
    for _ in range(20):
        sensor.scan((21.0, 9.8), fz)
    ms = (time.perf_counter() - t) / 20 * 1000
    check("scan returns", len(pts) > 1000, f"{len(pts)} points")
    check("scan cost", ms < 20, f"{ms:.2f} ms ({1000 / ms:.0f} Hz ceiling)")
    # Every ray must start clear of the dog itself -- at lens height they all
    # come back as the robot.
    near = np.linalg.norm(pts[:, :2] - np.array([21.0, 9.8]), axis=1)
    check("no self-hits", near.min() >= L.MIN_R - 1e-6,
          f"nearest return {near.min():.2f} m, minimum range {L.MIN_R} m")


def perception():
    """Does it flag the crates, and nothing else?

    The false-positive half matters more than the detection half: a layer that
    sees obstacles everywhere stops the dog in an empty corridor.
    """
    from cyberdog.sim.scene import levels
    from cyberdog.sim.sensing import lidar as L
    from cyberdog.sim.scene import obstacles
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot
    from cyberdog.sim.sensing.perception import LiveClearance

    fz = levels.floor_z(2)
    r = MujocoRobot(SCENE, start_xy=(21.0, 9.8), start_yaw=math.pi, start_z=fz)
    sensor, live = L.Lidar(r.model, r.data), LiveClearance(2)
    boxes = obstacles.boxes(2)

    stray = seen = 0
    for x in (44.0, 40.0, 36.0, 34.5, 30.0, 20.0, 16.5, 12.0, 6.0):
        r.reset((x, 9.5), math.pi)
        r.set_height(fz)
        live.update(sensor.scan((x, 9.5), fz), fz, (x, 9.5),
                    origin=(x, 9.5, fz + L.MOUNT_H))
        p = live.points
        seen += len(p)
        if not len(p):
            continue
        inside = np.zeros(len(p), bool)
        for bx0, bx1, by0, by1, _h, _n in boxes:
            inside |= ((p[:, 0] > bx0 - 0.3) & (p[:, 0] < bx1 + 0.3) &
                       (p[:, 1] > by0 - 0.3) & (p[:, 1] < by1 + 0.3))
        stray += int((~inside).sum())
    check("obstacles found", seen > 100, f"{seen} unexplained returns over the corridor")
    check("no false positives", stray == 0,
          f"{stray} returns not from a known box (walls/stairs/lift must be explained)")

    # And the inside of a box must read as solid, not as the one visible face.
    r.reset((20.0, 9.5), math.pi)
    r.set_height(fz)
    live.update(sensor.scan((20.0, 9.5), fz), fz, (20.0, 9.5),
                origin=(20.0, 9.5, fz + L.MOUNT_H))
    x0, x1, y0, y1, _h, _n = [b for b in boxes if b[5] == "cart"][0]
    mid = live((x0 + x1) / 2, (y0 + y1) / 2)
    check("boxes are solid", mid < 0.16,
          f"clearance {mid:.2f} m at the cart's centre (a surface-only sensor "
          f"would read it as open floor)")


def destination():
    """Does a goal inside a crate get moved out of it, and only then?"""
    from cyberdog.sim.scene import levels
    from cyberdog.sim.sensing import lidar as L
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot
    from cyberdog.sim.sensing.perception import LiveClearance, free_destination

    fz = levels.floor_z(2)
    r = MujocoRobot(SCENE, start_xy=(20.0, 9.5), start_yaw=math.pi, start_z=fz)
    sensor, live = L.Lidar(r.model, r.data), LiveClearance(2)
    live.update(sensor.scan((20.0, 9.5), fz), fz, (20.0, 9.5),
                origin=(20.0, 9.5, fz + L.MOUNT_H))

    moved, off = free_destination((16.0, 9.5), (20.0, 9.5), live, probe=(12.0, 9.5))
    check("blocked goal moves", moved is not None and abs(off) > 0.2,
          f"(16.0, 9.5) -> {None if moved is None else tuple(round(v, 2) for v in moved)}, "
          f"offset {off:+.2f} m")

    clean = LiveClearance(1)       # never updated: nothing detected anywhere
    same, off2 = free_destination((24.0, 9.5), (30.0, 9.5), clean)
    check("clear goal stays put", same == (24.0, 9.5) and off2 == 0.0,
          f"with nothing detected the planner's route is left alone ({off2:+.2f} m)")


def latency():
    """What the added layers cost per control tick. The numbers for the report."""
    from cyberdog.sim.scene import levels
    from cyberdog.sim.sensing import lidar as L
    from cyberdog.sim.sensing.dreaming import Dream
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot
    from cyberdog.sim.sensing.perception import LiveClearance, free_destination

    fz = levels.floor_z(2)
    r = MujocoRobot(SCENE, start_xy=(20.0, 9.2), start_yaw=math.pi, start_z=fz)
    sensor, live = L.Lidar(r.model, r.data), LiveClearance(2)

    def tick():
        live.update(sensor.scan((20.0, 9.2), fz), fz, (20.0, 9.2),
                    origin=(20.0, 9.2, fz + L.MOUNT_H))
        free_destination((16.0, 9.4), (20.0, 9.2), live, probe=(14.0, 9.5))

    for _ in range(5):
        tick()
    t = time.perf_counter()
    for _ in range(30):
        tick()
    per = (time.perf_counter() - t) / 30 * 1000

    dream = Dream(live, r.MAX_V)
    pose = (20.0, 9.2, math.pi)
    cands = [[(20 - i * 0.5, 9.2 + dy * i / 8) for i in range(9)]
             for dy in (0, .08, -.08, .16, -.16)]
    for p in cands:
        dream.factor(p, pose)
    t = time.perf_counter()
    for _ in range(20):
        for p in cands:
            dream.factor(p, pose)
    dms = (time.perf_counter() - t) / 20 * 1000

    budget = 1000.0 / 20                      # one 20 Hz control tick
    check("perception fits the tick", per < budget,
          f"{per:.1f} ms of a {budget:.0f} ms tick ({per / budget * 100:.0f}%)")
    check("dreaming fits the replan", dms < 1000,
          f"{dms:.1f} ms per replan, against a ~1.8 s VLM call")


def crowd():
    """Can it tell a person from a crate, and does it stop for the person?

    The dog is driven down the corridor at its real pace while somebody walks
    towards it, because standing still and watching is not the question --
    whether the two paths meet is, and that depends on both of them.

    Note which case this is. A person who steps across the corridor four
    metres ahead is *not* a conflict and the dog is right not to stop for one:
    crossing 2.1 m at walking pace takes under two seconds, by which time the
    dog has covered 1.4 m and they are against the far wall. The stop is for
    the one coming the other way, and for the one who steps out close. Asking
    for a stop in the first case is asking the dog to halt for everybody in
    the building.

    The false-positive half stays the one that matters most: a layer that
    calls the floor-2 cart a pedestrian stops the dog in an empty corridor.
    """
    from cyberdog.sim.robot.mujoco_robot import MujocoRobot
    from cyberdog.sim.scene import levels, pedestrians
    from cyberdog.sim.sensing import lidar as L
    from cyberdog.sim.sensing.perception import LiveClearance
    from cyberdog.sim.sensing.tracking import Tracker, conflict

    fz = levels.floor_z(2)
    r = MujocoRobot(SCENE, start_xy=(24.0, 9.5), start_yaw=math.pi, start_z=fz)
    sensor, live = L.Lidar(r.model, r.data), LiveClearance(2)
    dt, v = r.CONTROL_DT, 0.8

    def scan(at):
        live.update(sensor.scan(at, fz), fz, at, origin=(at[0], at[1], fz + L.MOUNT_H))
        return live.points

    # 1. Somebody walking up the corridor towards the dog, in the near lane.
    # Clear of the cart (x 16.0-17.2) throughout, so every moving return is
    # the person and every still one is the building.
    track = Tracker(dt)
    seen_moving, stopped_at = 0, None
    for k in range(60):
        t = k * dt
        dog = (26.0 - v * t, 9.3)
        r.reset(dog, math.pi)
        r.set_height(fz)
        r.move_mocaps([("ped_f2_0", (20.5 + 1.2 * t, 8.9, fz))])
        tracks = track.update(scan(dog))
        if track.movers():
            seen_moving += 1
        if stopped_at is None and conflict(dog, (-v, 0.0), tracks):
            stopped_at = round(math.dist(dog, (20.5 + 1.2 * t, 8.9)), 2)
    r.move_mocaps([("ped_f2_0", (0.0, 0.0, pedestrians.PARK_Z))])

    check("person seen moving", seen_moving > 20,
          f"tracked as moving on {seen_moving} of 60 ticks")
    check("stops for them", stopped_at is not None and stopped_at > 1.5,
          f"decided to wait {stopped_at} m short of them" if stopped_at
          else "never declared a conflict -- it would have walked into them")

    # 2. The same corridor with nobody in it, past the cart this time. Every
    # return is a crate, so nothing may be called moving.
    track = Tracker(dt)
    false_movers = 0
    for k in range(60):
        dog = (21.0 - v * k * dt, 9.3)
        r.reset(dog, math.pi)
        r.set_height(fz)
        track.update(scan(dog))
        false_movers += len(track.movers())
    check("crates are not people", false_movers == 0,
          f"{false_movers} ticks called a stationary crate a mover")


def run():
    """The whole stack, headless, on the routes that exercise each floor.

    Scored on ground truth: the collision counter compares the dog's pose to
    obstacles.boxes(), which the robot is never shown.
    """
    import subprocess
    py = sys.executable
    # (destination, expected outcome, extra args). The outcome is what this
    # build is known to do, not what it ought to do: "collides" is a documented
    # limit with a README entry, scored so that it shows up here the day it
    # changes in either direction.
    cases = [("restroom", "arrives", []), ("cafeteria", "arrives", []),
             ("room 201", "arrives", []),
             # The floor-1 route, past the trolley. It used to stop short, back
             # when the trolley was 1.45 m deep and getting past it was a
             # full-corridor crossing; at a real trolley's 0.7 m it is an
             # obstacle to be stepped around, which is what this scores.
             ("room 101", "arrives", []),
             # The floor-2 route whose door is a metre west of the crate --
             # the one that needs the crate's open side to be the north one.
             ("electrical engineering lab", "arrives", []),
             # Floor 3, both sides of the cartons: a north room and a south
             # one. The suite ran six routes for a long time and none of them
             # came at a box from the side its slack was not on, which is how
             # three configurations of this scene passed while colliding.
             ("biology lab", "arrives", []),
             # `chemistry lab` used to be that collision -- 0.4 s inside the
             # cartons, every tick of it while the goal pixel was off-frame and
             # avoidance was switched off. Then it stopped short of its door,
             # which was the honest answer while the detour could not hold a
             # line: the door is 1.2 m west of the cartons, so getting in means
             # passing them on the north and turning south immediately. What
             # was actually stopping it was the side of the detour being
             # re-decided every tick (see free_destination's `prefer`); with
             # the crossing committed to one side it arrives, 0.32 m clear.
             ("chemistry lab", "arrives", []),
             # And once more through a corridor with people in it, who are on
             # no map either and who have to be waited for rather than dodged.
             ("room 201", "arrives", ["--pedestrians", "3", "--seed", "1"])]
    for dest, expect, extra in cases:
        out = subprocess.run(
            [py, "-m", "cyberdog.sim.run_building", dest,
             "--auto-confirm", "--no-video"] + extra,
            capture_output=True, text=True).stdout
        arrived = "ARRIVED" in out
        hit = "COLLISIONS:" in out or "CONTACT:" in out
        got = "collides" if hit else "arrives" if arrived else "stops short"
        label = dest + (" (with people)" if extra else "")
        if expect != "arrives":
            label += f" (known limit: {expect})"
        check(f"route: {label}", got == expect,
              f"{got}" + ("" if got == expect else f", expected {expect}"))


STAGES = {"scene": scene, "lidar": lidar, "perception": perception,
          "destination": destination, "crowd": crowd, "latency": latency, "run": run}


if __name__ == "__main__":
    want = sys.argv[1:] or list(STAGES)
    unknown = [w for w in want if w not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; pick from {list(STAGES)}")
    if not os.path.exists(SCENE):
        raise SystemExit("no scene yet -- run: "
                         "python -m cyberdog.sim.scene.build_scene --building")
    for name in want:
        print(f"\n{name}")
        STAGES[name]()
    print(f"\n{'ALL PASS' if not _fails else 'FAILED: ' + ', '.join(_fails)}")
    sys.exit(1 if _fails else 0)

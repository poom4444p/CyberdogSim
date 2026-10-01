"""The Smart Handle and the safety mux: the person's say over the dog.

Spec L6 s1 and s4. Everything else in the stack decides what the dog does;
this is where the person holding it can overrule all of it. Three things, read
from the force along the handle (newtons, + pushing forward, - pulling back):

- a **tug** (a hard pull back) stops the dog dead and it stays stopped until
  the person presses **continue**. A tug means something is wrong that the dog
  did not see, so the dog never decides by itself that it is over;
- a **pull** lowers the pace the dog may walk at, for as long as it is held;
- a **push** raises that pace back, never above the normal walking pace. The
  handle can make the dog slower or stop it; it can never make it faster or
  less careful than the safety layer already allows.

The mux is the last thing between the controller and `set_velocity`: every
stop and slowdown upstream (LiDAR, people, the nose guard, easing for the
person) has already been applied to the command it gets. With nothing on the
handle it returns that command unchanged, to the bit, so a run without a
handle commands exactly what it did before the handle existed.

The real handle is an ESP32 with a load cell (spec L6 s1, `/handle/force` at
50 Hz). Until there is one, `HandleScript` plays timed events, so a run with a
tug in it repeats exactly. The thresholds below are guesses to be calibrated
on that load cell, not measurements.
"""

TUG_N = 40.0          # newtons back: at or past this, a tug -- stop and latch
PULL_N = 8.0          # newtons back: past this (short of a tug), a pull
PUSH_N = 8.0          # newtons forward: past this, a push
PACE_MIN = 0.30       # the lowest a pull takes the pace, as a fraction of the
                      # dog's top speed -- CRAWL, the pace it looks for a way at
PACE_RATE = 0.5       # fraction of top speed per second a pull or push moves it

# What the scripted events put on the handle.
SCRIPT_TUG_N = -60.0  # a tug is a jerk, not a hold:
SCRIPT_TUG_S = 0.2    # this long, then the hand relaxes
SCRIPT_PULL_N = -15.0
SCRIPT_PUSH_N = 15.0


class SafetyMux:
    """The handle's override, applied to the command the controller chose.

    `update` once a tick with what the handle reads, then `apply` to the
    command. `stopped` is latched by a tug and released only by continue.
    """

    def __init__(self):
        self.stopped = False
        self.pace = 1.0           # the ceiling, as a fraction of top speed
        self.stats = {"tugs": 0, "continues": 0, "stopped_ticks": 0,
                      "slowed_ticks": 0, "lowest_pace": 1.0}

    def update(self, force, pressed, dt):
        """Read one tick of the handle. Returns "tug", "continue" or None."""
        event = None
        if force <= -TUG_N:
            if not self.stopped:
                self.stopped = True
                self.stats["tugs"] += 1
                event = "tug"
        elif pressed and self.stopped:
            # Not while they are still pulling hard: that is the same tug.
            self.stopped = False
            self.stats["continues"] += 1
            event = "continue"
        elif force <= -PULL_N:
            self.pace = max(PACE_MIN, self.pace - PACE_RATE * dt)
        elif force >= PUSH_N:
            self.pace = min(1.0, self.pace + PACE_RATE * dt)
        self.stats["lowest_pace"] = min(self.stats["lowest_pace"], self.pace)
        return event

    def apply(self, cmd, top_speed):
        """The command that goes to the motors.

        Stopped is everything at zero, backing off included: the person said
        stop, and reversing into them is not stopping. Under a lowered pace
        only forward speed is capped; turning keeps the dog on its line and
        backing away from something stays the safety layer's call.
        """
        if self.stopped:
            self.stats["stopped_ticks"] += 1
            return (0.0, 0.0, 0.0)
        limit = self.pace * top_speed
        if cmd[0] > limit:
            self.stats["slowed_ticks"] += 1
            return (limit, cmd[1], cmd[2])
        return cmd


class HandleScript:
    """Timed handle events, for runs that have to repeat.

    `spec` is comma-separated events at seconds of sim time, the same `t=`
    the run prints:

        tug@30           a tug at t = 30 s
        continue@35      continue pressed at t = 35 s
        pull@20-25       pulled back from 20 s to 25 s
        push@26-28       pushed forward from 26 s to 28 s

    e.g. ``--handle "tug@30,continue@35"``.
    """

    def __init__(self, spec=""):
        self.holds = []           # (start, end, newtons)
        self.presses = []         # sim times continue is pressed
        for part in filter(None, (p.strip() for p in spec.split(","))):
            name, _, when = part.partition("@")
            try:
                if name == "tug":
                    t = float(when)
                    self.holds.append((t, t + SCRIPT_TUG_S, SCRIPT_TUG_N))
                elif name == "continue":
                    self.presses.append(float(when))
                elif name in ("pull", "push"):
                    a, _, b = when.partition("-")
                    a, b = float(a), float(b)
                    if b <= a:
                        raise ValueError
                    self.holds.append((a, b, SCRIPT_PULL_N if name == "pull"
                                       else SCRIPT_PUSH_N))
                else:
                    raise ValueError
            except ValueError:
                raise ValueError(f"handle event {part!r}: expected tug@T, "
                                 f"continue@T, pull@T1-T2 or push@T1-T2") from None
        self.presses.sort()

    def force(self, t):
        """Newtons along the handle at sim time `t`; the strongest pull wins."""
        now = [n for a, b, n in self.holds if a <= t < b]
        pulls = [n for n in now if n < 0]
        return min(pulls) if pulls else max(now, default=0.0)

    def pressed(self, t, dt):
        """Was continue pressed during the tick that ends at `t`?"""
        return any(t - dt < p <= t for p in self.presses)

    def will_continue(self, t):
        """Is a continue still to come? If not, a stopped dog stays stopped."""
        return any(p > t for p in self.presses)

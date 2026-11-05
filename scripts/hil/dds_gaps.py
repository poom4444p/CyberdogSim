"""How regularly rt/lowstate (from the world) and rt/lowcmd (from board_main)
arrive, on one interface: rate, median, p99 and largest gap. Runs until
SECONDS have passed or it is stopped (SIGTERM/SIGINT), then prints.

  python dds_gaps.py SECONDS IFACE        (onboard: l4tbr0; the Mac: en9)

A separate process with its own socket: it shows what is on the wire, not
how fresh the states board_main itself uses are. Those were 290 ms old on
the J4012 while this showed every LowState on time (go2_board sim_bus.py).
"""
import signal
import sys
import time

import numpy as np

if sys.platform == "darwin":
    try:
        from go2_board import _macos_timerfd  # noqa: F401  (before unitree_sdk2py)
    except ImportError:
        pass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_  # noqa: E402

seconds, iface = float(sys.argv[1]), sys.argv[2]
ChannelFactoryInitialize(0, iface)
T = {"lowstate (world)": [], "lowcmd (board)": []}
a = ChannelSubscriber("rt/lowstate", LowState_)
a.Init(lambda m: T["lowstate (world)"].append(time.monotonic()), 10)
b = ChannelSubscriber("rt/lowcmd", LowCmd_)
b.Init(lambda m: T["lowcmd (board)"].append(time.monotonic()), 10)

t0 = time.monotonic()
done = []
signal.signal(signal.SIGTERM, lambda *_: done.append(1))
try:
    while not done and time.monotonic() - t0 < seconds:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
for k, ts in T.items():
    g = np.diff(ts) * 1e3
    if len(g) == 0:
        print(f"{k:18s} nothing")
        continue
    rate = len(g) / (ts[-1] - ts[0])          # first to last: not the time before it started
    print(f"{k:18s} {rate:4.0f}/s  median {np.median(g):4.1f}  p99 {np.percentile(g, 99):5.1f}"
          f"  max {g.max():6.1f} ms  >20ms {(g > 20).sum():3d}  >50ms {(g > 50).sum():3d}", flush=True)

"""On a real-time robot (go2_board) the VAMOS call runs on a worker thread:
the control loop goes on ticking while it runs, and the answer lands when done."""
import threading
import types

import numpy as np

from cyberdog.sim import run_building as rb


class SlowVamos:
    def __init__(self):
        self.go = threading.Event()
        self.started = threading.Event()

    def request(self, image, prompt, cam_pose):
        self.started.set()
        self.go.wait(2.0)
        return [[(1.0, 2.0), (3.0, 4.0)]]

    def deliver(self, paths, pose, destination):
        return paths[0], paths, 0.9


class Robot:
    CONTROL_DT = 0.05
    REAL_TIME = True

    def get_image(self):
        return np.zeros((4, 4, 3), np.uint8)

    def camera_pose(self):
        return (0.0, 0.0, 0.0)

    def get_pose(self):
        return (0.0, 0.0, 0.0)


def stub():
    vamos = SlowVamos()
    run = types.SimpleNamespace(robot=Robot(), real_time=True, vlm_pool=None, vlm_worker=None,
                                pending=None, vlm_latency=None, policy=lambda floor: vamos,
                                vlm={"asked": 0, "answers": 0, "dropped": 0, "age": 0.0,
                                     "moved": 0.0})
    for name in ("ask_vamos", "take_answer", "vlm_busy"):
        setattr(run, name, getattr(rb.Run, name).__get__(run))
    return run, vamos


def test_the_loop_is_not_held_while_vamos_thinks():
    run, vamos = stub()
    run.ask_vamos(1, 0, {"state": "TRACK", "loc": (500, 600), "destination": (5.0, 5.0)})
    assert vamos.started.wait(1.0)
    for n in range(1, 5):                       # ticks go on: no answer yet, not blocked
        assert run.take_answer(1, n, (0, 0, 0), (5, 5)) is None
    assert run.vlm_busy()
    vamos.go.set()
    run.vlm_worker.result(timeout=1.0)
    chosen, _, safety = run.take_answer(1, 9, (0, 0, 0), (5, 5))
    assert chosen == [(1.0, 2.0), (3.0, 4.0)] and safety == 0.9
    assert run.pending is None and not run.vlm_busy() and run.vlm["answers"] == 1
    run.vlm_pool.shutdown()


def test_a_dropped_answer_still_holds_the_worker():
    run, vamos = stub()
    run.ask_vamos(1, 0, {"state": "TRACK", "loc": (500, 600), "destination": (5.0, 5.0)})
    assert vamos.started.wait(1.0)
    run.pending = None                          # the leg ended: answer dropped
    assert run.vlm_busy()                       # but the GPU is still on it
    vamos.go.set()
    run.vlm_worker.result(timeout=1.0)
    assert not run.vlm_busy()
    run.vlm_pool.shutdown()

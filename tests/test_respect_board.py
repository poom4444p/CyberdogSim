"""Run.respect_board: the brain works with the board's LiDAR guard, never
backs into the person on the handle, and asks for help when it cannot pass."""
import math
import types

from cyberdog.sim import run_building as rb


class Robot:
    def __init__(self):
        self.t, self.free = 0.0, None

    def board_clearance(self):
        return self.free

    def now(self):
        return self.t


def stub(free):
    robot = Robot()
    robot.free = free
    said = []
    run = types.SimpleNamespace(robot=robot, board_turning=None, board_since=0.0, board_push=None, board_flipped=False,
                                board_side=None, board_against=0, board_gave_up=False, gave_up=None, obs={},
                                board_swept=0.0, board_yaw=None, board_leg=0.0, board_plan_to=None,
                                say=said.append)
    run.respect_board = rb.Run.respect_board.__get__(run)
    return run, said


def open_(front=5.0, left=1.0, right=1.0):
    return {"front": front, "back": math.inf, "left": left, "right": right}


def test_never_backwards_with_a_person_on_the_handle():
    run, _ = stub(open_())
    assert run.respect_board((-0.3, 0.1, 0.2), (0, 0, 0)) == (0.0, 0.1, 0.2)


def test_open_ahead_passes_the_plan_through():
    run, said = stub(open_())
    assert run.respect_board((0.5, 0.0, 0.1), (0, 0, 0)) == (0.5, 0.0, 0.1) and not said


def test_pushed_against_the_guard_steps_to_the_roomier_side_then_carries_on():
    run, said = stub(open_(front=0.1, left=0.2, right=0.9))
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0)) == (0.0, -rb.BOARD_SIDE_V, 0.0)
    assert said and "Stepping right" in said[0]
    run.robot.t = 1.0
    assert run.respect_board((0.4, 0.0, 0.0), (0, -0.1, 0)) == (0.0, -rb.BOARD_SIDE_V, 0.0)  # still closed
    run.robot.free = open_(front=rb.BOARD_CLEAR_M + 0.1)
    assert run.respect_board((0.4, 0.0, 0.0), (0, -0.12, 0)) == (0.4, 0.0, 0.0)
    assert run.board_side is None and run.board_turning is None and len(said) == 1


def test_the_trolley_gap_steps_away_from_the_corner():
    # (23.2, 8.7) on floor 1: the trolley's corner inside the guard's 0.30 m
    # band on the left, the wall 0.45 m off on the right. Turning on the spot
    # never opened it; a step right puts the corner outside the band.
    run, said = stub(open_(front=0.05, left=0.08, right=0.45))
    assert run.respect_board((0.3, 0.0, 0.1), (23.2, 8.7, 0.0)) == (0.0, -rb.BOARD_SIDE_V, 0.0)
    run.robot.t, run.robot.free = 1.2, open_(front=3.0, left=0.22, right=0.31)
    assert run.respect_board((0.3, 0.0, 0.1), (23.2, 8.56, 0.0)) == (0.3, 0.0, 0.1)
    assert not run.board_gave_up and run.board_turning is None


def test_sidestep_stops_after_its_distance_and_turns():
    run, said = stub(open_(front=0.1, left=1.0, right=0.4))
    run.respect_board((0.4, 0.0, 0.0), (0, 0, 0))                                 # step left
    run.robot.t = 1.0
    out = run.respect_board((0.4, 0.0, 0.0), (0, rb.BOARD_SIDE_M + 0.01, 0))
    assert out == (0.0, 0.0, rb.BOARD_TURN_W) and "Turning left" in said[-1]


def test_sidestep_stops_after_its_time_and_turns():
    run, said = stub(open_(front=0.1, left=0.4, right=1.0))
    run.respect_board((0.4, 0.0, 0.0), (0, 0, 0))                                 # step right
    run.robot.t = rb.BOARD_SIDE_S + 0.1                                           # went nowhere
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0)) == (0.0, 0.0, -rb.BOARD_TURN_W)


def test_sidestep_stops_when_that_side_closes_in():
    run, said = stub(open_(front=0.1, left=0.2, right=0.5))
    run.respect_board((0.4, 0.0, 0.0), (0, 0, 0))                                 # step right
    run.robot.t, run.robot.free = 0.5, open_(front=0.1, left=0.32, right=rb.BOARD_SIDE_ROOM)
    assert run.respect_board((0.4, 0.0, 0.0), (0, -0.06, 0)) == (0.0, 0.0, -rb.BOARD_TURN_W)


def test_a_turning_plan_is_turned_its_way_not_the_roomier_side():
    # Facing the trolley at 30 deg: the plan asks right (round it), the board
    # sees more room on the left. Stepping left took the nose closer.
    run, said = stub(open_(front=0.15, left=2.2, right=0.0))
    assert run.respect_board((0.7, 0.0, -1.0), (23.13, 8.56, 0.52)) == (0.0, 0.0, -rb.BOARD_TURN_W)
    assert "Turning right" in said[0] and run.board_side is None


def test_a_plan_going_straight_still_steps_to_the_roomier_side():
    run, said = stub(open_(front=0.15, left=2.2, right=0.0))
    assert run.respect_board((0.7, 0.0, rb.BOARD_PLAN_W / 2), (0, 0, 0)) == (0.0, rb.BOARD_SIDE_V, 0.0)


def test_after_a_sidestep_the_turn_is_the_plans_way():
    run, said = stub(open_(front=0.1, left=1.0, right=0.4))
    run.respect_board((0.4, 0.0, 0.0), (0, 0, 0))                                 # straight: step left
    run.robot.t = rb.BOARD_SIDE_S + 0.1
    assert run.respect_board((0.4, 0.0, -0.8), (0, 0, 0)) == (0.0, 0.0, -rb.BOARD_TURN_W)


def test_the_turn_follows_the_plan_when_it_asks_the_other_way():
    # The J4012 run stuck at (23.2, 8.6), from its trace: both sides tight, the
    # plan barely turning, so the turn went right (the roomier side) -- then the
    # plan asked left for 4 s while the dog turned on into the wall.
    run, said = stub(open_(front=0.32, left=0.14, right=0.22))
    assert run.respect_board((0.56, 0.0, 0.06), (23.24, 8.75, -0.51))[2] == -rb.BOARD_TURN_W
    run.robot.free = open_(front=0.25, left=0.15, right=0.20)
    for k in range(1, rb.BOARD_PLAN_TICKS):
        run.robot.t = 0.05 * k
        assert run.respect_board((0.49, 0.0, 0.99), (23.25, 8.73, -0.57))[2] == -rb.BOARD_TURN_W
    run.robot.t = 0.05 * rb.BOARD_PLAN_TICKS
    assert run.respect_board((0.49, 0.0, 0.99), (23.25, 8.73, -0.57)) == (0.0, 0.0, rb.BOARD_TURN_W)
    assert "Turning left, the way on" in said[-1]


def test_one_tick_of_the_plan_the_other_way_does_not_turn_it():
    run, said = stub(open_(front=0.1, left=0.2, right=0.3))
    run.respect_board((0.4, 0.0, -0.8), (0, 0, 0))                                # right, the plan's way
    run.robot.t = 0.05
    assert run.respect_board((0.4, 0.0, 0.9), (0, 0, 0))[2] == -rb.BOARD_TURN_W   # one tick left
    run.robot.t = 0.10
    assert run.respect_board((0.4, 0.0, -0.7), (0, 0, 0))[2] == -rb.BOARD_TURN_W  # and back
    assert run.board_against == 0


def test_no_half_time_flip_against_a_plan_still_asking_this_way():
    run, said = stub(open_(front=0.1, left=0.2, right=0.3))
    run.respect_board((0.4, 0.0, 0.8), (0, 0, 0))                                 # left, the plan's way
    run.robot.t = rb.BOARD_GIVE_UP_S / 2 + 0.1
    assert run.respect_board((0.4, 0.0, 0.8), (0, 0, 0))[2] == rb.BOARD_TURN_W    # still left
    assert not any("other way" in s for s in said)


def test_a_plan_pausing_its_turn_does_not_flip_it():
    # Floor 3 lift lobby, J4012 trace (2026-11-05): turning left the plan's way
    # at 0.175 rad/s, 61 deg in, the plan stepped aside for somebody for a tick
    # (w 0, no way) -- and the half-time rule turned the dog back to the right.
    run, said = stub(open_(front=0.44, left=1.7, right=1.6))
    yaw, k = -1.84, 0
    assert run.respect_board((0.3, 0.0, 2.63), (43.6, 9.0, yaw)) == (0.0, 0.0, rb.BOARD_TURN_W)
    while run.robot.t < 6.0:
        k += 1
        run.robot.t, yaw = 0.05 * k, yaw + 0.05 * 0.175
        assert run.respect_board((0.0, 0.0, 2.0), (43.6, 9.0, yaw))[2] == rb.BOARD_TURN_W
    run.robot.t += 0.05
    assert run.respect_board((0.0, -0.2, 0.0), (43.6, 9.0, yaw)) == (0.0, 0.0, rb.BOARD_TURN_W)
    assert not any("other way" in s for s in said)


def sweep(run, yaw, plan_w, ticks=800, rate=0.44):
    """Drive respect_board at 20 Hz, the dog turning at `rate` of what it is
    asked (go2_flat_v5: 44 % of BOARD_TURN_W); `plan_w(yaw)` is the plan.
    Returns the turn's direction each tick until it gives up."""
    ways, out = [], (0.0, 0.0, 0.0)
    for k in range(1, ticks):
        run.robot.t = 0.05 * k
        yaw += 0.05 * rate * out[2]
        out = run.respect_board((0.55, 0.0, plan_w(yaw)), (23.2, 10.0, yaw))
        if run.board_gave_up:
            break
        ways.append(math.copysign(1, out[2]))
    return ways


def changes(ways):
    return sum(a != b for a, b in zip(ways, ways[1:]))


def test_a_plan_wanting_the_same_heading_does_not_shake_the_turn():
    # The trolley gap, floor 1, J4012 trace (2026-11-05): the plan wants the gap
    # straight ahead, so its w changes sign each time the turn carries the nose
    # past it. Followed every time, the dog shook its head +-30 deg for 10 s.
    run, said = stub(open_(front=0.0, left=0.6, right=0.1))
    run.respect_board((0.55, 0.0, -0.43), (23.2, 10.0, 0.45))       # right, the plan's way
    ways = sweep(run, 0.45, lambda yaw: 2.0 * (0.1 - yaw))
    assert ways[0] < 0 and changes(ways) == 1                        # the sweep's one turn back
    assert not any("the way on" in s for s in said) and run.board_gave_up


def test_a_weak_plan_the_other_way_does_not_undo_the_roomier_side():
    # Live run, same gap: turning left (the roomier side, the plan nearly
    # straight), the plan's -0.40 for the same heading turned it right, into
    # the trolley, where it stuck for 20 s while the plan asked left at 1.3.
    run, said = stub(open_(front=0.0, left=0.58, right=0.09))
    assert run.respect_board((0.55, 0.0, -0.28), (23.12, 9.97, 0.28)) == (0.0, rb.BOARD_SIDE_V, 0.0)
    run.robot.t = rb.BOARD_SIDE_S + 0.1                              # stepped nowhere: turn
    assert run.respect_board((0.55, 0.0, -0.28), (23.12, 9.97, 0.28))[2] == rb.BOARD_TURN_W
    for k in range(1, 20):
        run.robot.t = rb.BOARD_SIDE_S + 0.1 + 0.05 * k
        yaw = 0.28 + 0.05 * k * 0.175
        assert run.respect_board((0.55, 0.0, 2.0 * (0.14 - yaw)), (23.12, 9.97, yaw))[2] == rb.BOARD_TURN_W
    assert not any("the way on" in s for s in said)


def test_sweeps_a_quarter_turn_each_side_then_asks_for_help():
    # 8 s of asking was 80 deg at go2_flat_v5's 44 %, and turning back at half
    # of it only ever came back to where it started: the other side unseen.
    run, said = stub(open_(front=0.1, left=0.3, right=0.2))
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0.0)) == (0.0, 0.0, rb.BOARD_TURN_W)
    ways = sweep(run, 0.0, lambda yaw: 0.0)
    flip = ways.index(-1.0)
    assert math.isclose(flip * 0.05 * 0.44 * rb.BOARD_TURN_W, rb.BOARD_SWEEP_RAD, abs_tol=0.02)
    assert changes(ways) == 1 and run.robot.t < rb.BOARD_GIVE_UP_S and run.robot.t > 8.5
    assert math.isclose(run.board_swept, 3 * rb.BOARD_SWEEP_RAD, abs_tol=0.02)
    assert run.gave_up[0] == "go2_board" and "270 deg" in run.gave_up[1]


def test_no_room_either_side_turns_at_once():
    run, said = stub(open_(front=0.1, left=0.3, right=0.2))
    assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0)) == (0.0, 0.0, rb.BOARD_TURN_W)
    assert "Turning left" in said[0] and run.board_side is None


def test_asks_for_help_when_stepping_and_turning_do_not_open_the_way():
    run, said = stub(open_(front=0.05))
    assert run.respect_board((0.4, 0.0, 0.0), (3.0, 4.0, 0.0))[1] != 0.0         # sideways first
    t0 = rb.BOARD_SIDE_S + 0.1
    run.robot.t = t0
    first = run.respect_board((0.4, 0.0, 0.0), (3.0, 4.0, 0.0))[2]
    assert first != 0.0
    run.robot.t = t0 + rb.BOARD_GIVE_UP_S / 2 + 0.1
    assert run.respect_board((0.4, 0.0, 0.0), (3.0, 4.0, 0.0))[2] == -first   # the other way
    assert "other way" in said[-1]
    run.robot.t = t0 + rb.BOARD_GIVE_UP_S + 0.1
    assert run.respect_board((0.4, 0.0, 0.0), (3.0, 4.0, 0.0)) == (0.0, 0.0, 0.0)
    assert run.board_gave_up and run.gave_up[0] == "go2_board" and "help" in said[-1]


def test_never_steps_or_turns_backwards():
    run, _ = stub(open_(front=0.05, left=0.08, right=0.45))
    for k in range(120):
        run.robot.t = 0.1 * k
        assert run.respect_board((0.4, 0.0, 0.0), (0, 0, 0))[0] == 0.0


def test_blind_board_leaves_the_plan_alone_but_still_never_backwards():
    run, _ = stub(None)
    assert run.respect_board((-0.2, 0.0, 0.3), (0, 0, 0)) == (0.0, 0.0, 0.3)


def test_crawled_to_a_stop_by_the_guard_counts_as_stuck():
    # 0.5 m free: past BOARD_NEAR_M, but the guard gives too little to step at
    run, said = stub(open_(front=0.5, left=1.0, right=0.3))
    for k in range(30):
        run.robot.t = 0.1 * k
        out = run.respect_board((0.3, 0.0, 0.0), (44.6, 8.7, 0.0))      # not moving
    assert out == (0.0, rb.BOARD_SIDE_V, 0.0) and "Stepping left" in said[0]


def test_walking_on_is_not_stuck():
    run, said = stub(open_(front=0.5))
    for k in range(40):
        run.robot.t = 0.1 * k
        out = run.respect_board((0.3, 0.0, 0.0), (44.0 + 0.03 * k, 8.7, 0.0))
    assert out == (0.3, 0.0, 0.0) and not said

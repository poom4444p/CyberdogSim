"""Tests for sim/handle.py: the Smart Handle mock and the safety mux."""

import pytest

from cyberdog.sim.handle import (PACE_MIN, PULL_N, PUSH_N, TUG_N, HandleScript,
                                 SafetyMux)

DT = 0.05          # one control tick at 20 Hz
TOP = 0.8          # the Go2's MAX_V


class TestSafetyMux:
    """The override itself."""

    def test_no_force_passes_the_command_through_unchanged(self):
        mux = SafetyMux()
        for cmd in [(0.8, 0.0, 0.3), (0.0, 0.0, -1.5), (-0.2, 0.0, 0.0), (0.37, 0.0, 0.01)]:
            mux.update(0.0, False, DT)
            assert mux.apply(cmd, TOP) is cmd

    def test_tug_stops_on_the_tick_it_is_felt(self):
        mux = SafetyMux()
        assert mux.update(-TUG_N, False, DT) == "tug"
        assert mux.apply((0.8, 0.0, 0.5), TOP) == (0.0, 0.0, 0.0)

    def test_stopped_means_no_backing_off_either(self):
        mux = SafetyMux()
        mux.update(-TUG_N, False, DT)
        assert mux.apply((-0.2, 0.0, 0.0), TOP) == (0.0, 0.0, 0.0)

    def test_stays_stopped_after_the_tug_is_released(self):
        mux = SafetyMux()
        mux.update(-TUG_N, False, DT)
        for _ in range(100):
            assert mux.update(0.0, False, DT) is None
            assert mux.apply((0.8, 0.0, 0.0), TOP) == (0.0, 0.0, 0.0)

    def test_continue_releases(self):
        mux = SafetyMux()
        mux.update(-TUG_N, False, DT)
        assert mux.update(0.0, True, DT) == "continue"
        assert not mux.stopped
        assert mux.apply((0.8, 0.0, 0.0), TOP) == (0.8, 0.0, 0.0)

    def test_continue_while_still_tugging_is_ignored(self):
        mux = SafetyMux()
        mux.update(-TUG_N, False, DT)
        assert mux.update(-TUG_N * 1.5, True, DT) is None
        assert mux.stopped

    def test_continue_with_nothing_to_release_does_nothing(self):
        mux = SafetyMux()
        assert mux.update(0.0, True, DT) is None
        assert mux.stats["continues"] == 0

    def test_a_held_tug_counts_once(self):
        mux = SafetyMux()
        events = [mux.update(-TUG_N * 1.5, False, DT) for _ in range(4)]
        assert events == ["tug", None, None, None]
        assert mux.stats["tugs"] == 1

    def test_pull_lowers_the_pace_no_further_than_the_floor(self):
        mux = SafetyMux()
        for _ in range(200):
            mux.update(-(PULL_N + 1), False, DT)
        assert mux.pace == pytest.approx(PACE_MIN)
        assert mux.apply((0.8, 0.0, 0.4), TOP) == (pytest.approx(PACE_MIN * TOP), 0.0, 0.4)

    def test_a_light_touch_is_not_a_pull(self):
        mux = SafetyMux()
        for _ in range(100):
            mux.update(-(PULL_N - 1), False, DT)
            mux.update(PUSH_N - 1, False, DT)
        assert mux.pace == 1.0

    def test_push_undoes_a_pull_and_goes_no_higher(self):
        mux = SafetyMux()
        for _ in range(200):
            mux.update(-(PULL_N + 1), False, DT)
        for _ in range(400):
            mux.update(PUSH_N + 1, False, DT)
        assert mux.pace == 1.0
        assert mux.apply((0.8, 0.0, 0.0), TOP) == (0.8, 0.0, 0.0)

    def test_push_never_speeds_up_a_slowed_command(self):
        """The safety layer slowed it; the handle may not undo that."""
        mux = SafetyMux()
        for _ in range(100):
            mux.update(PUSH_N * 3, False, DT)
        assert mux.apply((0.24, 0.0, 0.0), TOP) == (0.24, 0.0, 0.0)

    def test_lower_pace_leaves_turning_and_reversing_alone(self):
        mux = SafetyMux()
        for _ in range(200):
            mux.update(-(PULL_N + 1), False, DT)
        assert mux.apply((0.0, 0.0, 1.2), TOP) == (0.0, 0.0, 1.2)
        assert mux.apply((-0.2, 0.0, 0.0), TOP) == (-0.2, 0.0, 0.0)


class TestHandleScript:
    """Timed events, for runs that have to repeat."""

    def test_empty_is_no_force_and_no_presses(self):
        h = HandleScript("")
        assert h.force(10.0) == 0.0
        assert not h.pressed(10.0, DT)
        assert not h.will_continue(0.0)

    def test_tug_is_a_short_jerk(self):
        h = HandleScript("tug@30")
        assert h.force(29.99) == 0.0
        assert h.force(30.0) <= -TUG_N
        assert h.force(31.0) == 0.0

    def test_pull_and_push_hold_for_their_span(self):
        h = HandleScript("pull@20-25,push@26-28")
        assert -TUG_N < h.force(22.0) <= -PULL_N
        assert h.force(25.5) == 0.0
        assert h.force(27.0) >= PUSH_N

    def test_a_pull_wins_over_a_push_at_the_same_time(self):
        h = HandleScript("push@10-20,tug@15")
        assert h.force(15.0) <= -TUG_N

    def test_continue_lands_on_one_tick(self):
        h = HandleScript("continue@35")
        ticks = [round(35.0 + k * DT - 0.2, 2) for k in range(9)]
        assert sum(h.pressed(t, DT) for t in ticks) == 1
        assert h.will_continue(34.9) and not h.will_continue(35.0)

    @pytest.mark.parametrize("bad", ["shove@3", "tug", "tug@x", "pull@5", "pull@9-4",
                                     "continue@"])
    def test_bad_events_are_refused(self, bad):
        with pytest.raises(ValueError, match="handle event"):
            HandleScript(bad)

    def test_script_through_the_mux(self):
        """tug@1, continue@2: stopped from the tug's tick to the continue's."""
        h, mux = HandleScript("tug@1,continue@2"), SafetyMux()
        moving = []
        for k in range(60):
            t = round(k * DT, 2)
            mux.update(h.force(t), h.pressed(t, DT), DT)
            moving.append(mux.apply((0.8, 0.0, 0.0), TOP)[0] > 0)
        assert all(moving[:20]) and not any(moving[20:40]) and all(moving[40:])

"""hold_side: a committed detour side survives one tick of scan noise."""
from cyberdog.sim import run_building as rb

LEFT, RIGHT = ((23.4, 10.3), 1.0), ((23.4, 8.5), -0.8)


def step(state, offer):
    lean, held, against = state
    aim, off, lean, held, against = rb.hold_side(lean, *offer, held, against)
    return (aim, off), (lean, held, against)


def test_first_offer_commits():
    got, state = step((0.0, None, 0), LEFT)
    assert got == LEFT and state == (1.0, LEFT, 0)


def test_one_tick_of_the_other_side_keeps_the_old_aim():
    # Floor-1 trolley, HIL 15:36: left empty for one tick, right offered.
    _, state = step((0.0, None, 0), LEFT)
    got, state = step(state, RIGHT)
    assert got == LEFT and state[0] == 1.0
    got, state = step(state, LEFT)          # the left is back
    assert got == LEFT and state == (1.0, LEFT, 0)


def test_the_other_side_is_believed_after_enough_ticks():
    _, state = step((0.0, None, 0), LEFT)
    for _ in range(rb.SIDE_FLIP_TICKS - 1):
        got, state = step(state, RIGHT)
        assert got == LEFT
    got, state = step(state, RIGHT)
    assert got == RIGHT and state == (-1.0, RIGHT, 0)


def test_straight_on_or_no_way_changes_nothing():
    _, state = step((0.0, None, 0), LEFT)
    for offer in (((24.0, 9.0), 0.0), (None, 0.0)):
        got, after = step(state, offer)
        assert got == offer and after == state

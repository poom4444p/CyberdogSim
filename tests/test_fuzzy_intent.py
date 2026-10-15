"""Tests for the PoC intent parser's fuzzy match -- run_building.fuzzy_place.

Spec L3 step 1: "lookup/fuzzy match of destination against Behavior Layer node
list". Speech-to-text gets names nearly right, so a near miss is matched; it is
never taken on trust (main asks before the dog moves), and anything doubtful --
a different room number, two names about as close -- is refused, not guessed.
"""
import pytest

from cyberdog.planning.building_router import BuildingRouter
from cyberdog.sim.run_building import fuzzy_place, resolve_stops


@pytest.fixture(scope="module")
def router():
    return BuildingRouter()


@pytest.mark.parametrize("said, place", [
    ("take me to the libary", "library"),
    ("go to the chemistry lap", "chemistry lab"),
    ("I want the rest room", "restroom"),
    ("cafe teria please", "cafeteria"),
    ("take me to the physic lab", "physics lab"),
])
def test_near_misses_are_matched(router, said, place):
    assert fuzzy_place(router.locations, said)[0] == place


@pytest.mark.parametrize("said", [
    "take me to room 211",          # no such room; 201 and 210 are not typos of it
    "take me to the moon",
    "take me to the chemistry",     # half a name is not a name
    "I need to pee",                # no name at all: that is --nlu's job
])
def test_doubtful_is_refused(router, said):
    assert fuzzy_place(router.locations, said) is None


def test_a_room_number_is_never_bent():
    assert fuzzy_place(["room 201", "room 210"], "take me to room 210")[0] == "room 210"
    assert fuzzy_place(["room 201"], "take me to room 210") is None


def test_two_names_about_as_close_is_ambiguous():
    assert fuzzy_place(["physics lab", "physic lap"], "physics lap") is None


def test_exact_name_is_not_a_guess(router):
    guessed = []
    assert resolve_stops(router, "take me to the library", guessed=guessed) == [("library", None)]
    assert guessed == []


def test_fuzzy_match_is_flagged_for_confirmation(router):
    guessed = []
    stops = resolve_stops(router, "take me to the libary, then the cafe teria", guessed=guessed)
    assert stops == [("library", None), ("cafeteria", None)]
    assert guessed == [(0, "library", "libary"), (1, "cafeteria", "cafe teria")]


def test_nothing_close_is_still_a_refusal(router):
    with pytest.raises(SystemExit):
        resolve_stops(router, "take me to the moon")

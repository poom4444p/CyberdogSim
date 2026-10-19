"""Tests for the PoC intent parser: language/place_match and resolve_stops.

Spec L3 step 1: "lookup/fuzzy match of destination against Behavior Layer node
list". Speech-to-text gets names nearly right, so a near miss is matched; it is
never taken on trust (main asks before the dog moves), and anything doubtful --
a different room number, two names about as close -- is refused, not guessed.
"""
import difflib
import sys
import types

import pytest

from cyberdog.language.place_match import (ALIASES, alias_place, fuzzy_place, near_place,
                                           numbers_to_digits, sound)
from cyberdog.planning.building_router import BuildingRouter
from cyberdog.sim.run_building import resolve_stops


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
    # "cafe" is a nickname of the cafeteria, found before the near miss.
    assert guessed == [(0, "library", "libary"), (1, "cafeteria", "cafe")]


def test_nothing_close_is_still_a_refusal(router):
    with pytest.raises(SystemExit):
        resolve_stops(router, "take me to the moon")


# --- number words -------------------------------------------------------------

@pytest.mark.parametrize("said, digits", [
    ("take me to room two oh one", "take me to room 201"),
    ("room two zero one", "room 201"),
    ("room two hundred and five", "room 205"),
    ("room two ten", "room 210"),
    ("room three-oh-nine", "room 309"),
    ("go to room 201", "go to room 201"),
    ("oh take me to the library", "oh take me to the library"),   # a lone "oh" is a word
])
def test_number_words_become_digits(said, digits):
    assert numbers_to_digits(said) == digits


def test_a_room_said_in_words_is_found_and_confirmed(router):
    guessed = []
    assert resolve_stops(router, "take me to room two oh one", guessed=guessed) == [("room 201", None)]
    assert [(k, name, heard.lower()) for k, name, heard in guessed] == [
        (0, "room 201", "take me to room two oh one")]


# --- sound: vowels and spellings speech-to-text gets wrong ----------------------

def test_sound_folds_vowels_and_spellings():
    assert sound("fisics") == sound("physics")
    # Not always equal, but nearer by sound than by spelling: what lets the
    # match clear FUZZY_MIN.
    for said, name in [("libery", "library"), ("offis", "office"), ("servar", "server")]:
        assert (difflib.SequenceMatcher(None, sound(said), sound(name)).ratio()
                > difflib.SequenceMatcher(None, said, name).ratio())


@pytest.mark.parametrize("said, place", [
    ("take me to the libery", "library"),
    ("go to the cafitiria", "cafeteria"),
    ("go to the fisics lab", "physics lab"),
    ("go to the physix lap", "physics lab"),
    ("take me to the offis", "office"),
    ("take me to the rest rum", "restroom"),
    ("go to ruum 201", "room 201"),
    ("go to the servar rum", "server room"),
])
def test_vowel_mistakes_are_matched_by_sound(router, said, place):
    assert fuzzy_place(router.locations, said) is None      # spelling alone misses
    assert near_place(router.locations, said, hazards(router))[0] == place


def test_what_was_heard_is_reported_as_said(router):
    assert near_place(router.locations, "go to the servar rum", hazards(router)) == (
        "server room", "servar rum")


@pytest.mark.parametrize("said", [
    "turn left",                    # "left" and "lift" sound alike: short names never match by sound
    "take me left",
    "take me to the store",         # sounds like "stairway": no-go places never match by sound
    "I'm lost",
    "take me home",
    "go to the lab",
    "go to room 211",
    "I need a rest",
    "go to the roof",
])
def test_sound_does_not_invent_places(router, said):
    assert near_place(router.locations, numbers_to_digits(said), hazards(router)) is None


def hazards(router):
    return {n for n in router.locations if router.hazard_place(n)}


# --- --nlu: the model only when nothing is named, and held to the map ----------

@pytest.fixture
def model(monkeypatch):
    """A stand-in for language/infer.py: no torch, and it counts its calls."""
    answers = {"I need to pee right now": "restroom", "I need to pee": "restroom", "take me to the moon": "unknown",
               "I want to borrow a book": "librery", "go somewhere nice": "the beach",
               "garble": ValueError}
    calls = []

    def parse_command(text):
        calls.append(text)
        # The splitter hands over "Go to garble" for "garble": match loosely.
        said = next(v for k, v in answers.items() if k.lower() in text.lower())
        if said is ValueError:
            raise ValueError("Model output did not contain JSON")
        return {"target_location": said}

    fake = types.ModuleType("cyberdog.language.infer")
    fake.load_model = lambda: None
    fake.parse_command = parse_command
    monkeypatch.setitem(sys.modules, "cyberdog.language.infer", fake)
    return calls


def test_nlu_is_not_asked_when_a_place_is_named(router, model):
    guessed = []
    assert resolve_stops(router, "take me to the libary", use_nlu=True,
                         guessed=guessed) == [("library", None)]
    assert model == [] and guessed == [(0, "library", "libary")]


def test_nlu_answer_is_confirmed(router, model):
    guessed = []
    assert resolve_stops(router, "I need to pee", use_nlu=True,
                         guessed=guessed) == [("restroom", None)]
    assert guessed == [(0, "restroom", "I need to pee")]


def test_nlu_near_name_is_held_to_the_map(router, model):
    assert resolve_stops(router, "I want to borrow a book", use_nlu=True) == [("library", None)]


@pytest.mark.parametrize("said", ["take me to the moon", "go somewhere nice", "garble"])
def test_nlu_unknown_or_off_map_or_broken_is_a_refusal(router, model, said):
    with pytest.raises(SystemExit):
        resolve_stops(router, said, use_nlu=True)


def test_one_stop_by_name_one_by_model(router, model):
    guessed = []
    stops = resolve_stops(router, "go to the libary, then I need to pee", use_nlu=True,
                          guessed=guessed)
    assert stops == [("library", None), ("restroom", None)]
    assert [c.lower() for c in model] == ["i need to pee"]
    assert [g[:2] for g in guessed] == [(0, "library"), (1, "restroom")]


# --- movements are not trips ------------------------------------------------------

@pytest.mark.parametrize("said", ["turn left", "take me left", "go back", "stop",
                                  "please turn around", "Go straight ahead", "wait"])
def test_a_movement_is_refused_before_the_model(router, model, said):
    with pytest.raises(SystemExit, match="movement, not a place"):
        resolve_stops(router, said, use_nlu=True)
    assert model == []


@pytest.mark.parametrize("said, place", [
    ("go to the restroom on the left", "restroom"),     # a place named: a trip
    ("I need to pee right now", "restroom"),            # "right" mid-sentence is not a turn
])
def test_a_trip_with_a_direction_word_is_still_a_trip(router, model, said, place):
    assert resolve_stops(router, said, use_nlu=True) == [(place, None)]


# --- the parser stops at the end of its answer ------------------------------------

def test_generation_stops_when_the_json_closes():
    infer = pytest.importorskip("cyberdog.language.infer")
    assert not infer.json_closed('{"target_location": "restroom"')
    assert infer.json_closed('{"target_location": "restroom", "query": null}')
    assert infer.json_closed('{"a": {"b": 1}}\nmodel\n{"a"')
    assert not infer.json_closed('{"a": {"b": 1}')
    assert not infer.json_closed("model")


# --- nicknames: the model's own alias tables, read as rules -----------------------

@pytest.mark.parametrize("said, place, nickname", [
    ("take me to the toilet", "restroom", "toilet"),
    ("where is the bathroom", "restroom", "bathroom"),
    ("go to the canteen", "cafeteria", "canteen"),
    ("take me to the ee lab", "electrical engineering lab", "ee lab"),
    ("go to the IT room", "server room", "it room"),
    ("take me to the front door", "main entrance", "front door"),
])
def test_a_nickname_is_found_without_the_model_and_confirmed(router, model, said, place, nickname):
    guessed = []
    assert resolve_stops(router, said, use_nlu=True, guessed=guessed) == [(place, None)]
    assert model == [] and guessed == [(0, place, nickname)]


def test_nickname_needs_whole_words(router):
    assert alias_place(router.locations, "look over there") is None      # "loo"
    assert alias_place(router.locations, "the main office please") == ("office", "main office")


def test_stairs_have_no_nicknames(router):
    # "steps" would make "take a few steps back" a trip to the stairs.
    assert "steps" not in ALIASES and "stairs" not in ALIASES.values()
    with pytest.raises(SystemExit):
        resolve_stops(router, "take a few steps back")

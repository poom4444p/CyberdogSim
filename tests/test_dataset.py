"""Tests for generate_dataset.py -- the parser's vocabulary, no model needed."""

import random
import re

import pytest

from cyberdog.language.command_splitter import split_destinations
from cyberdog.language.generate_dataset import (
    HELD_OUT_ALIASES, HELD_OUT_NAV_TEMPLATES, HELD_OUT_UNKNOWN_PLACES,
    LAB_ALIASES, OTHER_ALIASES, UNKNOWN_LOCATION, UNKNOWN_PLACES,
    generate_synthetic_dataset,
)


def _has_phrase(text, phrase):
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None


def _template_regex(template):
    # "the {location}" is "room {location}" for numbered rooms.
    body = re.escape(template).replace(re.escape("the {location}"), r"(?:the|room)\ .+")
    return re.compile(body, re.IGNORECASE)


@pytest.fixture(scope="module")
def training():
    random.seed(0)
    return generate_synthetic_dataset(8000)


@pytest.fixture(scope="module")
def held_out():
    random.seed(0)
    return generate_synthetic_dataset(1000, held_out=True)


class TestHeldOutSplit:
    """The held-out vocabulary must never reach training, or the held-out
    score stops measuring generalisation."""

    def test_held_out_aliases_not_in_training_tables(self):
        trained = {a for d in (OTHER_ALIASES, LAB_ALIASES) for v in d.values() for a in v}
        for aliases in HELD_OUT_ALIASES.values():
            for a in aliases:
                assert a not in trained, a

    def test_held_out_aliases_name_real_locations(self):
        for loc in HELD_OUT_ALIASES:
            assert loc in OTHER_ALIASES or loc in LAB_ALIASES, loc

    def test_held_out_unknown_places_not_in_training_list(self):
        assert not set(HELD_OUT_UNKNOWN_PLACES) & set(UNKNOWN_PLACES)

    def test_no_held_out_alias_or_place_in_training_text(self, training):
        held = [a for v in HELD_OUT_ALIASES.values() for a in v] + HELD_OUT_UNKNOWN_PLACES
        for s in training:
            for phrase in held:
                assert not _has_phrase(s["text"], phrase), (phrase, s["text"])

    def test_no_held_out_template_in_training_text(self, training):
        patterns = [_template_regex(t) for t in HELD_OUT_NAV_TEMPLATES]
        for s in training:
            for p in patterns:
                assert not p.fullmatch(s["text"]), s["text"]

    def test_every_held_out_sample_is_new(self, held_out):
        patterns = [_template_regex(t) for t in HELD_OUT_NAV_TEMPLATES]
        held = [a for v in HELD_OUT_ALIASES.values() for a in v] + HELD_OUT_UNKNOWN_PLACES
        for s in held_out:
            new_phrasing = any(p.fullmatch(s["text"]) for p in patterns)
            new_name = any(_has_phrase(s["text"], a) for a in held)
            assert new_phrasing or new_name, s["text"]


class TestUnknown:

    def test_training_has_unknown_samples(self, training):
        unknown = [s for s in training if s["location"] == UNKNOWN_LOCATION]
        assert len(unknown) > 0.03 * len(training)

    def test_unknown_places_contain_no_real_alias(self):
        # "post office" would teach the model that "office" is sometimes
        # unknown. A shared generic word ("music room" / "server room") is fine.
        aliases = [a for d in (OTHER_ALIASES, LAB_ALIASES, HELD_OUT_ALIASES)
                   for v in d.values() for a in v]
        for place in UNKNOWN_PLACES + HELD_OUT_UNKNOWN_PLACES:
            for a in aliases:
                assert not _has_phrase(place, a), (place, a)


class TestLift:

    def test_lift_and_elevator_map_to_lift(self, training):
        for word in ("lift", "elevator"):
            labels = {s["location"] for s in training if _has_phrase(s["text"], word)}
            assert labels == {"lift"}, (word, labels)


class TestSplitterPhrasing:
    """New sentence openings must reach the model as written, not as
    "Go to Where's the library?"."""

    @pytest.mark.parametrize("text", [
        "Where's the library?",
        "Show me the way to the caf.",
        "Lead me to room 105.",
        "How do I get to the lift?",
        "Which way to the lavatory?",
        "Could you guide me to the elevator?",
    ])
    def test_single_stop_passes_through(self, text):
        assert split_destinations(text) == [text]

    def test_new_verb_splits_a_second_stop(self):
        assert split_destinations("Bring me to the library and then lead me to the caf") == \
            ["Bring me to the library", "Lead me to the caf"]

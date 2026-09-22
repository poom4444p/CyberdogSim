"""Rule-based floor extraction, run BEFORE the NLU model.

The model was never trained on floor phrases ("I want to pee on the second
floor" -> it outputs "second floor" as the location). Until the model is
retrained with a floor field, this pulls the floor out with regexes and
hands the model the command without it:

    extract_floor("I want to pee in the second floor")
    -> (2, "I want to pee")
    extract_floor("Take me to the restroom upstairs", current_floor=2)
    -> (3, "Take me to the restroom")

Recognized (case-insensitive), with an optional leading "on/in/at/to/of the":
  absolute: "second floor", "2nd floor", "floor 2", "floor two", "level 2",
            "2F", "ground floor" (= 1), "top floor" (= NUM_FLOORS)
  relative: "upstairs", "downstairs", "one floor up", "two floors down",
            "up a floor", "down two levels", "the floor above/below"
Relative phrases need current_floor (the robot's floor from localization);
without it they raise ValueError. The result may be outside 1..NUM_FLOORS
(e.g. "upstairs" on the top floor) -- the caller decides how to report that.

Running on Linux:
    Plain Python 3, no dependencies. Self-test:
        cd input_treating_layer
        python3 floor_parser.py
"""
import re

from generate_dataset import NUM_FLOORS

_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ORDINAL = r"(?P<ord>\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
_NUMBER = r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_PREFIX = r"(?P<pre>(?:\b(?:on|in|at|to|of|from)\s+)?(?:\bthe\s+)?)"

_PATTERNS = [
    # "second floor", "2nd floor", "the 3rd level"
    rf"{_PREFIX}\b{_ORDINAL}\s+(?:floor|level|storey|story)\b",
    # "floor 2", "level two"
    rf"{_PREFIX}\b(?:floor|level)\s+(?:number\s+)?{_NUMBER}\b",
    # "2F" / "2/F"
    rf"{_PREFIX}\b(?P<f>\d+)\s*/?\s*f\b",
    # "ground floor", "top floor"
    rf"{_PREFIX}\b(?P<named>ground|top)\s+(?:floor|level)\b",
]
_COUNT = r"(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
_UNIT = r"(?:floors?|levels?|storeys?|stories)"

# Relative phrases (value is an offset from current_floor). Checked first.
_RELATIVE_PATTERNS = [
    # "upstairs", "downstairs"
    rf"{_PREFIX}\b(?P<dir>upstairs|downstairs)\b",
    # "one floor up", "two floors down", "a floor above"
    rf"{_PREFIX}\b(?:{_COUNT}\s+)?{_UNIT}\s+(?P<dir>up|down|above|below|higher|lower)\b",
    # "up one floor", "down two levels", "up a floor"
    rf"{_PREFIX}\b(?P<dir>up|down)\s+(?:{_COUNT}\s+)?{_UNIT}\b",
]

_REGEXES = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]
_RELATIVE_REGEXES = [re.compile(p, re.IGNORECASE) for p in _RELATIVE_PATTERNS]


def _to_int(match):
    g = match.groupdict()
    if g.get("named"):
        return 1 if g["named"].lower() == "ground" else NUM_FLOORS
    token = (g.get("ord") or g.get("num") or g.get("f")).lower()
    digits = re.match(r"\d+", token)
    return int(digits.group()) if digits else _WORDS[token]


def _offset(match):
    g = match.groupdict()
    n = (g.get("n") or "one").lower()
    count = int(n) if n.isdigit() else 1 if n in ("a", "an") else _WORDS[n]
    up = g["dir"].lower() in ("upstairs", "up", "above", "higher")
    return count if up else -count


def _strip(text, m):
    # "to the 2nd floor library": the phrase modifies the next word, so
    # keep "to the" and drop only the floor words.
    rest = text[m.end():]
    keep = m.group("pre") if re.match(r"\s*[A-Za-z]", rest) and not re.match(
        r"\s*(?:please|now|right|thanks)\b", rest, re.IGNORECASE) else ""
    cleaned = text[:m.start()] + " " + keep + rest
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)   # "pee ." -> "pee."
    cleaned = re.sub(r",+([.!?])", r"\1", cleaned)      # "pee,." -> "pee."
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return re.sub(r"^[,\s]+|[,\s]+$", "", cleaned)


def extract_floor(text, current_floor=None):
    """Return (floor or None, text with the floor phrase removed).

    current_floor: the robot's floor, needed for "upstairs"/"downstairs".
    """
    for rx in _RELATIVE_REGEXES:
        m = rx.search(text)
        if m:
            if current_floor is None:
                raise ValueError(f"'{m.group(0).strip()}' is relative; current_floor is required")
            return current_floor + _offset(m), _strip(text, m)
    for rx in _REGEXES:
        m = rx.search(text)
        if m:
            return _to_int(m), _strip(text, m)
    return None, text


if __name__ == "__main__":
    cases = [
        ("I want to pee in the second floor", 2, "I want to pee"),
        ("Go to the restroom on floor 3.", 3, "Go to the restroom."),
        ("Take me to the 2nd floor library.", 2, "Take me to the library."),
        ("Head to the library on level two", 2, "Head to the library"),
        ("Bathroom, 3F please", 3, "Bathroom, please"),
        ("Go to the restroom on the ground floor.", 1, "Go to the restroom."),
        ("Find the hallway on the top floor", NUM_FLOORS, "Find the hallway"),
        ("Go to room 310.", None, "Go to room 310."),
        ("I need to pee.", None, "I need to pee."),
        ("Check if the lights are on in the library", None, "Check if the lights are on in the library"),
        ("Take me to the stairs.", None, "Take me to the stairs."),
    ]
    # (text, current_floor, expected floor, expected cleaned text)
    relative_cases = [
        ("Take me to the restroom upstairs", 2, 3, "Take me to the restroom"),
        ("I need to pee, downstairs please", 2, 1, "I need to pee, please"),
        ("Go to the library one floor down.", 2, 1, "Go to the library."),
        ("Go to the restroom two floors up", 1, 3, "Go to the restroom"),
        ("I need to pee, two floors up.", 1, 3, "I need to pee."),
        ("Go up a floor to the restroom", 1, 2, "Go to the restroom"),
        ("Find the hallway on the floor above", 1, 2, "Find the hallway"),
        ("Find the hallway on the floor below", 3, 2, "Find the hallway"),
        ("Go to the upstairs restroom", 1, 2, "Go to the restroom"),
        ("Restroom upstairs", 3, 4, "Restroom"),   # out of range: caller reports it
    ]
    failed = 0
    for text, want_floor, want_text in cases:
        floor, cleaned = extract_floor(text)
        ok = floor == want_floor and cleaned == want_text
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {text!r} -> ({floor}, {cleaned!r})")
    for text, current, want_floor, want_text in relative_cases:
        floor, cleaned = extract_floor(text, current_floor=current)
        ok = floor == want_floor and cleaned == want_text
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {text!r} (on floor {current}) -> ({floor}, {cleaned!r})")
    try:
        extract_floor("Restroom upstairs")
        failed += 1
        print("[FAIL] relative phrase without current_floor should raise")
    except ValueError:
        print("[PASS] relative phrase without current_floor raises ValueError")
    print("All good." if not failed else f"{failed} failure(s).")
    raise SystemExit(1 if failed else 0)

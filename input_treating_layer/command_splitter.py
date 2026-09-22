"""Rule-based multi-destination splitting, run BEFORE the NLU model.

The model was trained on one destination per command. This splits a
multi-stop command into single-stop commands the model already handles:

    split_destinations("Go to the library, then the cafeteria and after that room 105")
    -> ["Go to the library", "Go to the cafeteria", "Go to room 105"]

Split points (case-insensitive):
  - "then", "and then", "after that", "afterwards", "followed by", "next", ";"
  - "and <movement verb>"            ("... and go to the cafeteria")
  - "and" / "," / "as well as" / "plus" followed by a PLACE
                                     ("the library and the cafeteria",
                                      "the library, the caf, and room 310",
                                      "room 105 and 310")
A place is any alias the model was trained on (OTHER_ALIASES / LAB_ALIASES in
generate_dataset.py, imported so the vocabularies can't drift) or a room
number. "and check if ..." is NOT a split -- the training set uses it for
questions about the same destination.

Known limits: a question only applies to the stop it's written with
("check if the lights are on in the library and the cafeteria" asks about
the library only); slang joined by "and" ("I need to pee and eat") isn't split.

Running on Linux:
    Plain Python 3, no dependencies. Self-test:
        cd input_treating_layer
        python3 command_splitter.py
"""
import re

from generate_dataset import LAB_ALIASES, OTHER_ALIASES

_ALIASES = sorted({a for d in (OTHER_ALIASES, LAB_ALIASES) for names in d.values() for a in names},
                  key=len, reverse=True)
# Something that names a destination: "the library", "to the caf", "room 310", "310"
_PLACE = (r"(?:to\s+)?(?:the\s+)?(?:" + "|".join(re.escape(a) for a in _ALIASES)
          + r"|rooms?\s+\d+|\d{3})\b")

# Movement verbs: "and <verb>" starts a new destination. ("check" is left out
# on purpose -- "and check if ..." is a question about the same destination.)
_VERBS = r"(?:go|head|navigate|proceed|move|walk|take\s+me|bring\s+me|get\s+me|visit|stop\s+by)"

_SPLIT = re.compile(
    r"\s*(?:[,;]\s*)?\b(?:and\s+then|then|and\s+after\s+that|after\s+that|afterwards|"
    r"followed\s+by|and\s+next|next(?!\s+to\b))\b\s*,?\s*"
    r"|\s*;\s*"
    rf"|\s*,?\s*\band\s+(?={_VERBS}\b)"
    # "..., the cafeteria", "..., and room 310", "... and the caf", "... as well as the lib"
    rf"|\s*,\s*(?:(?:and|plus)\s+(?:also\s+)?)?(?={_PLACE})"
    rf"|\s+(?:and(?:\s+also)?|as\s+well\s+as|plus)\s+(?={_PLACE})",
    re.IGNORECASE,
)
_LEADING_FIRST = re.compile(r"^\s*(?:first(?:ly)?|to\s+start)\s*,?\s*", re.IGNORECASE)
# A segment that already reads like a command (verb, or one of the model's
# slang/need phrases) is passed through as-is; otherwise "Go to" is added.
_COMMAND_START = re.compile(
    rf"^\s*(?:please\s+)?(?:{_VERBS}|check\b|i\b|i'm\b|let's\b|we\b|nature's\b|can\s+you|could\s+you)",
    re.IGNORECASE,
)


def split_destinations(text):
    """Return a list of single-destination commands (length 1 if no split)."""
    parts = [p for p in _SPLIT.split(text) if p and p.strip(" ,.;")]
    out = []
    for part in parts:
        part = _LEADING_FIRST.sub("", part).strip(" ,;")
        if not part:
            continue
        part = re.sub(r"\bboth\s+", "", part, flags=re.IGNORECASE)        # "both the lib"
        part = re.sub(r"\brooms\s+(?=\d)", "room ", part, flags=re.IGNORECASE)  # "rooms 105"
        if not _COMMAND_START.match(part):
            part = re.sub(r"^(?:to|go\s+to)\s+", "", part, flags=re.IGNORECASE)
            if re.match(r"^\d{3}\b", part):                               # "310" -> "room 310"
                part = "room " + part
            part = f"Go to {part}"
        out.append(part[0].upper() + part[1:])
    return out or [text]


if __name__ == "__main__":
    cases = [
        ("Go to room 105.", ["Go to room 105."]),
        ("Go to the library, then the cafeteria.",
         ["Go to the library", "Go to the cafeteria."]),
        ("Take me to the library and then to room 310",
         ["Take me to the library", "Go to room 310"]),
        ("First go to the office, then the restroom, and after that the main entrance.",
         ["Go to the office", "Go to the restroom", "Go to the main entrance."]),
        ("Go to the cafeteria and go to the library", ["Go to the cafeteria", "Go to the library"]),
        ("I need to pee, then take me to the cafeteria",
         ["I need to pee", "Take me to the cafeteria"]),
        ("Go to the restroom upstairs; then the library",
         ["Go to the restroom upstairs", "Go to the library"]),
        # Questions keep their "and"
        ("Go to room 310 and check if the lights are on",
         ["Go to room 310 and check if the lights are on"]),
        ("Go to room 310 and check if the lights are on, then go to the library",
         ["Go to room 310 and check if the lights are on", "Go to the library"]),
        # "next to" is a place description, not a split
        ("Go to the room next to the library", ["Go to the room next to the library"]),
        # Places joined by "and" / commas / "as well as"
        ("Go to the library and the cafeteria", ["Go to the library", "Go to the cafeteria"]),
        ("Take me to the lib and caf", ["Take me to the lib", "Go to caf"]),
        ("Go to the library, the cafeteria, and room 310.",
         ["Go to the library", "Go to the cafeteria", "Go to room 310."]),
        ("Go to the library, the cafeteria and room 310",
         ["Go to the library", "Go to the cafeteria", "Go to room 310"]),
        ("Go to room 105 and 310", ["Go to room 105", "Go to room 310"]),
        ("Take me to rooms 105 and 310", ["Take me to room 105", "Go to room 310"]),
        ("I want to visit both the library and the chem lab",
         ["I want to visit the library", "Go to the chem lab"]),
        ("Go to the office as well as the restroom", ["Go to the office", "Go to the restroom"]),
        ("Go to the ee lab and also the server room", ["Go to the ee lab", "Go to the server room"]),
        ("Go to the restroom upstairs and the library", ["Go to the restroom upstairs", "Go to the library"]),
        # Must NOT split
        ("Check the library, the door is open", ["Check the library, the door is open"]),
        ("I'm starving, let's go eat.", ["I'm starving, let's go eat."]),
        ("I'm hungry, take me somewhere to eat.", ["I'm hungry, take me somewhere to eat."]),
        ("I need to know if anyone is there in the library. Go check.",
         ["I need to know if anyone is there in the library. Go check."]),
    ]
    failed = 0
    for text, want in cases:
        got = split_destinations(text)
        ok = got == want
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {text!r}\n        -> {got}")
    print("All good." if not failed else f"{failed} failure(s).")
    raise SystemExit(1 if failed else 0)

"""Which place a heard command names, when speech-to-text got it nearly right.

Spec L3 step 1: "lookup/fuzzy match of destination against Behavior Layer
node list". Speech-to-text gets place names nearly right -- "libary",
"chemistry lap", "rest room", "room two oh one", and with an accent the
vowels go first: "libery", "cafitiria", "servar rum". An exact match refuses
every one of them. So a near miss is matched too, but never taken on trust:
the caller says what it understood and waits for continue before the dog
sets off. What this module promises is only that it does not guess when it
is unsure -- a different room number, or two places about as close as each
other, is None.

Three passes, cheapest and surest first:

  numbers_to_digits   "room two oh one" -> "room 201", before anything else
  fuzzy_place         spelling: difflib against every run of words
  near_place          ...and if spelling finds nothing, sound: the same
                      match on spelling folded to how it sounds (`sound`),
                      every vowel the same vowel

Plain Python, no dependencies: this is the PoC parser the dog runs without
--nlu, and the check the Gemma parser's answer is held to with it.
"""
import difflib
import re

FUZZY_MIN = 0.8           # difflib ratio; "libary" is 0.92, "the chemistry" 0.69
FUZZY_MARGIN = 0.05       # a runner-up this close is ambiguous: refuse, not guess
SOUND_MIN_LEN = 5         # letters: shorter names match on spelling only, because
                          # with every vowel alike "lift" and "left" are one word,
                          # and "turn left" became a trip to the lift


def fuzzy_place(names, command):
    """The place `command` nearly names, as (name, what was heard), or None.

    Each name is compared with every run of as many words in the command, and
    one more ("cafe teria"). Numbers must match exactly: "room 210" is a
    different room from "room 201", not a misspelling of it. Two names about
    as close as each other is a question this cannot answer, so it is None.
    """
    words = [w.strip(".,!?;:'\"") for w in command.lower().split()]
    words = [w for w in words if w]
    best = {}
    for name in names:
        n = len(name.split())
        digits = [c for c in name if c.isdigit()]
        for k in (n, n + 1):
            for i in range(len(words) - k + 1):
                heard = " ".join(words[i:i + k])
                if [c for c in heard if c.isdigit()] != digits:
                    continue
                r = difflib.SequenceMatcher(None, heard, name).ratio()
                if r > best.get(name, (0.0, ""))[0]:
                    best[name] = (r, heard)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    if not ranked or ranked[0][1][0] < FUZZY_MIN:
        return None
    if len(ranked) > 1 and ranked[1][1][0] > ranked[0][1][0] - FUZZY_MARGIN:
        return None
    name, (_, heard) = ranked[0]
    return name, heard


# --- sound ----------------------------------------------------------------
# Spelling folded to roughly how it is said. Not a phonetic algorithm, just
# the confusions speech-to-text makes with English place names: which vowel,
# ph or f, c or k or s, one letter or two.
_SPELLINGS = [("ph", "f"), ("ck", "k"), ("ch", "k"), ("wh", "w"), ("q", "k"),
              ("x", "ks"), ("z", "s")]


def sound(text):
    """'fisics' and 'physics' -> 'fasaks'; 'libery' -> 'labara', near 'library' -> 'labrara'."""
    s = text.lower()
    for written, said in _SPELLINGS:
        s = s.replace(written, said)
    s = re.sub(r"c(?=[eiy])", "s", s)                   # city, cent
    s = s.replace("c", "k")                             # cafe, lecture
    s = re.sub(r"[aeiouy]+", "a", s)                    # every vowel alike
    return re.sub(r"([b-df-hj-np-tv-z])\1+", r"\1", s)  # offis -> ofis


def near_place(names, command, hazards=()):
    """fuzzy_place on spelling, and on sound if spelling finds nothing.

    Sound skips names shorter than SOUND_MIN_LEN and the no-go places in
    `hazards`: "the store" sounding like "stairway" helps nobody, because
    that place is a refusal anyway, and the stairs are only ever taken from
    their name said out loud (BuildingRouter.hazard_named). Two places that
    sound the same are a question this cannot answer, so None.
    """
    near = fuzzy_place(names, command)
    if near is not None:
        return near
    by_sound = {}
    for name in names:
        if len(name.replace(" ", "")) >= SOUND_MIN_LEN and name not in hazards:
            by_sound.setdefault(sound(name), []).append(name)
    # Folded word by word, so the run that matched can be reported as it was
    # said ("servar rum"), not as its folded form ("sarvar ram").
    words = [w.strip(".,!?;:'\"") for w in command.lower().split()]
    words = [w for w in words if w]
    folded = [sound(w) for w in words]
    near = fuzzy_place(list(by_sound), " ".join(folded))
    if near is None or len(by_sound[near[0]]) > 1:
        return None
    k = len(near[1].split())
    heard = next(" ".join(words[i:i + k]) for i in range(len(words) - k + 1)
                 if " ".join(folded[i:i + k]) == near[1])
    return by_sound[near[0]][0], heard


# --- nicknames ----------------------------------------------------------------
# "toilet", "canteen", "ee lab": the aliases the Gemma parser is trained on
# (generate_dataset.py), read here as well, so a nickname reaches its place
# without the model -- with --nlu off it used to be refused. One table for
# both, so they cannot drift; add a building's real nicknames there.
# The stairs' aliases are left out: the stairs are only ever taken from
# their name said out loud (BuildingRouter.hazard_named), and "steps" is in
# "take a few steps back".
def _aliases():
    from cyberdog.language.generate_dataset import LAB_ALIASES, OTHER_ALIASES
    return {alias.lower(): place
            for table in (OTHER_ALIASES, LAB_ALIASES)
            for place, said in table.items() if place != "stairs"
            for alias in said if alias.lower() != place}


ALIASES = _aliases()


def alias_place(names, command):
    """The place a nickname in `command` stands for, as (place, nickname), or None.

    Whole words only ("loo" is not in "look"), longest first ("main office"
    before "office"). A nickname is still a guess -- "dining hall" holds
    "hall", which is the hallway -- so the caller asks before the dog moves.
    """
    low = command.lower()
    for alias in sorted(ALIASES, key=len, reverse=True):
        if ALIASES[alias] in names and re.search(rf"\b{re.escape(alias)}\b", low):
            return ALIASES[alias], alias
    return None


# --- movements ----------------------------------------------------------------
# "turn left" names no place, so it used to reach the Gemma parser, which was
# fine-tuned on destinations and always answers with one: "turn left" came
# back as the restroom. A command like that is a movement, not a trip, and
# the honest answer is that the dog only goes to places.
# Anchored at the start: "I need to pee right now" is a trip, and "right"
# anywhere in it must not make it a movement. The start may be the "Go to"
# that command_splitter puts before a command with no verb ("turn left" ->
# "Go to turn left").
_MOVEMENT = re.compile(
    r"^(?:go to\s+)?(?:(?:please|ok|okay|now|hey)[,\s]+)*"
    r"(turn|stop|wait|halt|faster|slower|slow down|speed up|back up|backwards?|"
    r"(?:go|walk|move|step|take me)\s+(?:left|right|back|backwards|forwards?|straight|ahead)|"
    r"left|right|forwards?|straight|ahead)\b")


def movement(command):
    """The movement a command that names no place starts with, or None.

    Only for a command already found to name no place: "the restroom on
    the left" is a trip to the restroom.
    """
    m = _MOVEMENT.search(command.lower().strip())
    return m.group(1) if m else None


# --- number words -----------------------------------------------------------
_DIGITS = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMBER = set(_DIGITS) | set(_TEENS) | set(_TENS) | {"hundred"}
_ZERO = ("o", "oh", "zero")


def _below_100(words):
    """'twenty one' -> 21, 'ten' -> 10, 'oh one' -> 1; None if it is not one."""
    if not words:
        return 0
    if len(words) == 1:
        w = words[0]
        return _DIGITS.get(w, _TEENS.get(w, _TENS.get(w)))
    if len(words) == 2 and words[0] in _TENS and words[1] in _DIGITS:
        return _TENS[words[0]] + _DIGITS[words[1]]
    if len(words) == 2 and words[0] in _ZERO and words[1] in _DIGITS:
        return _DIGITS[words[1]]
    return None


def _number(run):
    """A run of number words as a digit string, or None to leave it as words."""
    if "hundred" in run:                                # two hundred (and) one
        k = run.index("hundred")
        if k != 1 or run[0] not in _DIGITS:
            return None
        rest = _below_100([w for w in run[k + 1:] if w != "and"])
        return None if rest is None else str(_DIGITS[run[0]] * 100 + rest)
    if len(run) >= 2 and all(w in _DIGITS for w in run):
        return "".join(str(_DIGITS[w]) for w in run)    # two oh one, two zero one
    if len(run) >= 2 and run[0] in _DIGITS and _below_100(run[1:]) is not None:
        return f"{_DIGITS[run[0]]}{_below_100(run[1:]):02d}"   # two ten -> 210
    n = _below_100(run)
    return None if n is None else str(n)


def numbers_to_digits(text):
    """'take me to room two oh one' -> 'take me to room 201'.

    Room numbers are said, not spelt, and fuzzy_place matches digits exactly.
    A lone "o" or "oh" is left alone -- on its own it is usually just "oh" --
    and so is any run this cannot read as one number.
    """
    out, run = [], []

    def flush():
        low = [w.lower().strip(".,!?") for w in run]
        if any(w in _NUMBER and w not in _ZERO for w in low):
            n = _number(low)
            out.append(n if n is not None else " ".join(run))
        else:
            out.extend(run)
        run.clear()

    for w in re.split(r"[\s\-]+", text.strip()):
        low = w.lower().strip(".,!?")
        if (low in _NUMBER or low in _ZERO
                or (low == "and" and "hundred" in (r.lower() for r in run))):
            run.append(w)
        else:
            flush()
            out.append(w)
    flush()
    return " ".join(out)

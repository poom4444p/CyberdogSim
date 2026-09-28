# ---- Running on Linux ----
# No GPU or ML libraries needed -- plain Python 3.
#   python -m cyberdog.language.generate_dataset   # -> datasets/raw_dataset_english.json
# Output path comes from cyberdog.paths, so it works from any directory.
# --------------------------
import json
import random

from cyberdog import paths

#Config the building
NUM_FLOORS = 3
ROOMS_PER_FLOOR = 10

# Canonical location -> every alias the model is trained to map back to it.
# Module-level so other rule-based helpers (command_splitter.py) share the
# exact same vocabulary instead of a copy that can drift.
OTHER_ALIASES = {
    "hallway": ["hallway", "hall", "corridor"],
    "library": ["library", "lib"],
    "office": ["office", "main office"],
    "main entrance": ["main entrance", "front gate", "entrance", "front door",
                      "front entrance", "main door", "exit"],
    "cafeteria": ["cafeteria", "caf", "canteen", "cafe", "lunchroom"],
    "restroom": ["restroom", "bathroom", "toilet", "washroom", "loo", "gents",
                 "men's room", "ladies' room", "wc"],
    "server room": ["server room", "server closet", "IT room"],
    # The only way between floors (the stairs are refused by rules before the
    # model runs -- see BuildingRouter.hazard_named -- so they are not here).
    "lift": ["lift", "elevator"],
}

LAB_ALIASES = {
    "computer engineering lab": ["computer engineering lab", "computer lab", "cs lab", "comp eng lab",
                                 "computer engineering laboratory", "computer room"],
    "electrical engineering lab": ["electrical engineering lab", "ee lab", "electrical lab",
                                   "electrical engineering laboratory"],
    "mechanical engineering lab": ["mechanical engineering lab", "me lab", "mechanical lab",
                                   "mechanical engineering laboratory"],
    "chemistry lab": ["chemistry lab", "chem lab", "chemistry laboratory"],
    "biology lab": ["biology lab", "bio lab", "biology laboratory"],
    "physics lab": ["physics lab", "phys lab", "physics laboratory"],
}

# What the model answers for a place this building does not have. Without it,
# every training example ends in a real destination, so an unfamiliar word
# gets mapped onto the nearest name the model knows -- and the dog walks
# confidently to the wrong room. Callers treat this as "say so and stay put".
UNKNOWN_LOCATION = "unknown"

# Places that are NOT in the building. None contains an alias from the tables
# above ("post office" would teach the model that "office" is sometimes
# unknown), and they are not added to the splitter's vocabulary.
UNKNOWN_PLACES = [
    "gym", "parking lot", "car park", "swimming pool", "auditorium",
    "bookstore", "bus stop", "rooftop", "basement", "music room",
    "art studio", "garden",
]

# Held out: never in the training set, only in `generate_synthetic_dataset(
# held_out=True)`, which is what `evaluate.py --held-out` scores. Testing on
# the training vocabulary only says the model memorised it; these say whether
# it generalises to a name or a phrasing it has not seen. The splitter still
# knows the aliases -- it is rules, not the model, and has nothing to learn.
HELD_OUT_ALIASES = {
    "office": ["admin office"],
    "main entrance": ["way out"],
    "cafeteria": ["dining hall"],
    "restroom": ["lavatory"],
    "server room": ["data room"],
    "electrical engineering lab": ["electronics lab"],
    "mechanical engineering lab": ["mech lab"],
}
HELD_OUT_NAV_TEMPLATES = [
    "Which way to the {location}?",
    "Could you guide me to the {location}?",
    "We're going to the {location}.",
]
HELD_OUT_UNKNOWN_PLACES = ["sports field", "chapel", "pharmacy"]


def _generate_room_codes():
    codes = []
    for floor in range(1, NUM_FLOORS + 1):
        for room_num in range(1, ROOMS_PER_FLOOR + 1):
            codes.append(f"{floor}{room_num:02d}")
    return codes


def generate_synthetic_dataset(num_samples=1000, held_out=False):
    """Synthetic (command -> intent) samples.

    held_out=True draws only from the HELD_OUT_* vocabulary, so every sample
    has something the training set never contained: a navigation command in
    an unseen phrasing, or a question naming a place by an unseen alias.
    """
    # "Other" locations: single instance per room, generic "the {location}" phrasing.
    other_aliases = OTHER_ALIASES
    # Department labs (placeholders -- swap for your university's real dept
    # names once confirmed). Each is its own destination, not a generic "lab".
    lab_aliases = LAB_ALIASES
    unknown_places = UNKNOWN_PLACES
    if held_out:
        other_aliases = {k: v for k, v in HELD_OUT_ALIASES.items() if k in OTHER_ALIASES}
        lab_aliases = {k: v for k, v in HELD_OUT_ALIASES.items() if k in LAB_ALIASES}
        unknown_places = HELD_OUT_UNKNOWN_PLACES
    other_locations = list(other_aliases)
    lab_locations = list(lab_aliases)

    # Classrooms are numbered "<floor><2-digit room>" (e.g. floor 3, room 10
    # -> "310"), not a single generic "classroom" location.
    room_codes = _generate_room_codes()

    # Navigation / QA templates for the "other" + "lab" groups (take "the ...")
    nav_templates = [
        "Go to the {location}.",
        "Navigate to the {location}.",
        "Head over to the {location}.",
        "Please move to the {location}.",
        "I need you to go to the {location} right now.",
        "Proceed to the {location}.",
        # How people actually ask: questions and requests, not only orders.
        "Take me to the {location}, please.",
        "Can you take me to the {location}?",
        "Bring me to the {location}.",
        "Lead me to the {location}.",
        "Show me the way to the {location}.",
        "Where's the {location}?",
        "How do I get to the {location}?",
        "I want to go to the {location}.",
        "I'd like to go to the {location}.",
    ]
    if held_out:
        nav_templates = HELD_OUT_NAV_TEMPLATES
    qa_templates = [
        ("Go to the {location} and check if {query}", "{query}"),
        ("Navigate to the {location} and tell me if {query}", "{query}"),
        ("Check the {location}, {query}", "{query}"),
        ("Head to the {location} and see if {query}", "{query}"),
        ("I need to know if {query} in the {location}. Go check.", "{query}")
    ]

    # Same idea for numbered rooms, but phrased "room 310" not "the 310"
    room_nav_templates = [t.replace("the {location}", "room {location}") for t in nav_templates]
    room_qa_templates = [
        ("Go to room {location} and check if {query}", "{query}"),
        ("Navigate to room {location} and tell me if {query}", "{query}"),
        ("Check room {location}, {query}", "{query}"),
        ("Head to room {location} and see if {query}", "{query}"),
        ("I need to know if {query} in room {location}. Go check.", "{query}")
    ]

    queries = [
        "the lights are on",
        "anyone is there",
        "the door is open",
        "the projector is running",
        "the AC is turned off",
        "there are obstacles on the floor"
    ]

    # Slang phrases that imply a destination without naming any room at all
    # (e.g. "I need to pee" -> restroom). Navigation-only, no query field.
    implicit_slang = [
        ("I need to pee.", "restroom"),
        ("I need to go number 1.", "restroom"),
        ("I need to go number 2.", "restroom"),
        ("I gotta go potty.", "restroom"),
        ("Nature's calling.", "restroom"),
        ("I need to use the bathroom real quick.", "restroom"),
        ("I'm starving, let's go eat.", "cafeteria"),
        ("I need to grab some food.", "cafeteria"),
        ("I'm hungry, take me somewhere to eat.", "cafeteria"),
        ("I need to go study.", "library"),
        ("Let's go hit the books.", "library"),
        ("I need to wash my hands.", "restroom"),
        ("I have to freshen up.", "restroom"),
        ("I'm thirsty, let's get a drink.", "cafeteria"),
        ("Let's grab a coffee.", "cafeteria"),
        ("It's lunchtime.", "cafeteria"),
        ("I need somewhere quiet to read.", "library"),
        ("I want to borrow a book.", "library"),
        ("Get me out of this building.", "main entrance"),
        ("I'm done for the day, let's leave.", "main entrance"),
        ("I need to get to another floor.", "lift"),
    ]
    SLANG_PROBABILITY = 0.0 if held_out else 0.1  # fraction drawn from implicit_slang
    UNKNOWN_PROBABILITY = 0.08  # fraction naming a place the building lacks

    dataset = []

    for i in range(num_samples):
        if random.random() < SLANG_PROBABILITY:
            text, loc = random.choice(implicit_slang)
            dataset.append({
                "text": text,
                "location": loc,
                "task_type": "navigation",
                "question": None
            })
            continue

        # Pick a location group first (not weighted by how many locations
        # are in each group), so the 30 room numbers don't drown out the
        # labs/other locations just because there are more of them.
        group = ("unknown" if random.random() < UNKNOWN_PROBABILITY
                 else random.choice(["room", "lab", "other"]))
        if group == "unknown":
            # Same templates as a real place, so the only thing that tells
            # the two apart is the name itself.
            loc = UNKNOWN_LOCATION
            loc_text = random.choice(unknown_places)
            nav_tpl, qa_tpl = nav_templates, qa_templates
        elif group == "room":
            code = random.choice(room_codes)
            loc = f"room {code}"
            loc_text = code
            nav_tpl, qa_tpl = room_nav_templates, room_qa_templates
        elif group == "lab":
            loc = random.choice(lab_locations)
            loc_text = random.choice(lab_aliases[loc])
            nav_tpl, qa_tpl = nav_templates, qa_templates
        else:
            loc = random.choice(other_locations)
            loc_text = random.choice(other_aliases[loc])
            nav_tpl, qa_tpl = nav_templates, qa_templates

        task_choice = random.choice(["navigation", "visual_qa"])
        if held_out and group == "room":
            # A room number is never held out, so only a held-out phrasing
            # (a navigation template) makes this sample new.
            task_choice = "navigation"

        if task_choice == "navigation":
            text = random.choice(nav_tpl).format(location=loc_text)
            dataset.append({
                "text": text,
                "location": loc,
                "task_type": "navigation",
                "question": None
            })
        else:
            template, query_pattern = random.choice(qa_tpl)
            q = random.choice(queries)
            text = template.format(location=loc_text, query=q)
            dataset.append({
                "text": text,
                "location": loc,
                "task_type": "visual_qa",
                "question": q
            })

    # Shuffle the dataset to mix navigation and visual_qa tasks
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    print("Generating synthetic English dataset...")
    # 4000 samples so the ~44 distinct locations (30 numbered rooms + 6 labs
    # + 8 other) each get reasonable coverage, not just a handful of examples.
    data = generate_synthetic_dataset(4000)
    
    output_file = str(paths.RAW_DATASET)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully generated {len(data)} samples and saved to {output_file}")
    
    # verify sample data
    print("\n Example Data Samples:")
    for i in range(2):
        print(json.dumps(data[i], indent=2))

# ---- Running on Linux ----
# No GPU or ML libraries needed -- plain Python 3.
#   cd input_treating_layer
#   python3 generate_dataset.py      # -> raw_dataset_english.json
# Run it from inside input_treating_layer/ (output path is relative).
# --------------------------
import json
import random

#Config the building
NUM_FLOORS = 3
ROOMS_PER_FLOOR = 10

# Canonical location -> every alias the model is trained to map back to it.
# Module-level so other rule-based helpers (command_splitter.py) share the
# exact same vocabulary instead of a copy that can drift.
OTHER_ALIASES = {
    "hallway": ["hallway", "hall", "corridor"],
    "library": ["library", "lib"],
    "office": ["office"],
    "main entrance": ["main entrance", "front gate", "entrance", "front door"],
    "cafeteria": ["cafeteria", "caf", "canteen"],
    "restroom": ["restroom", "bathroom", "toilet", "washroom"],
    "server room": ["server room"],
}

LAB_ALIASES = {
    "computer engineering lab": ["computer engineering lab", "computer lab", "cs lab", "comp eng lab"],
    "electrical engineering lab": ["electrical engineering lab", "ee lab", "electrical lab"],
    "mechanical engineering lab": ["mechanical engineering lab", "me lab", "mechanical lab"],
    "chemistry lab": ["chemistry lab", "chem lab"],
    "biology lab": ["biology lab", "bio lab"],
    "physics lab": ["physics lab", "phys lab"],
}


def _generate_room_codes():
    codes = []
    for floor in range(1, NUM_FLOORS + 1):
        for room_num in range(1, ROOMS_PER_FLOOR + 1):
            codes.append(f"{floor}{room_num:02d}")
    return codes


def generate_synthetic_dataset(num_samples=1000):
    # "Other" locations: single instance per room, generic "the {location}" phrasing.
    other_locations = [
        "hallway", "library", "office",
        "main entrance", "cafeteria", "restroom", "server room"
    ]
    other_aliases = OTHER_ALIASES

    # Department labs (placeholders -- swap for your university's real dept
    # names once confirmed). Each is its own destination, not a generic "lab".
    lab_locations = [
        "computer engineering lab", "electrical engineering lab",
        "mechanical engineering lab", "chemistry lab", "biology lab", "physics lab"
    ]
    lab_aliases = LAB_ALIASES

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
        "Proceed to the {location}."
    ]
    qa_templates = [
        ("Go to the {location} and check if {query}", "{query}"),
        ("Navigate to the {location} and tell me if {query}", "{query}"),
        ("Check the {location}, {query}", "{query}"),
        ("Head to the {location} and see if {query}", "{query}"),
        ("I need to know if {query} in the {location}. Go check.", "{query}")
    ]

    # Same idea for numbered rooms, but phrased "room 310" not "the 310"
    room_nav_templates = [
        "Go to room {location}.",
        "Navigate to room {location}.",
        "Head over to room {location}.",
        "Please move to room {location}.",
        "I need you to go to room {location} right now.",
        "Proceed to room {location}."
    ]
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
    ]
    SLANG_PROBABILITY = 0.1  # fraction of samples drawn from implicit_slang

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
        group = random.choice(["room", "lab", "other"])
        if group == "room":
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
    # 4000 samples so the ~43 distinct locations (30 numbered rooms + 6 labs
    # + 7 other) each get reasonable coverage, not just a handful of examples.
    data = generate_synthetic_dataset(4000)
    
    output_file = "raw_dataset_english.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully generated {len(data)} samples and saved to {output_file}")
    
    # verify sample data
    print("\n Example Data Samples:")
    for i in range(2):
        print(json.dumps(data[i], indent=2))

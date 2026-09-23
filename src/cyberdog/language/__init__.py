"""Spoken command -> structured navigation intent.

`infer.py` runs the gemma-2b-it LoRA fine-tune; `command_splitter.py` and
`floor_parser.py` are the deterministic rules around it, because splitting
"the library then the cafeteria" into two destinations and reading "on the
second floor" as floor 2 are things a rule does correctly every time.

Training and data generation (`train.py`, `generate_dataset.py`,
`prepare_dataset.py`, `evaluate.py`) live here too but are not part of the
runtime path.
"""

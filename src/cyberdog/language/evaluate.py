"""Batch-evaluate the Input Treating Layer model on freshly generated samples.

Usage:
    python -m cyberdog.language.evaluate --num-samples 100

Running on Linux:
    Setup (once):
        python3 -m venv .venv && source .venv/bin/activate
        pip install torch transformers peft trl datasets accelerate
        (NVIDIA GPU: install the CUDA build of torch from pytorch.org first)
    Loads the adapter from models/lora:
        python -m cyberdog.language.evaluate --num-samples 100
    Uses cuda automatically if available (see infer.get_device).
    KMP_DUPLICATE_LIB_OK below is a macOS-only fix and does nothing on Linux.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random

from cyberdog.language.generate_dataset import generate_synthetic_dataset
from cyberdog.language.infer import parse_command


def _norm(text):
    return (text or "").strip().lower()


def evaluate(num_samples: int, seed: int, held_out: bool = False):
    random.seed(seed)
    samples = generate_synthetic_dataset(num_samples, held_out=held_out)

    results = []
    counts = {"valid_json": 0, "location": 0, "task": 0, "query": 0, "exact": 0}

    for sample in samples:
        expected = {
            "target_location": sample["location"],
            "task": sample["task_type"],
            "query": sample["question"],
        }

        row = {"text": sample["text"], "expected": expected}

        try:
            actual = parse_command(sample["text"])
            row["actual"] = actual
            row["valid_json"] = True
        except ValueError as e:
            row["actual"] = None
            row["valid_json"] = False
            row["error"] = str(e)
            results.append(row)
            continue

        loc_match = _norm(actual.get("target_location")) == _norm(expected["target_location"])
        task_match = actual.get("task") == expected["task"]
        query_match = _norm(actual.get("query")) == _norm(expected["query"])

        row.update(loc_match=loc_match, task_match=task_match, query_match=query_match)
        results.append(row)

        counts["valid_json"] += 1
        counts["location"] += loc_match
        counts["task"] += task_match
        counts["query"] += query_match
        counts["exact"] += loc_match and task_match and query_match

    return samples, results, counts


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate the Input Treating Layer model")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123,
                         help="Different from any seed used to build the training data")
    parser.add_argument("--save-failures", type=str, default="eval_failures.json")
    parser.add_argument("--held-out", action="store_true",
                        help="Score only aliases, phrasings and unknown places the "
                             "training set never contained (HELD_OUT_* in generate_dataset.py)")
    args = parser.parse_args()

    samples, results, counts = evaluate(args.num_samples, args.seed, args.held_out)
    n = len(samples)

    print(f"\nSamples evaluated: {n}")
    print(f"  Valid JSON output      : {counts['valid_json']}/{n} ({counts['valid_json']/n:.1%})")
    print(f"  target_location correct: {counts['location']}/{n} ({counts['location']/n:.1%})")
    print(f"  task correct           : {counts['task']}/{n} ({counts['task']/n:.1%})")
    print(f"  query correct          : {counts['query']}/{n} ({counts['query']/n:.1%})")
    print(f"  Fully correct (all 3)  : {counts['exact']}/{n} ({counts['exact']/n:.1%})")

    failures = [
        r for r in results
        if not r.get("valid_json") or not (r.get("loc_match") and r.get("task_match") and r.get("query_match"))
    ]
    if failures and args.save_failures:
        with open(args.save_failures, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(failures)} failing case(s) to {args.save_failures} for inspection.")
    elif not failures:
        print("\nNo failures.")

    if args.held_out:
        print(
            "\nNote: held-out samples -- every one has an alias, phrasing or unknown place "
            "the training set never contained, so this measures generalization."
        )
    else:
        print(
            "\nNote: these samples come from the same templates used to build the training "
            "set (generate_dataset.py), so this measures in-distribution correctness. "
            "Run with --held-out to measure generalization to unseen wording."
        )


if __name__ == "__main__":
    main()

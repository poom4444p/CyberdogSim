# NOTE for Linux/Windows users: this project was built and trained on a Mac
# (Apple Silicon, MPS backend). The line below works around a macOS-only
# issue where multiple copies of the OpenMP runtime (libomp) get linked in
# via conda + PyTorch and crash the process. It is a no-op on Linux/Windows,
# so it's safe to leave in place there too.
#
# Running on Linux: this is a library module, not a script -- it's imported by
# scripts/language/try_parser.py, evaluate.py, main_planner.py and tests/test_locations.py.
# Setup (once):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install torch transformers peft trl datasets accelerate
#   (NVIDIA GPU: install the CUDA build of torch from pytorch.org first)
# Older NVIDIA GPUs without bfloat16 support (pre-Ampere, e.g. GTX 10xx / T4):
# change torch_dtype=torch.bfloat16 to torch.float16 in load_model().
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = os.environ.get("KMP_DUPLICATE_LIB_OK") or "TRUE"

import json

import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, StoppingCriteria,
                          StoppingCriteriaList)
from peft import PeftModel

BASE_MODEL_NAME = "unsloth/gemma-2b-it"
from cyberdog.paths import LORA_ADAPTER as ADAPTER_DIR
SYSTEM_PROMPT = (
    "You are a robotic instruction parser. Extract intent into JSON format "
    "containing 'target_location', 'task', and 'query'."
)

_tokenizer = None
_model = None


def get_device() -> str:
    """Pick the best available torch device for this machine.

    - Linux/Windows with an NVIDIA GPU -> "cuda"
    - Mac with Apple Silicon (what this project was trained/tested on) -> "mps"
    - Anything else -> "cpu" (works, just slow)

    This already does the right thing automatically on Linux — nothing to
    change here unless you want to force a specific device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load the base model + LoRA adapter. Safe to call repeatedly."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    device = get_device()
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    _model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    return _tokenizer, _model


def json_closed(text: str) -> bool:
    """Has `text` finished its first JSON object -- every { opened so far closed?"""
    depth = 0
    for c in text:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


class _JsonDone(StoppingCriteria):
    """Stop generating once the answer's JSON closes.

    The fine-tune never learned to stop: after its answer (~20 tokens) it
    writes "model" and the same answer again until max_new_tokens, and only
    the first copy is ever read (_extract_json). Measured on an M-series Mac,
    "I need to pee": 100 tokens in 6.3 s as it was, 20 tokens in 1.4 s with
    this, same answer.
    """

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer, self.prompt_len = tokenizer, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        return json_closed(self.tokenizer.decode(input_ids[0, self.prompt_len:]))


def generate_raw(user_text: str, max_new_tokens: int = 100) -> str:
    """Run the model on a user command and return its raw text output,
    up to the end of its first JSON object."""
    tokenizer, model = load_model()

    combined_text = SYSTEM_PROMPT + "\n\n" + user_text
    messages = [{"role": "user", "content": combined_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    prompt_len = inputs.input_ids.shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             stopping_criteria=StoppingCriteriaList(
                                 [_JsonDone(tokenizer, prompt_len)]))
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def parse_command(user_text: str, max_new_tokens: int = 100) -> dict:
    """Parse a natural-language robot command into structured intent JSON.

    Returns a dict with 'target_location', 'task', and 'query' keys.
    Raises ValueError if the model output can't be parsed as JSON.
    """
    raw_answer = generate_raw(user_text, max_new_tokens=max_new_tokens)
    return _extract_json(raw_answer)


def _extract_json(text: str) -> dict:
    """Decode the first JSON object in the model's output, ignoring anything
    that follows (small fine-tuned models sometimes trail off into repeated
    or malformed text after a valid first answer)."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Model output did not contain JSON: {text!r}")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {text!r}") from e
    return obj

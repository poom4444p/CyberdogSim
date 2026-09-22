# ---- Running on Linux ----
# Setup (once):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install torch transformers peft trl datasets accelerate
#   (NVIDIA GPU: install the CUDA build of torch from pytorch.org first)
# Run from inside input_treating_layer/ (loads ./mac_lora_final):
#   cd input_treating_layer
#   python3 test_model.py
# Uses cuda automatically if available (see infer.get_device). KMP_DUPLICATE_LIB_OK
# below is a macOS-only fix and does nothing on Linux.
# --------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from infer import load_model, generate_raw

def main():
    print("="*50)
    print(" CYBERDOG INPUT TREATING LAYER (TEST) ")
    print("="*50)

    print("\n[1] Loading Base Model + Fine-Tuned LoRA Weights...")
    load_model()
    print("Model is ready!")

    print("\n--- Interactive Testing ---")
    print("Type a command for Cyberdog (or type 'quit' to exit)")

    while True:
        user_input = input("\n[You (User)]: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        print("[Cyberdog Brain]: ", end="", flush=True)
        print(generate_raw(user_input))

if __name__ == "__main__":
    main()

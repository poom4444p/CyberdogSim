# NOTE for Linux/Windows users: this script was built and tested on a Mac
# with Apple Silicon (MPS backend) -- it used to be called train_mac.py.
# Training on Linux with an NVIDIA GPU works too and needs no manual edits —
# device selection below (get_device(), from infer.py) already picks "cuda"
# first, "mps" second, "cpu" last. The KMP_DUPLICATE_LIB_OK env var is a
# macOS-only workaround for a libomp conflict; it's a no-op elsewhere.
#
# Running on Linux:
#   Setup (once):
#     python3 -m venv .venv && source .venv/bin/activate
#     pip install torch transformers peft trl datasets accelerate
#     (NVIDIA GPU: install the CUDA build of torch from pytorch.org first)
#   Reads datasets/gemma_training_data.jsonl, writes models/lora:
#     python -m cyberdog.language.train
#   Check the GPU is used:  python3 -c "import torch; print(torch.cuda.is_available())"
#   Older NVIDIA GPUs without bfloat16 (pre-Ampere, e.g. GTX 10xx / T4): change
#   torch_dtype=torch.bfloat16 to torch.float16 below.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

from cyberdog.language.infer import get_device

from cyberdog import paths

def main():
    # 1. Pick the fastest available device (see infer.get_device for the order)
    device = get_device()
    print(f"Using device: {device}" + (" (this will be slow)" if device == "cpu" else ""))

    # 2. โหลดโมเดล (ใช้เวอร์ชันของ unsloth ที่ปลดล็อคแล้ว ไม่ต้องใส่ token)
    model_name = "unsloth/gemma-2b-it"
    print(f"Loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bfloat16 keeps memory usage down; supported on Apple Silicon, modern
    # NVIDIA GPUs (Ampere+), and CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # 3. เตรียมการทำ LoRA (เทรนเฉพาะบางส่วนเพื่อให้ใช้แรมน้อยลงและเทรนเร็วขึ้น)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. โหลด Dataset ที่เราเพิ่งสร้าง
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=str(paths.TRAINING_JSONL), split="train")

    def fix_system_role(examples):
        # SFTTrainer automatically looks for the 'messages' column in newer versions.
        # We simply override the 'messages' column to merge 'system' into 'user'.
        new_batch = []
        for conversation in examples["messages"]:
            new_conv = []
            sys_text = ""
            for msg in conversation:
                if msg["role"] == "system":
                    sys_text = msg["content"]
                elif msg["role"] == "user":
                    combined = sys_text + "\n\n" + msg["content"] if sys_text else msg["content"]
                    new_conv.append({"role": "user", "content": combined})
                else:
                    new_conv.append(msg)
            new_batch.append(new_conv)
        return {"messages": new_batch}

    # Override the messages column in the dataset natively
    dataset = dataset.map(fix_system_role, batched=True)

    # Hold out a fixed-size validation slice (not a %) so eval cost doesn't
    # balloon as the dataset grows -- watching eval_loss lets us catch
    # overfitting instead of just training for a fixed number of steps blind.
    eval_size = min(150, len(dataset) // 10)
    split = dataset.train_test_split(test_size=eval_size, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # 5. ตั้งค่าการเทรน
    training_args = TrainingArguments(
        output_dir="./mac_lora_outputs",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=2,
        num_train_epochs=3,        # ~3 full passes over the training split
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,   # keep the checkpoint with lowest eval_loss,
        metric_for_best_model="eval_loss",  # not just the last one (which may be overfit)
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args
    )

    # 6. เริ่มเทรน!
    print("Starting training...")
    trainer.train()

    # 7. เซฟโมเดลที่เทรนเสร็จ
    model.save_pretrained(str(paths.LORA_ADAPTER))
    tokenizer.save_pretrained(str(paths.LORA_ADAPTER))
    print(f"Done! Model saved to {paths.LORA_ADAPTER}")

if __name__ == "__main__":
    main()

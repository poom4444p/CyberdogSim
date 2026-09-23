# ---- Running on Linux ----
# No GPU or ML libraries needed -- plain Python 3.
#   python -m cyberdog.language.prepare_dataset    # raw_dataset -> training jsonl
# Input/output paths come from cyberdog.paths, so it works from any directory.
# --------------------------
import json

from cyberdog import paths

def convert_to_conversation(sample):
    """Converts raw data into conversation format for Gemma fine-tuning."""
    user_command = sample.get("text", "")
    
    target_json = {
        "target_location": sample.get("location", ""),
        "task": sample.get("task_type", ""),
        "query": sample.get("question", None)
    }
    
    # Convert Dictionary to JSON String
    assistant_response = json.dumps(target_json, ensure_ascii=False)
    
    # Query for training Gemma
    conversation = {
        "messages": [
            {
                "role": "system", 
                "content": "You are a robotic instruction parser. Extract intent into JSON format containing 'target_location', 'task', and 'query'."
            },
            {
                "role": "user", 
                "content": user_command
            },
            {
                "role": "assistant", 
                "content": assistant_response
            }
        ]
    }
    
    return conversation

if __name__ == "__main__":
    input_file = str(paths.RAW_DATASET)
    output_file = str(paths.TRAINING_JSONL)
    
    print(f"Loading raw dataset from {input_file}...")
    
    # Load JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)
        
    print(f"Loaded {len(raw_dataset)} samples.")
    print("Converting dataset for Gemma fine-tuning...")
    
    # Convert all samples to conversation format
    converted_dataset = [convert_to_conversation(sample) for sample in raw_dataset]
    
    # Save as JSONL (JSON Lines) for Fine-tuning
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Done! Saved {len(converted_dataset)} conversation samples to {output_file}")

#!/usr/bin/env python3
import os
import json
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path

def download_and_modify_model(model_id, output_dir=None, modify=True):
    """
    Downloads a Huggingface model and optionally modifies its adapter_config.json file.
    
    Args:
        model_id: Huggingface model ID (e.g., "mateoguaman/paligemma2-3b-pt-224-sft-lora-magicsoup")
        output_dir: Directory to save the model (default: ./models)
        modify: Whether to modify the adapter_config.json file
    
    Returns:
        Path to the downloaded model
    """
    # Extract model name from model_id
    model_name = model_id.split('/')[-1]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = './models'
    
    # Always create a separate folder for the model
    model_dir = os.path.join(output_dir, model_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading model {model_id} to {model_dir}...")
    
    # Download the model
    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False  # Use actual files, not symlinks
    )
    
    print(f"Model downloaded to: {local_dir}")
    
    # Check if adapter_config.json exists
    adapter_config_path = os.path.join(local_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"Found adapter_config.json at {adapter_config_path}")
        
        if modify:
            # Load the config
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
            
            # Display the current config
            print("Current adapter config:")
            print(json.dumps(config, indent=2))
            
            # Check if problematic key exists
            if 'eva_config' in config:
                print("\nFound 'eva_config' key that might cause issues with older PEFT versions")
                
                # Remove problematic key
                if input("Remove 'eva_config' key? (y/n): ").lower() == 'y':
                    del config['eva_config']
                    
                    # Save modified config
                    with open(adapter_config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print("Removed 'eva_config' key and saved modified config")
            
            # Manual editing option
            if input("\nWould you like to manually edit the config file? (y/n): ").lower() == 'y':
                # Determine which editor to use
                editor = os.environ.get('EDITOR', 'nano')
                os.system(f"{editor} {adapter_config_path}")
                print(f"Config file edited with {editor}")
    else:
        print("No adapter_config.json found in the model directory")
    
    return local_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and modify Huggingface models")
    parser.add_argument("--model_id", help="Huggingface model ID")
    parser.add_argument("--output-dir", help="Base directory to save models (default: ./models)")
    parser.add_argument("--no-modify", action="store_true", help="Skip modification prompt")
    
    args = parser.parse_args()
    
    local_path = download_and_modify_model(
        args.model_id, 
        args.output_dir,
        not args.no_modify
    )
    
    print(f"\nModel ready at: {local_path}")
    print(f"You can now use this path in your application with model_name_or_path=\"{local_path}\"")
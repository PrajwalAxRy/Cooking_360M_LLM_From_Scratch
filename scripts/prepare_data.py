import os
import tiktoken
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
import argparse

def process_and_save(config_path):
    """
    Downloads, tokenizes, and saves the TinyStories dataset to .bin files
    for training. This is a one-time setup script.

    Can also be simply done by downloding it from Hugging Face.
    """
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    dataset_name = data_config['dataset_name']
    tokenizer_name = data_config['tokenizer']
    train_bin_path = data_config['train_bin_path']
    val_bin_path = data_config['val_bin_path']

    # --- 2. Check if data already exists ---
    if os.path.exists(train_bin_path) and os.path.exists(val_bin_path):
        print(f"Tokenized data files '{train_bin_path}' and '{val_bin_path}' already exist. Skipping preparation. Delete them if you want to reprocess.")
        return

    # --- 3. Download and Process Dataset ---
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")
    dataset_loaded = load_dataset(dataset_name)
    
    print(f"Initializing tokenizer: '{tokenizer_name}'")
    enc = tiktoken.get_encoding(tokenizer_name)

    def tokenize_function(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token) # Add End-of-Text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing dataset splits (this may take a while)...")

    tokenized = dataset_loaded.map(
        tokenize_function,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=os.cpu_count(),
    )

    # --- 4. Save to Memory-Mapped Files ---
    for split, dset in tokenized.items():
        filename = train_bin_path if split == 'train' else val_bin_path
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        
        print(f"Writing {len(dset)} documents to '{filename}'...")
        # Use uint16 since gpt2 vocab_size is 50257, which fits in 16 bits
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        for example in tqdm(dset, desc=f"Writing {split} split"):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
        print(f"Finished writing {split} split. Total tokens: {arr_len}")

if __name__ == '__main__':
    # Create Parser
    parser = argparse.ArgumentParser(description="Download and tokenize a dataset for LLM training.")
    # Add argument for config file path
    parser.add_argument('--config', type=str, default='configs/gemma_270m.yaml',
                        help='Path to the configuration YAML file.')
    # Parse arguments
    args = parser.parse_args()

    # Run the processing function
    process_and_save(args.config)

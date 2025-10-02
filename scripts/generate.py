### Inference Script

import torch
import tiktoken
import argparse

from llm_foundry.model.AxeGPT import AxeGPT
from llm_foundry.utils.config import load_config

def generate(config_path: str, checkpoint_path: str, prompt: str, max_new_tokens: int = 100, temperature: float = 1, top_k: int = 100):
    """
    Generate text using a pre-trained AxeGPT model.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ##Load config
    model_config = load_config(config_path)

    ##Initialize model
    model = AxeGPT(model_config['model'])
    model.to(device)

    ##Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    ##Load tokenizer
    tokenizer = tiktoken.get_encoding(model_config["data"]["tokenizer"])
    #Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)  # Add batch dimension

    #Generate text
    print("\n------- Generating text... -------\n")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        output_text = tokenizer.decode(output_ids[0].tolist())

    print(output_text)
    print("\n------- Generation complete -------\n")

def main():
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained AxeGPT model.")
    parser.add_argument("--config", type=str, default="configs/llm_360M.yaml", help="Path to the model configuration file.")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model_10.pth", help="Path to the model checkpoint file.")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there", help="Prompt text to start the generation.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=100, help="Top-K sampling.")

    args = parser.parse_args()

    generate(args.config, args.checkpoint, args.prompt, args.max_new_tokens, args.temperature, args.top_k)

if __name__ == "__main__":
    main()
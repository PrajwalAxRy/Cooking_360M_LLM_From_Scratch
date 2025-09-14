import torch 
import os
import yaml
from tqdm import tqdm

import tiktoken

from llm_foundry.model.AxeGPT import AxeGPT
from llm_foundry.data.dataset import MemmapDataset
from llm_foundry.utils.config import load_config

class Trainer:
    """
     Trainer class to setup training and evaluation loops for LLMs.
    """
    def __init__(self, config_path:str):
        #Load Model Conifg
        self.model_config = load_config(config_path)

        #Use correct Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        torch.set_default_device(self.device)

        ## Use tokenizer
        self.tokenizer = tiktoken.get_encoding(self.model_config["data"]["tokenizer"])

        ## Dataset
        self.train_data = MemmapDataset(
            bin_path=self.model_config['data']['train_bin_path'],
            context_length=self.model_config['model']['context_length'],
            batch_size=self.model_config['training']['batch_size'],
            device_type=self.device
        )
        self.val_data = MemmapDataset(
            bin_path=self.model_config['data']['val_bin_path'],
            context_length=self.model_config['model']['context_length'],
            batch_size=self.model_config['training']['batch_size'],
            device_type=self.device
        )


        # Model
        self.model = AxeGPT(self.model_config["model"])
        self.model.to(self.device)
        print(f"Model intialized with {sum(p.numel() for p in self.model.parameters()):,} parameters.")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
                                           self.model.parameters(), 
                                           lr=self.model_config["training"]["learning_rate"],
                                           betas=(self.model_config["training"]["beta1"], self.model_config["training"]["beta2"]),
                                           weight_decay=self.model_config["training"]["weight_decay"]
                                           )

        # Training State
        self.best_val_loss = float('inf')
        self.step = 0

        # Directory for saving other outputs if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

    @torch.no_grad()
    def run_eval(self):
        """
         Runs evaluation on the validation dataset.
        """
        self.model.eval() # Set model to evaluation mode. Note: Make sure to call model.train() to set it back to training mode after eval.

        losses = torch.zeros(self.model_config['training']['eval_iters'])
        
        for k in range(self.model_config['training']['eval_iters']):
            X, Y = self.val_data.get_batch()
            _, loss = self.model(X, Y)
            losses[k] = loss.item()
        
        val_loss = losses.mean()
        print(f"Step {self.step}: Validation loss {val_loss:.4f}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"New best validation loss: {self.best_val_loss:.4f}. Saving model checkpoint...")
            torch.save(self.model.state_dict(), f"outputs/best_model_{self.step}.pth")

        self.model.train() # Set model back to training mode
        
    @torch.no_grad()
    def _run_sampling(self):
        """
         Generates sample text from the model.
        """
        self.model.eval() # Set model to evaluation mode. 
        sampling_config = self.model_config['training']['sampling']

        print(f"\n--- Sampling at step {self.step} ---")

        prompt = sampling_config["prompt"]
        
        #tokenize
        context = self.tokenizer.encode(prompt)
        # Make it a tensor
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        # Add batch dimension
        context = context.unsqueeze(0)  # Shape: (1, sequence_length) 

        generated_tokens = self.model.generate(context, max_new_tokens=25)
        generated_text = self.tokenizer.decode(generated_tokens[0].tolist()) # "tolist() to convert tensor to list"

        print(generated_text)
        print("\n--------------------------------------------\n")

        with open(f"outputs/{sampling_config['output_file']}", "a") as f:
            f.write(f"\n--- Sampling at step {self.step} ---\n")
            f.write(generated_text)
            f.write("\n--------------------------------------------\n")
        
        self.model.train() # Set model back to training mode

    def train(self):
        """
         Main training loop.
        """
        self.model.train() # Set model to training mode.

        total_step = self.model_config['training']['max_iters']
        progress_bar = tqdm(range(total_step), desc="Training", unit="step")

        for step in progress_bar:
            self.step = step # Keep track of current step

            if step > 0:
                if step % self.model_config['training']['eval_interval'] == 0:
                    self.run_eval()
                if (self.model_config['training']['sampling']['enable'] and 
                    step % self.model_config['training']['sampling']['interval'] == 0):
                    self._run_sampling()

            # Reset gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Gradient Accumulation
            accumulation_steps = self.model_config['training']['gradient_accumulation_steps']
            for _ in range(accumulation_steps):
                inputs, targets = self.train_data.get_batch()
                output, loss = self.model(inputs, targets) # we dont care about output here

                # Scale loss by accumulation steps. We do this because we want to average the gradients over the accumulation steps.
                loss = loss / accumulation_steps
                loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config['training']['grad_clip'])
            self.optimizer.step()

            ## Progress bar update
            progress_bar.set_description(
                f"Step {step+1}/{total_step} | Loss: {loss.item()*accumulation_steps:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
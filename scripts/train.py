import argparse
from llm_foundry.engine.trainer import Trainer

def main():
    """
    Main entry point for starting a training run.
    """
    parser = argparse.ArgumentParser(description="Train AxeGPT Model")
    parser.add_argument(
        '--config', 
        type=str,  
        default='configs/llm_270m.yaml',
        help='Path to the configuration YAML file.'
    )
    args = parser.parse_args()

    # Initialize and run the trainer
    trainer = Trainer(config_path=args.config)
    trainer.train()

if __name__ == '__main__':
    main()
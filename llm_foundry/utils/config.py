import yaml

def load_config(path="configs/gemma_270m.yaml"):
    """
    Loads a YAML configuration file.

    Args:
        path (str): The path to the YAML file.

    Returns:
        A dictionary containing the configuration.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

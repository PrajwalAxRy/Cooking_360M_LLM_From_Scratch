import torch
import yaml
import pytest

from llm_foundry.model.AxeGPT import AxeGPT
from llm_foundry.utils.config import load_config

# Define the path to your config file
CONFIG_PATH = "configs/llm_270m.yaml"

@pytest.fixture(scope="module")
def config():
    """Pytest fixture to load the config file once per module."""
    return load_config(CONFIG_PATH)

def test_model_initialization(config):
    """Tests if the AxeGPT can be initialized without errors."""
    try:
        model = AxeGPT(config['model'])
        assert model is not None, "Model should not be None after initialization"
    except Exception as e:
        pytest.fail(f"Model initialization failed with an exception: {e}")

def test_model_forward_pass(config):
    """
    Tests the forward pass of the model with a dummy input.
    Verifies the output shape of the logits.
    """
    model_config = config['model']
    model = AxeGPT(model_config)

    # Create a dummy input tensor
    batch_size = 4
    # Use a shorter sequence length for testing to speed things up
    seq_len = config['data']['context_length'] 
    vocab_size = model_config['vocab_size']
    
    # Dummy input token IDs
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Perform a forward pass
    logits, loss = model(dummy_input)

    # Check the shape of the output logits
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, (
        f"Logits shape is incorrect. Expected {expected_shape}, but got {logits.shape}"
    )

    # Loss should be None since we didn't provide targets
    assert loss is None, f"Loss should be None when no targets are provided, but got {loss}"

def test_model_forward_pass_with_targets(config):
    """
    Tests the forward pass of the model with both inputs and targets.
    Verifies that a scalar loss value is computed.
    """
    model_config = config['model']
    model = AxeGPT(model_config)

    # Create dummy input and target tensors
    batch_size = 4
    seq_len = config['data']['context_length']
    vocab_size = model_config['vocab_size']
    
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

    # Perform a forward pass
    logits, loss = model(dummy_input, dummy_targets)

    # Check that the loss is a single scalar value and is not None
    assert loss is not None, "Loss should not be None when targets are provided"
    assert loss.ndim == 0, f"Loss should be a scalar, but has {loss.ndim} dimensions"


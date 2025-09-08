import torch
import numpy as np

class MemmapDataset:
    """
    A PyTorch-style Dataset that loads data from a memory-mapped file.
    This is highly efficient for large datasets that don't fit in RAM.
    """
    def __init__(self, bin_path, context_length, batch_size, device_type='cpu'):
        """
        Initializes the dataset.

        Args:
            bin_path (str): Path to the .bin file containing tokenized data.
            context_length (int): The sequence length for each sample.
            batch_size (int): The number of sequences in a batch.
            device_type (str): The device to move tensors to ('cpu' or 'cuda').
        """
        self.context_length = context_length
        self.batch_size = batch_size
        self.device_type = device_type
        
        print(f"Initializing dataset from {bin_path}...")
        # We use np.uint16 because the gpt2 tokenizer has a vocab size of 50257,
        # which fits within the range of an unsigned 16-bit integer (0-65535).
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"Successfully loaded {len(self.data)} tokens.")


    def get_batch(self):
        """
        Selects a random batch of data from the memory-mapped file.

        Returns:
            A tuple (x, y) where x is the input tensor and y is the target tensor.
        """
        # Generate random starting points for each sequence in the batch
        ix = torch.randint(len(self.data) - self.context_length, (self.batch_size,))

        # Create input sequences (x)
        x_list = []
        for i in ix:
            x_seq = torch.from_numpy(self.data[i : i + self.context_length].astype(np.int64)) #from_numpy creates a tensor from a numpy array
            x_list.append(x_seq)
        x = torch.stack(x_list) # x_list is a list of tensors, stack them into a single tensor of batch_size x context_length

        # Create target sequences (y), which are shifted by one
        y_list = []
        for i in ix:
            y_seq = torch.from_numpy(self.data[i + 1 : i + 1 + self.context_length].astype(np.int64))
            y_list.append(y_seq)
        y = torch.stack(y_list)

        # Move tensors to the appropriate device
        if self.device_type == 'cuda':
            return x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        else:
            return x, y


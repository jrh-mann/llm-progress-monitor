'''
Stage 3: Load activations and prepare datasets
'''
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict
import torch
import gc
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from jaxtyping import Float, Int

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActivationData:
    """Container for loaded activation data"""
    def __init__(
        self, 
        token_ids: Int[torch.Tensor, "seq_len"], 
        activations: Float[torch.Tensor, "n_layers seq_len d_model"]
    ):
        self.token_ids = token_ids  # Shape: (seq_len,)
        self.activations = activations  # Shape: (n_layers, seq_len, d_model)


def load_activations(
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None
) -> List[ActivationData]:
    """
    Load activations from saved files.
    
    Args:
        activations_dir: Directory containing activation files
        start_idx: Starting index for loading activations (defaults to 0)
        end_idx: Ending index for loading activations (defaults to None, loads all available)
        layer_idx: Specific layer to extract (defaults to None, loads all layers)
        
    Returns:
        List of ActivationData objects containing token_ids and activations
        
    Raises:
        FileNotFoundError: If activations_dir doesn't exist
        ValueError: If no activations are found in the specified range
    """
    if not os.path.exists(activations_dir):
        raise FileNotFoundError(f"Activations directory not found: {activations_dir}")
    
    logger.info(f"Loading activations from: {activations_dir}")
    
    # Determine end_idx if not specified
    if end_idx is None:
        # Find the maximum index in the directory
        activation_files = [f for f in os.listdir(activations_dir) if f.endswith('.pt')]
        if not activation_files:
            raise ValueError(f"No activation files found in {activations_dir}")
        indices = [int(f.split('.')[0]) for f in activation_files]
        end_idx = max(indices) + 1
    
    logger.info(f"Loading activations from index {start_idx} to {end_idx}")
    
    activations_list: List[ActivationData] = []
    
    with torch.no_grad():
        for i in range(start_idx, end_idx):
            if i % 100 == 0 and i > 0:
                logger.info(f"Loaded {i - start_idx} activations")
                gc.collect()
                torch.cuda.empty_cache()
            
            filename = f'{i}.pt'
            filepath = os.path.join(activations_dir, filename)
            
            if os.path.exists(filepath):
                # Load tuple of (token_ids, activations)
                token_ids, activations = torch.load(filepath)
                
                # Extract specific layer if requested
                if layer_idx is not None:
                    activations = activations[layer_idx:layer_idx+1]  # Keep dimension: (1, seq_len, d_model)
                
                # Move to CPU to save GPU memory
                token_ids = token_ids.to('cpu')
                activations = activations.to('cpu')
                
                activations_list.append(ActivationData(token_ids, activations))
    
    if not activations_list:
        raise ValueError(f"No activations loaded in range [{start_idx}, {end_idx})")
    
    logger.info(f"Successfully loaded {len(activations_list)} activations")
    return activations_list


class TokensRemainingDataset(Dataset):
    """
    Dataset for predicting tokens remaining from activations.
    
    Each sample consists of:
    - activation: tensor of shape (d_model,) representing the activation at a specific position
    - tokens_remaining: int representing how many tokens remain after this position
    - total_tokens: int representing the total number of tokens in the sequence
    """
    
    def __init__(self, activations_data: List[ActivationData], layer_idx: int = 0):
        """
        Args:
            activations_data: List of ActivationData objects
            layer_idx: Which layer to use from the activations (default: 0 if single layer, else first layer)
        """
        self.data: List[Tuple[torch.Tensor, int, int]] = []
        
        for act_data in activations_data:
            # activations shape: (n_layers, seq_len, d_model)
            seq_len = act_data.activations.shape[1]
            
            for pos in range(seq_len):
                # Extract activation at this position
                activation = act_data.activations[layer_idx, pos, :]  # Shape: (d_model,)
                tokens_remaining = seq_len - pos
                
                self.data.append((activation, tokens_remaining, seq_len))
        
        logger.info(f"Created dataset with {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Float[torch.Tensor, "d_model"], int, int]:
        return self.data[idx]


def prepare_dataloaders(
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    test_size: float = 0.2,
    batch_size: int = 64,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Load activations and prepare train/test dataloaders.
    
    Args:
        activations_dir: Directory containing activation files
        start_idx: Starting index for loading activations
        end_idx: Ending index for loading activations
        layer_idx: Specific layer to extract (if None, loads all layers and uses layer 0)
        test_size: Fraction of data to use for testing
        batch_size: Batch size for dataloaders
        shuffle: Whether to shuffle the data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, test_dataloader, stats_dict)
        stats_dict contains: {'train_size', 'test_size', 'total_samples', 'total_sequences'}
    """
    # Load activations
    activations_data = load_activations(
        activations_dir=activations_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        layer_idx=layer_idx
    )
    
    # Train-test split
    train_activations, test_activations = train_test_split(
        activations_data,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Split into {len(train_activations)} train and {len(test_activations)} test sequences")
    
    # Create datasets
    dataset_layer_idx = 0  # Use first layer (or the only layer if layer_idx was specified)
    train_dataset = TokensRemainingDataset(train_activations, layer_idx=dataset_layer_idx)
    test_dataset = TokensRemainingDataset(test_activations, layer_idx=dataset_layer_idx)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stats = {
        'train_sequences': len(train_activations),
        'test_sequences': len(test_activations),
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'total_sequences': len(activations_data)
    }
    
    logger.info(f"Created dataloaders - Train samples: {stats['train_samples']}, Test samples: {stats['test_samples']}")
    
    return train_dataloader, test_dataloader, stats

'''
Stage 4: Train probe to predict tokens remaining
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any, List
import logging
import os
from pathlib import Path
from pydantic import BaseModel
from jaxtyping import Float, Int

from pipeline.load_activations import prepare_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingStats(BaseModel):
    """Statistics from probe training"""
    train_losses: List[float]
    test_losses: List[float]
    final_train_loss: Optional[float]
    final_test_loss: Optional[float]
    n_epochs: int
    n_bins: int
    learning_rate: float
    input_dim: int
    train_sequences: int
    test_sequences: int
    train_samples: int
    test_samples: int
    total_sequences: int


class LogBinClassifier(nn.Module):
    """
    Classifier that predicts tokens remaining using logarithmic binning.
    
    The model bins the target values (tokens remaining) into logarithmic bins
    to handle the wide range of possible values more effectively.
    """
    
    def __init__(self, input_dim: int, n_bins: int = 11):
        """
        Args:
            input_dim: Dimension of input activations
            n_bins: Number of logarithmic bins for classification
        """
        super().__init__()
        self.n_bins = n_bins
        self.linear = nn.Linear(input_dim, n_bins, dtype=torch.bfloat16)

    def forward(self, x: Float[torch.Tensor, "batch d_model"]) -> Float[torch.Tensor, "batch n_bins"]:
        return self.linear(x)


def bin_targets(y: Int[torch.Tensor, "batch"], n_bins: int = 11) -> Int[torch.Tensor, "batch"]:
    """
    Convert continuous target values to logarithmic bins using natural log (base e).
    
    Binning formula: floor(ln(y + 1)), clamped to [0, n_bins - 1]
    
    Args:
        y: Target values (tokens remaining)
        n_bins: Number of bins to use
        
    Returns:
        Binned target values as long tensor
    """
    return (y + 1).log().floor().clamp(0, n_bins - 1).to('cuda', dtype=torch.long)


def calculate_class_weights(train_dataloader: DataLoader, n_bins: int = 11) -> Float[torch.Tensor, "n_bins"]:
    """
    Calculate inverse frequency weights for class balancing.
    
    Args:
        train_dataloader: Training data loader
        n_bins: Number of bins
        
    Returns:
        Weight tensor for loss function
    """
    logger.info("Calculating class weights for balanced training")
    
    class_counts = torch.zeros(n_bins)
    total_samples = 0

    for X, y, _ in train_dataloader:
        y_binned = bin_targets(y, n_bins)
        for i in range(n_bins):
            class_counts[i] += (y_binned == i).sum().item()
        total_samples += len(y)

    # Calculate inverse frequency weights
    weight = total_samples / (n_bins * class_counts)
    weight = weight.to('cuda', dtype=torch.bfloat16)
    
    logger.info(f"Class frequencies: {class_counts}")
    logger.info(f"Class weights: {weight}")
    
    return weight


def train_probe(
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    test_size: float = 0.2,
    batch_size: int = 64,
    n_bins: int = 11,
    learning_rate: float = 0.0001,
    n_epochs: int = 1,
    save_path: Optional[str] = None,
    random_state: int = 42
) -> Tuple[LogBinClassifier, TrainingStats]:
    """
    Train a probe to predict tokens remaining from activations.
    
    Args:
        activations_dir: Directory containing activation files
        start_idx: Starting index for loading activations
        end_idx: Ending index for loading activations
        layer_idx: Specific layer to extract (if None, uses layer 0)
        test_size: Fraction of data to use for testing
        batch_size: Batch size for training
        n_bins: Number of logarithmic bins for classification
        learning_rate: Learning rate for optimizer
        n_epochs: Number of training epochs
        save_path: Path to save trained model weights (optional)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, training_stats)
        training_stats contains loss history and final metrics
        
    Raises:
        FileNotFoundError: If activations_dir doesn't exist
        ValueError: If no activations are found
    """
    logger.info("Starting probe training")
    
    # Prepare data loaders
    train_dataloader, test_dataloader, data_stats = prepare_dataloaders(
        activations_dir=activations_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        layer_idx=layer_idx,
        test_size=test_size,
        batch_size=batch_size,
        shuffle=True,
        random_state=random_state
    )
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_dataloader))
    input_dim = sample_batch[0].shape[1]  # (batch_size, d_model)
    
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Using {n_bins} logarithmic bins for classification")
    
    # Initialize model
    model = LogBinClassifier(input_dim, n_bins).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Calculate class weights for balanced training
    class_weights = calculate_class_weights(train_dataloader, n_bins)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    logger.info(f"Starting training for {n_epochs} epochs")
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()
        
        for i, (X, y, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(X.to('cuda'))
            y_binned = bin_targets(y, n_bins)
            loss = loss_fn(preds, y_binned)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # Calculate test loss periodically
            if i % 10 == 0:
                model.eval()
                test_loss_sum = 0
                test_batches = 0
                
                with torch.no_grad():
                    for X_test, y_test, _ in test_dataloader:
                        preds_test = model(X_test.to('cuda'))
                        y_test_binned = bin_targets(y_test, n_bins)
                        test_loss = loss_fn(preds_test, y_test_binned)
                        test_loss_sum += test_loss.item()
                        test_batches += 1
                
                avg_test_loss = test_loss_sum / test_batches
                test_losses.append(avg_test_loss)
                model.train()
                
                if i % 10 == 0:
                    logger.info(f"Batch {i}, Train Loss: {loss.item():.4f}, Test Loss: {avg_test_loss:.4f}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.linear.weight, save_path)
        logger.info(f"Model weights saved to: {save_path}")
    
    # Compile training statistics
    training_stats = TrainingStats(
        train_losses=train_losses,
        test_losses=test_losses,
        final_train_loss=train_losses[-1] if train_losses else None,
        final_test_loss=test_losses[-1] if test_losses else None,
        n_epochs=n_epochs,
        n_bins=n_bins,
        learning_rate=learning_rate,
        input_dim=input_dim,
        **data_stats
    )
    
    logger.info("Training completed successfully")
    logger.info(f"Final train loss: {training_stats.final_train_loss:.4f}")
    logger.info(f"Final test loss: {training_stats.final_test_loss:.4f}")
    
    return model, training_stats
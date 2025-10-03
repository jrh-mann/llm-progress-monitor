'''
Stage 2: Store activations
'''
import nnsight
import torch
import json
import os
import gc
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from pydantic import ValidationError
from transformers import AutoTokenizer, PreTrainedTokenizer

from pipeline.types import RolloutResponse, ChatMessage, FormattedResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_validate_responses(rollouts_path: str) -> List[RolloutResponse]:
    """
    Load and validate rollout responses from JSON file.
    
    Args:
        rollouts_path: Path to JSON file containing rollout responses
        
    Returns:
        List of validated RolloutResponse objects
        
    Raises:
        FileNotFoundError: If rollouts_path doesn't exist
        ValueError: If response data is invalid
    """
    if not os.path.exists(rollouts_path):
        raise FileNotFoundError(f"Rollouts file not found: {rollouts_path}")
    
    logger.info(f"Loading rollouts from: {rollouts_path}")
    with open(rollouts_path, 'r') as f:
        raw_data = json.load(f)
    
    if not raw_data:
        raise ValueError(f"No responses found in {rollouts_path}")
    
    # Validate each response using Pydantic
    try:
        responses: List[RolloutResponse] = [RolloutResponse(**item) for item in raw_data]
    except ValidationError as e:
        logger.error(f"Invalid response data in {rollouts_path}")
        raise ValueError(f"Response validation failed: {e}") from e
    
    logger.info(f"Loaded and validated {len(responses)} responses")
    return responses


def format_responses_with_chat_template(
    responses: List[RolloutResponse],
    tokenizer: PreTrainedTokenizer
) -> List[FormattedResponse]:
    """
    Format responses with chat template.
    
    Args:
        responses: List of validated rollout responses
        tokenizer: Tokenizer to use for chat template formatting
        
    Returns:
        List of formatted responses with chat template applied
    """
    formatted_responses: List[FormattedResponse] = []
    
    for response in responses:
        # Apply chat template format
        messages: List[ChatMessage] = [
            {"role": "user", "content": response.instruction},
            {"role": "assistant", "content": response.response}
        ]
        
        chat_formatted_result = tokenizer.apply_chat_template(
            messages,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=False
        )
        
        if not isinstance(chat_formatted_result, str):
            raise TypeError(f"Expected string from apply_chat_template, got {type(chat_formatted_result)}")
        
        formatted_item: FormattedResponse = {
            'instruction': response.instruction,
            'response': response.response,
            'chat_formatted': chat_formatted_result,
            'char_length': response.char_length,
            'tokens_length': response.tokens_length
        }
        formatted_responses.append(formatted_item)
    
    logger.info(f"Formatted {len(formatted_responses)} responses with chat template")
    return formatted_responses

import threading
import queue
from concurrent.futures import ThreadPoolExecutor


def _save_activations_worker(save_queue: queue.Queue) -> None:
    """Worker function to save activations from queue to disk."""
    while True:
        item = save_queue.get()
        if item is None:  # Sentinel to stop worker
            break
        
        save_path, token_ids, activations = item
        try:
            # Save as tuple of (token_ids, activations)
            torch.save((token_ids, activations), save_path)
            logger.info(f"Saved activations to {save_path}")
        except Exception as e:
            logger.error(f"Error saving activations to {save_path}: {e}")
        finally:
            save_queue.task_done()


def store_activations(
    model_name: str,
    rollouts_path: str,
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    layer_idx: Optional[int] = None
) -> None:
    """
    Extract and store activations from model rollouts (simplified version).
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-4B")
        rollouts_path: Path to JSON file containing rollout responses
        activations_dir: Directory to save activation tensors (defaults to './rollouts/activations')
        start_idx: Starting index for processing responses (defaults to 0)
        end_idx: Ending index for processing responses (defaults to None, processes all)
        dtype: Data type for model (defaults to torch.bfloat16)
        device_map: Device mapping strategy (defaults to "auto")
        layer_idx: Specific layer index to extract activations from (defaults to None, extracts all layers)
        
    Raises:
        FileNotFoundError: If rollouts_path doesn't exist
        ValueError: If response data is invalid
    """
    # Load and validate responses
    responses = load_and_validate_responses(rollouts_path)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Format responses with chat template
    formatted_responses = format_responses_with_chat_template(responses, tokenizer)
    
    # Create activations directory
    os.makedirs(activations_dir, exist_ok=True)
    logger.info(f"Activations will be saved to: {activations_dir}")
    
    # Load model with nnsight
    logger.info(f"Loading model: {model_name}")
    model = nnsight.LanguageModel(model_name, device_map=device_map, dtype=dtype)
    
    # Determine range to process
    end_idx = end_idx if end_idx is not None else len(formatted_responses)
    responses_to_process = formatted_responses[start_idx:end_idx]
    
    logger.info(f"Processing {len(responses_to_process)} responses (indices {start_idx} to {end_idx})")
    
    # Create queue for saving activations and start worker thread
    save_queue: queue.Queue = queue.Queue(maxsize=10)  # Limit queue size to prevent excessive memory usage
    save_thread = threading.Thread(target=_save_activations_worker, args=(save_queue,))
    save_thread.start()
    
    try:
        # Process one at a time (simple and reliable)
        with torch.no_grad():
            for i, formatted_response in enumerate(responses_to_process):
                actual_idx = start_idx + i
                
                # Clear GPU memory
                gc.collect()
                torch.cuda.empty_cache()
                
                chat_formatted = formatted_response['chat_formatted']
                
                try:
                    # Tokenize to get token IDs (do this first, outside trace)
                    token_ids = tokenizer.encode(chat_formatted, return_tensors='pt')[0]
                    
                    # Extract activations using nnsight trace
                    with model.trace(chat_formatted):
                        layer_outputs = []
                        
                        if layer_idx is not None:
                            # Extract single layer
                            num_layers = len(model.model.layers)  # type: ignore[attr-defined]
                            if layer_idx >= num_layers or layer_idx < 0:
                                raise ValueError(f"layer_idx {layer_idx} is out of range. Model has {num_layers} layers (0-{num_layers-1})")
                            layer_outputs.append(model.model.layers[layer_idx].output[0])  # type: ignore[attr-defined]
                        else:
                            # Extract all layers
                            for layer in model.model.layers:  # type: ignore[attr-defined]
                                layer_outputs.append(layer.output[0])
                        
                        # Stack: (num_layers, seq, d_model)
                        activations = torch.stack(layer_outputs, dim=0)
                        
                        # Move to CPU and add to save queue
                        token_ids_cpu = token_ids.cpu()
                        activations_cpu = activations.cpu()
                        save_path = Path(activations_dir) / f"{actual_idx}.pt"
                        
                        # Put in queue for background saving as tuple (token_ids, activations)
                        save_queue.put((save_path, token_ids_cpu, activations_cpu))
                        logger.info(f"Queued activations for response {actual_idx}")
                    
                except Exception as e:
                    logger.error(f"Error on response {actual_idx}: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
        
        # Wait for all saves to complete
        save_queue.join()
        logger.info("All activations processed and queued for saving")
        
    finally:
        # Stop the save worker thread
        save_queue.put(None)  # Sentinel to stop worker
        save_thread.join()
    
    logger.info("Activation storage complete!")

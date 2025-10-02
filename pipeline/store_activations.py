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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

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


def save_activation_worker(save_queue: queue.Queue, activations_dir: str) -> None:
    """
    Worker function to save activations from a queue.
    
    Args:
        save_queue: Queue containing save tasks
        activations_dir: Directory to save activations
    """
    while True:
        try:
            task = save_queue.get(timeout=1)
            if task is None:  # Sentinel value to stop worker
                break
                
            actual_idx, token_ids_no_padding, item_activations_no_padding = task
            save_path = Path(activations_dir) / f"{actual_idx}.pt"
            
            logger.info(f"Saving activations for response {actual_idx} (shape: {item_activations_no_padding.shape})")
            torch.save((token_ids_no_padding, item_activations_no_padding), save_path)
            logger.info(f"Saved activations for response {actual_idx} (shape: {item_activations_no_padding.shape})")
            
            save_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error saving activation {actual_idx}: {e}")
            save_queue.task_done()


def store_activations(
    model_name: str,
    rollouts_path: str,
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    num_save_workers: int = 4,
    layer_idx: Optional[int] = None
) -> None:
    """
    Extract and store activations from model rollouts with pipelined processing.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-4B")
        rollouts_path: Path to JSON file containing rollout responses
        activations_dir: Directory to save activation tensors (defaults to './rollouts/activations')
        start_idx: Starting index for processing responses (defaults to 0)
        end_idx: Ending index for processing responses (defaults to None, processes all)
        batch_size: Number of responses to process in parallel (defaults to 1)
        dtype: Data type for model (defaults to torch.bfloat16)
        device_map: Device mapping strategy (defaults to "auto")
        num_save_workers: Number of worker threads for saving (defaults to 4)
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
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    if layer_idx is not None:
        logger.info(f"Processing {len(responses_to_process)} responses (indices {start_idx} to {end_idx}) with batch_size={batch_size}, extracting layer {layer_idx} only")
    else:
        logger.info(f"Processing {len(responses_to_process)} responses (indices {start_idx} to {end_idx}) with batch_size={batch_size}, extracting all layers")
    
    # Set up pipelined processing with save queue and workers
    save_queue: queue.Queue = queue.Queue(maxsize=num_save_workers * 2)  # Buffer for smooth pipeline
    
    # Start save worker threads
    save_workers = []
    for i in range(num_save_workers):
        worker = threading.Thread(target=save_activation_worker, args=(save_queue, activations_dir))
        worker.daemon = True
        worker.start()
        save_workers.append(worker)
    
    logger.info(f"Started {num_save_workers} save worker threads")
    
    # Extract and save activations in batches with pipelining
    with torch.no_grad():
        for batch_start in range(0, len(responses_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(responses_to_process))
            batch = responses_to_process[batch_start:batch_end]
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Use pre-formatted chat templates instead of re-formatting
            batch_texts = [item['chat_formatted'] for item in batch]
            
            # Tokenize batch with padding
            batch_encodings = tokenizer(
                batch_texts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False  # We need to add padding
            )
            
            # Extract activations
            try:
                with model.trace(batch_texts):
                    logger.info("Starting activation extraction...")
                    
                    # Get dimensions from first layer
                    logger.info("Getting dimensions from first layer...")
                    first_layer_output = model.model.layers[0].output  # type: ignore[attr-defined]
                    batch_size_actual, seq_len, d_model = first_layer_output.shape
                    
                    if layer_idx is not None:
                        num_layers = len(model.model.layers)  # type: ignore[attr-defined]
                        if layer_idx >= num_layers or layer_idx < 0:
                            raise ValueError(f"layer_idx {layer_idx} is out of range. Model has {num_layers} layers (0-{num_layers-1})")
                        
                        logger.info(f"Dimensions: batch_size={batch_size_actual}, seq_len={seq_len}, d_model={d_model}, extracting layer {layer_idx}")
                        
                        # Extract activations from specified layer only - no pre-allocation
                        logger.info(f"Extracting activations from layer {layer_idx}...")
                        target_layer = model.model.layers[layer_idx]  # type: ignore[attr-defined]
                        # Shape: (batch, seq, d_model) -> add layer dimension: (batch, 1, seq, d_model)
                        activations = target_layer.output.unsqueeze(1)
                    else:
                        num_layers = len(model.model.layers)  # type: ignore[attr-defined]
                        logger.info(f"Dimensions: batch_size={batch_size_actual}, seq_len={seq_len}, d_model={d_model}, num_layers={num_layers}")
                        
                        # Pre-allocate tensor for all layers: (batch, layer, seq, d_model)
                        logger.info("Pre-allocating activations tensor for all layers...")
                        activations = torch.zeros(batch_size_actual, num_layers, seq_len, d_model, dtype=first_layer_output.dtype, device=first_layer_output.device)
                        logger.info(f"Pre-allocated tensor shape: {activations.shape}")
                        
                        # Fill tensor directly from all layers
                        logger.info("Extracting activations from all layers...")
                        for layer_idx_iter, layer in enumerate(model.model.layers):  # type: ignore[attr-defined]
                            logger.info(f"Extracting from layer {layer_idx_iter}/{num_layers-1}")
                            # Extract activations: layer.output has shape (batch, seq, d_model)
                            activations[:, layer_idx_iter, :, :] = layer.output
                    
                    logger.info("Activation extraction complete!")
                    
                    # Queue activations for saving (pipelined with next batch extraction)
                    for i in range(len(batch)):
                        actual_idx = start_idx + batch_start + i
                        
                        # Get token IDs for this specific item
                        token_ids = batch_encodings['input_ids'][i]
                        
                        # Get attention mask to identify non-padding tokens
                        attention_mask = batch_encodings['attention_mask'][i]
                        
                        # Find the indices of non-padding tokens
                        non_padding_indices = attention_mask.nonzero(as_tuple=True)[0]
                        
                        # Extract only non-padding token IDs
                        token_ids_no_padding = token_ids[non_padding_indices]
                        
                        # Get activations for this item: (layer, seq, d_model) or (1, seq, d_model)
                        item_activations = activations[i]
                        
                        # Extract only activations for non-padding tokens: (layer, non_padding_seq, d_model) or (1, non_padding_seq, d_model)
                        item_activations_no_padding = item_activations[:, non_padding_indices, :]
                        
                        # Move to CPU to free GPU memory before queuing for save
                        token_ids_cpu = token_ids_no_padding.cpu()
                        activations_cpu = item_activations_no_padding.cpu()
                        
                        # Queue for saving (this will block if queue is full, providing backpressure)
                        save_queue.put((actual_idx, token_ids_cpu, activations_cpu))
                    
            except Exception as e:
                logger.error(f"Error on batch starting at {start_idx + batch_start}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
    
    # Wait for all saves to complete
    logger.info("Waiting for all saves to complete...")
    save_queue.join()
    
    # Stop save workers
    for _ in range(num_save_workers):
        save_queue.put(None)  # Sentinel value to stop workers
    
    for worker in save_workers:
        worker.join()
    
    logger.info("Activation storage complete!")
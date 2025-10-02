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


def store_activations(
    model_name: str,
    rollouts_path: str,
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto"
) -> None:
    """
    Extract and store activations from model rollouts.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-4B")
        rollouts_path: Path to JSON file containing rollout responses
        activations_dir: Directory to save activation tensors (defaults to './rollouts/activations')
        start_idx: Starting index for processing responses (defaults to 0)
        end_idx: Ending index for processing responses (defaults to None, processes all)
        batch_size: Number of responses to process in parallel (defaults to 1)
        dtype: Data type for model (defaults to torch.bfloat16)
        device_map: Device mapping strategy (defaults to "auto")
        
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
    
    logger.info(f"Processing {len(responses_to_process)} responses (indices {start_idx} to {end_idx}) with batch_size={batch_size}")
    
    # Extract and save activations in batches
    with torch.no_grad():
        for batch_start in range(0, len(responses_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(responses_to_process))
            batch = responses_to_process[batch_start:batch_end]
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Prepare batch inputs by formatting with chat template
            batch_texts = []
            for item in batch:
                # Format the instruction and response using the chat template
                messages = [
                    {"role": "user", "content": item['instruction']},
                    {"role": "assistant", "content": item['response']}
                ]
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                batch_texts.append(formatted_text)
            
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
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Get dimensions from first layer to pre-allocate tensor
                    first_layer_output = model.model.layers[0].output  # type: ignore[attr-defined]
                    batch_size_actual, seq_len, d_model = first_layer_output.shape
                    num_layers = len(model.model.layers)  # type: ignore[attr-defined]
                    
                    # Pre-allocate tensor: (batch, layer, seq, d_model)
                    activations = torch.zeros(batch_size_actual, num_layers, seq_len, d_model, dtype=first_layer_output.dtype, device=first_layer_output.device)
                    
                    # Fill tensor directly
                    for layer_idx, layer in enumerate(model.model.layers):  # type: ignore[attr-defined]
                        # Extract activations: layer.output has shape (batch, seq, d_model)
                        activations[:, layer_idx, :, :] = layer.output
                    
                    # Prepare data for parallel saving
                    save_tasks = []
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
                        
                        # Get activations for this item: (layer, seq, d_model)
                        item_activations = activations[i]
                        
                        # Extract only activations for non-padding tokens: (layer, non_padding_seq, d_model)
                        item_activations_no_padding = item_activations[:, non_padding_indices, :]
                        
                        # Prepare save task
                        save_path = Path(activations_dir) / f"{actual_idx}.pt"
                        save_tasks.append((actual_idx, token_ids_no_padding, item_activations_no_padding, save_path))
                    
                    # Save in parallel using ThreadPoolExecutor
                    from concurrent.futures import ThreadPoolExecutor
                    
                    def save_activation_item(task_data):
                        actual_idx, token_ids_no_padding, item_activations_no_padding, save_path = task_data
                        logger.info(f"Saving activations for response {actual_idx} (shape: {item_activations_no_padding.shape})")
                        torch.save((token_ids_no_padding, item_activations_no_padding), save_path)
                        logger.info(f"Saved activations for response {actual_idx} (shape: {item_activations_no_padding.shape})")
                    
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        executor.map(save_activation_item, save_tasks)
                    
            except Exception as e:
                logger.error(f"Error on batch starting at {start_idx + batch_start}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
    
    logger.info("Activation storage complete!")
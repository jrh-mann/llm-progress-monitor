'''
Stage 1: Generate prompt rollouts
'''
import vllm
import torch
import json
import os
import logging
from typing import List, Dict, Optional, Union, Sequence, cast, Any, Tuple
from pathlib import Path
from pydantic import ValidationError

from pipeline.types import PromptData, ChatMessage, RolloutResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_prompt_with_chat_template(
    prompt: str, 
    tokenizer: Any, 
    enable_thinking: bool
) -> str:
    """
    Format a single prompt using the chat template.
    
    Args:
        prompt: The instruction text to format
        tokenizer: The tokenizer with chat template support
        enable_thinking: Whether to enable thinking mode
        
    Returns:
        Formatted prompt string ready for generation
    """
    message: ChatMessage = {"role": "user", "content": prompt}
    
    result = tokenizer.apply_chat_template(
        [message],  # type: ignore[arg-type]
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=enable_thinking
    )
    
    if isinstance(result, str):
        return result
    else:
        raise TypeError(f"Expected string from apply_chat_template, got {type(result)}")


def prepare_prompts_for_generation(
    prompts_to_generate: List[PromptData],
    tokenizer: Any,
    thinking_mode: int
) -> Tuple[List[str], List[str]]:
    """
    Prepare prompts for generation based on thinking mode.
    
    Args:
        prompts_to_generate: List of prompt data to process
        tokenizer: The tokenizer with chat template support
        thinking_mode: 0 (no thinking), 1 (thinking), or 2 (both)
        
    Returns:
        Tuple of (formatted_instructions, original_instructions)
    """
    formatted_instructions: List[str] = []
    original_instructions: List[str] = []
    
    for prompt_dict in prompts_to_generate:
        if thinking_mode == 0:
            # Only non-thinking version
            original_instructions.append(prompt_dict.instruction)
            formatted_instruction = format_prompt_with_chat_template(
                prompt_dict.instruction, tokenizer, enable_thinking=False
            )
            formatted_instructions.append(formatted_instruction)
            
        elif thinking_mode == 1:
            # Only thinking version
            original_instructions.append(prompt_dict.instruction)
            formatted_instruction = format_prompt_with_chat_template(
                prompt_dict.instruction, tokenizer, enable_thinking=True
            )
            formatted_instructions.append(formatted_instruction)
            
        elif thinking_mode == 2:
            # Both versions - non-thinking first, then thinking
            original_instructions.extend([prompt_dict.instruction, prompt_dict.instruction])
            
            formatted_no_thinking = format_prompt_with_chat_template(
                prompt_dict.instruction, tokenizer, enable_thinking=False
            )
            formatted_thinking = format_prompt_with_chat_template(
                prompt_dict.instruction, tokenizer, enable_thinking=True
            )
            
            formatted_instructions.extend([formatted_no_thinking, formatted_thinking])
    
    return formatted_instructions, original_instructions


def process_outputs_to_responses(
    outputs: List[vllm.RequestOutput],
    original_instructions: List[str]
) -> List[RolloutResponse]:
    """
    Convert vLLM outputs to structured RolloutResponse objects.
    
    Args:
        outputs: List of vLLM RequestOutput objects
        original_instructions: List of original instruction strings
        
    Returns:
        List of RolloutResponse objects
    """
    return [
        RolloutResponse(
            instruction=original_instruction,
            response=output.outputs[0].text,
            char_length=len(output.outputs[0].text),
            tokens_length=len(output.outputs[0].token_ids)
        )
        for output, original_instruction in zip(outputs, original_instructions)
    ]


def save_responses_to_file(
    responses: List[RolloutResponse],
    save_path: str,
    model_name: str,
    thinking_mode: int
) -> None:
    """
    Save responses to a JSON file.
    
    Args:
        responses: List of RolloutResponse objects to save
        save_path: Directory to save the file
        model_name: Name of the model used for generation
        thinking_mode: The thinking mode used (for filename)
    """
    os.makedirs(save_path, exist_ok=True)
    model_short_name = model_name.split("/")[-1]
    output_file = Path(save_path) / f"{model_short_name}-{thinking_mode}.json"
    
    logger.info(f"Saving {len(responses)} responses to {output_file}")
    with open(output_file, 'w') as f:
        # Convert Pydantic models to dicts for JSON serialization
        json.dump([r.model_dump() for r in responses], f, indent=2)


def generate_rollouts(
    model_name: str,
    prompts_path: str,
    num_rollouts: int = 1000,
    max_tokens: int = 32768,
    save_path: str = '/workspace/llm-progress-monitor/rollouts',
    thinking_mode: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[RolloutResponse]:
    """
    Generate text completions (rollouts) for a set of prompts using vLLM.

    Args:
        model_name: HuggingFace model identifier (e.g., "google/gemma-2-2b-it")
        prompts_path: Path to JSON file containing prompts with 'instruction' field
        num_rollouts: Number of rollouts to generate (defaults to 1000)
        max_tokens: Maximum tokens to generate per rollout (defaults to 32768)
        save_path: Directory to save rollout results (defaults to './rollouts')
        thinking_mode: Thinking mode - 0: disabled, 1: enabled, 2: both (defaults to 1)
        temperature: Sampling temperature (defaults to 1.0)
        top_p: Nucleus sampling parameter (defaults to 1.0)

    Returns:
        List of RolloutResponse dictionaries containing instructions and responses

    Raises:
        FileNotFoundError: If prompts_path doesn't exist
        ValueError: If prompts data is invalid
    """
    # Validate inputs
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    if num_rollouts <= 0:
        raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
    
    if thinking_mode not in [0, 1, 2]:
        raise ValueError(f"thinking_mode must be 0, 1, or 2, got {thinking_mode}")

    # Load the model using vLLM
    logger.info(f"Loading model: {model_name}")
    llm = vllm.LLM(
        model=model_name,
    )

    # Load prompts from JSON file with validation
    logger.info(f"Loading prompts from: {prompts_path}")
    with open(prompts_path, 'r') as f:
        raw_data = json.load(f)
    
    if not raw_data:
        raise ValueError(f"No prompts found in {prompts_path}")
    
    # Validate each prompt using Pydantic
    try:
        prompts_data: List[PromptData] = [PromptData(**item) for item in raw_data]
    except ValidationError as e:
        logger.error(f"Invalid prompt data in {prompts_path}")
        raise ValueError(f"Prompt validation failed: {e}") from e

    # Get the tokenizer to apply chat template
    tokenizer = llm.get_tokenizer()

    # Determine how many to generate based on thinking mode
    if thinking_mode == 2:
        # For mode 2, we need half the prompts since we'll generate twice per prompt
        num_to_generate = min(num_rollouts // 2, len(prompts_data))
    else:
        num_to_generate = min(num_rollouts, len(prompts_data))
    
    logger.info(f"Generating rollouts for {num_to_generate} prompts (thinking_mode={thinking_mode}, max_tokens={max_tokens})")
    
    # Filter to only the prompts we'll generate
    prompts_to_generate = prompts_data[:num_to_generate]
    
    # Prepare prompts for generation
    formatted_instructions, original_instructions = prepare_prompts_for_generation(
        prompts_to_generate, tokenizer, thinking_mode
    )

    logger.info(f"Formatted {len(formatted_instructions)} prompts with chat template")

    # Generate completions using vLLM
    sampling_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    outputs: List[vllm.RequestOutput] = llm.generate(
        formatted_instructions, 
        sampling_params=sampling_params
    )

    # Process outputs into structured responses
    responses = process_outputs_to_responses(outputs, original_instructions)

    # Save results to JSON
    save_responses_to_file(responses, save_path, model_name, thinking_mode)

    logger.info(f"Rollout generation complete!")
    return responses
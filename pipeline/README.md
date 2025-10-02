# Pipeline Documentation

This document provides a comprehensive overview of the LLM Progress Monitor pipeline, designed for contributors who want to modify or extend the codebase for their own projects.

## Table of Contents
- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Stage 1: Generate Prompt Rollouts](#stage-1-generate-prompt-rollouts)
- [Stage 2: Store Activations](#stage-2-store-activations)
- [Stage 3: Load Activations](#stage-3-load-activations)
- [Stage 4: Train Probe](#stage-4-train-probe)
- [Type Definitions](#type-definitions)
- [Data Formats](#data-formats)
- [Common Modifications](#common-modifications)

---

## Overview

This pipeline extracts and analyzes internal model representations to train probes that predict tokens remaining in LLM generations. The core hypothesis is that language models maintain internal representations of progress through their response generation, which can be linearly decoded.

**Key Research Question**: Can we detect how far through its response an LLM believes it is by examining its internal activations?

**Pipeline Flow**:
```
Prompts (JSON) → [Stage 1] → Rollouts (JSON) → [Stage 2] → Activations (.pt files)
                                                              ↓
                                                         [Stage 3]
                                                              ↓
                                                      Train/Test Split
                                                              ↓
                                                         [Stage 4]
                                                              ↓
                                                      Trained Probe
```

---

## Pipeline Architecture

The pipeline consists of four sequential stages:

1. **Generate Prompt Rollouts**: Generate completions from an LLM using vLLM for efficiency
2. **Store Activations**: Extract and save layer activations using nnsight
3. **Load Activations**: Load saved activations and prepare datasets for training
4. **Train Probe**: Train a linear probe to predict tokens remaining from activations

Each stage is independent and can be run separately, with intermediate results saved to disk.

---

## Stage 1: Generate Prompt Rollouts

**File**: `generate_prompt_rollouts.py`

### Conceptual Overview

**What**: Generates text completions from an LLM for a set of input prompts.

**Why**: We need diverse, long-form responses to analyze how models represent progress internally. Using vLLM allows efficient batch generation.

**How**: 
- Loads prompts from JSON
- Formats them using the model's chat template
- Generates completions using vLLM with specified sampling parameters
- Saves instruction-response pairs with metadata (length in chars and tokens)

### Key Features

- **Thinking Mode**: Supports three modes for models with thinking capabilities (e.g., Qwen):
  - `0`: No thinking (standard generation)
  - `1`: With thinking (enable thinking mode)
  - `2`: Both (generates two completions per prompt)
- **Chat Template**: Automatically applies model-specific chat formatting
- **Validation**: Uses Pydantic for runtime validation of prompts and responses

### API Reference

#### `generate_rollouts()`

Main entry point for Stage 1.

```python
def generate_rollouts(
    model_name: str,
    prompts_path: str,
    num_rollouts: int = 1000,
    max_tokens: int = 32768,
    save_path: str = '/workspace/llm-progress-monitor/rollouts',
    thinking_mode: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[RolloutResponse]
```

**Parameters**:
- `model_name`: HuggingFace model identifier (e.g., `"Qwen/Qwen3-4B"`)
- `prompts_path`: Path to JSON file containing prompts (see [Data Formats](#data-formats))
- `num_rollouts`: Number of rollouts to generate (default: 1000)
- `max_tokens`: Maximum tokens per generation (default: 32768)
- `save_path`: Directory to save results (default: `./rollouts`)
- `thinking_mode`: 0 (disabled), 1 (enabled), or 2 (both) (default: 1)
- `temperature`: Sampling temperature (default: 1.0)
- `top_p`: Nucleus sampling parameter (default: 1.0)

**Returns**: List of `RolloutResponse` objects

**Raises**:
- `FileNotFoundError`: If prompts_path doesn't exist
- `ValueError`: If prompts data is invalid or num_rollouts ≤ 0

**Output**: Saves to `{save_path}/{model_short_name}-{thinking_mode}.json`

#### Helper Functions

##### `format_prompt_with_chat_template()`
```python
def format_prompt_with_chat_template(
    prompt: str, 
    tokenizer: Any, 
    enable_thinking: bool
) -> str
```
Formats a single prompt using the model's chat template.

##### `prepare_prompts_for_generation()`
```python
def prepare_prompts_for_generation(
    prompts_to_generate: List[PromptData],
    tokenizer: Any,
    thinking_mode: int
) -> Tuple[List[str], List[str]]
```
Prepares prompts for generation based on thinking mode. Returns (formatted_instructions, original_instructions).

##### `process_outputs_to_responses()`
```python
def process_outputs_to_responses(
    outputs: List[vllm.RequestOutput],
    original_instructions: List[str]
) -> List[RolloutResponse]
```
Converts vLLM outputs to structured `RolloutResponse` objects.

##### `save_responses_to_file()`
```python
def save_responses_to_file(
    responses: List[RolloutResponse],
    save_path: str,
    model_name: str,
    thinking_mode: int
) -> None
```
Saves responses to a JSON file with standardized naming.

### Example Usage

```python
from pipeline.generate_prompt_rollouts import generate_rollouts

responses = generate_rollouts(
    model_name="Qwen/Qwen3-4B",
    prompts_path="/workspace/llm-progress-monitor/splits/harmful_train.json",
    num_rollouts=1000,
    max_tokens=16384,
    thinking_mode=1,
    temperature=1.0
)
```

---

## Stage 2: Store Activations

**File**: `store_activations.py`

### Conceptual Overview

**What**: Extracts internal layer activations from the model as it processes completed rollouts.

**Why**: Activations contain the model's internal representations at each token position. By extracting these, we can analyze what information the model encodes about its progress.

**How**:
- Loads rollout responses and formats them with chat templates
- Uses `nnsight` to trace through the model and capture layer outputs
- Handles batching and memory management (crucial for large models)
- Saves activations with corresponding token IDs to disk as `.pt` files
- Uses pipelined processing: while saving one batch, the next batch is being extracted

### Key Features

- **Pipelined Processing**: Multi-threaded saving overlaps with extraction for efficiency
- **Memory Management**: Strips padding, moves to CPU immediately, aggressive garbage collection
- **Layer Selection**: Can extract all layers or a specific layer only
- **Resume Support**: Can process subsets using `start_idx` and `end_idx`

### API Reference

#### `store_activations()`

Main entry point for Stage 2.

```python
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
) -> None
```

**Parameters**:
- `model_name`: HuggingFace model identifier
- `rollouts_path`: Path to JSON file from Stage 1
- `activations_dir`: Directory to save activation tensors (default: `./rollouts/activations`)
- `start_idx`: Starting index for processing (default: 0)
- `end_idx`: Ending index for processing (default: None = all)
- `batch_size`: Number of responses to process in parallel (default: 1)
- `dtype`: Model data type (default: `torch.bfloat16`)
- `device_map`: Device mapping strategy (default: `"auto"`)
- `num_save_workers`: Number of worker threads for saving (default: 4)
- `layer_idx`: Specific layer to extract (default: None = all layers)

**Returns**: None (saves to disk)

**Raises**:
- `FileNotFoundError`: If rollouts_path doesn't exist
- `ValueError`: If response data is invalid or layer_idx out of range

**Output**: Creates `{activations_dir}/{idx}.pt` files, each containing:
```python
(token_ids, activations)
# token_ids: Tensor of shape (seq_len,)
# activations: Tensor of shape (n_layers, seq_len, d_model) or (1, seq_len, d_model) if layer_idx specified
```

#### Helper Functions

##### `load_and_validate_responses()`
```python
def load_and_validate_responses(rollouts_path: str) -> List[RolloutResponse]
```
Loads and validates rollout responses using Pydantic.

##### `format_responses_with_chat_template()`
```python
def format_responses_with_chat_template(
    responses: List[RolloutResponse],
    tokenizer: PreTrainedTokenizer
) -> List[FormattedResponse]
```
Formats responses with chat template for consistent processing.

##### `save_activation_worker()`
```python
def save_activation_worker(save_queue: queue.Queue, activations_dir: str) -> None
```
Worker function that runs in a separate thread to save activations from a queue.

### Example Usage

```python
from pipeline.store_activations import store_activations

store_activations(
    model_name="Qwen/Qwen3-4B",
    rollouts_path="/workspace/llm-progress-monitor/rollouts/Qwen3-4B-1.json",
    activations_dir="/workspace/llm-progress-monitor/rollouts/activations",
    batch_size=2,
    layer_idx=10  # Extract only layer 10
)
```

### Performance Considerations

- **Batch Size**: Larger batches use more GPU memory but are faster. Start with 1 for 7B+ models.
- **Save Workers**: More workers reduce I/O bottleneck but use more CPU. 4-8 is typically optimal.
- **Layer Selection**: Extracting a single layer saves ~90% memory compared to all layers.

---

## Stage 3: Load Activations

**File**: `load_activations.py`

### Conceptual Overview

**What**: Loads saved activation tensors and creates PyTorch datasets for probe training.

**Why**: Transforms raw activations into supervised learning data where X = activation at position t, y = tokens remaining after position t.

**How**:
- Loads `.pt` files from disk
- Creates samples for each (position, activation) pair in each sequence
- Computes target labels as `tokens_remaining = seq_len - position`
- Splits data into train/test sets
- Returns PyTorch DataLoaders ready for training

### Key Features

- **Position-Level Samples**: Each token position becomes a training sample
- **Train/Test Split**: Uses sklearn for reproducible splits at the sequence level
- **Lazy Loading Option**: Can load activations incrementally to save memory
- **Layer Selection**: Extract a specific layer from multi-layer activations

### API Reference

#### `prepare_dataloaders()`

Main entry point for Stage 3.

```python
def prepare_dataloaders(
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    test_size: float = 0.2,
    batch_size: int = 64,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]
```

**Parameters**:
- `activations_dir`: Directory containing activation files
- `start_idx`: Starting index for loading (default: 0)
- `end_idx`: Ending index for loading (default: None = all)
- `layer_idx`: Specific layer to extract (default: None = use layer 0)
- `test_size`: Fraction for testing (default: 0.2)
- `batch_size`: Batch size for DataLoaders (default: 64)
- `shuffle`: Whether to shuffle training data (default: True)
- `random_state`: Random seed for reproducibility (default: 42)

**Returns**: `(train_dataloader, test_dataloader, stats_dict)`
- `stats_dict` contains: `{'train_sequences', 'test_sequences', 'train_samples', 'test_samples', 'total_sequences'}`

**Raises**:
- `FileNotFoundError`: If activations_dir doesn't exist
- `ValueError`: If no activations found in range

#### Classes

##### `ActivationData`
```python
class ActivationData:
    def __init__(
        self, 
        token_ids: Int[torch.Tensor, "seq_len"], 
        activations: Float[torch.Tensor, "n_layers seq_len d_model"]
    ):
        self.token_ids = token_ids
        self.activations = activations
```
Container for loaded activation data with shape-annotated tensors.

##### `TokensRemainingDataset`
```python
class TokensRemainingDataset(Dataset):
    def __init__(self, activations_data: List[ActivationData], layer_idx: int = 0)
    
    def __getitem__(self, idx: int) -> Tuple[Float[torch.Tensor, "d_model"], int, int]:
        # Returns: (activation, tokens_remaining, total_tokens)
```
PyTorch Dataset where each sample is one position's activation with its tokens remaining. Returns shape-annotated activation vectors.

#### Helper Functions

##### `load_activations()`
```python
def load_activations(
    activations_dir: str = '/workspace/llm-progress-monitor/rollouts/activations',
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None
) -> List[ActivationData]
```
Loads activation tensors from disk.

### Example Usage

```python
from pipeline.load_activations import prepare_dataloaders

train_loader, test_loader, stats = prepare_dataloaders(
    activations_dir="/workspace/llm-progress-monitor/rollouts/activations",
    start_idx=0,
    end_idx=1000,
    layer_idx=0,  # Use first layer only
    test_size=0.2,
    batch_size=128
)

print(f"Train samples: {stats['train_samples']}")
print(f"Test samples: {stats['test_samples']}")

for activations, tokens_remaining, total_tokens in train_loader:
    # activations: (batch_size, d_model)
    # tokens_remaining: (batch_size,)
    # total_tokens: (batch_size,)
    break
```

---

## Stage 4: Train Probe

**File**: `train_probe.py`

### Conceptual Overview

**What**: Trains a linear probe to predict tokens remaining from activations.

**Why**: Tests whether progress information is linearly represented in the model's activations. A linear probe is the simplest decoder—if it works, the information is explicitly encoded.

**How**:
- Uses logarithmic binning (base e) to discretize the target (tokens remaining) into bins
- Trains a single linear layer to classify activations into bins
- Uses cross-entropy loss with class balancing to handle imbalanced token distributions
- Evaluates on held-out test set

### Key Features

- **Logarithmic Binning**: Bins tokens remaining as `floor(ln(n+1))` using natural logarithm (base e) to handle wide range of values
- **Class Balancing**: Weights loss by inverse class frequency
- **Periodic Evaluation**: Computes test loss every 10 batches during training
- **Model Saving**: Optional save of probe weights

### API Reference

#### `train_probe()`

Main entry point for Stage 4.

```python
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
) -> Tuple[LogBinClassifier, TrainingStats]
```

**Parameters**:
- `activations_dir`: Directory containing activation files
- `start_idx`: Starting index for loading activations
- `end_idx`: Ending index for loading activations
- `layer_idx`: Specific layer to extract (default: None = uses layer 0)
- `test_size`: Fraction for testing (default: 0.2)
- `batch_size`: Batch size for training (default: 64)
- `n_bins`: Number of logarithmic bins for classification (default: 11)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.0001)
- `n_epochs`: Number of training epochs (default: 1)
- `save_path`: Path to save model weights (default: None = don't save)
- `random_state`: Random seed (default: 42)

**Returns**: `(model, training_stats)`
- `model`: Trained `LogBinClassifier` instance
- `training_stats`: `TrainingStats` object with loss history and metrics

**Raises**:
- `FileNotFoundError`: If activations_dir doesn't exist
- `ValueError`: If no activations found

#### Classes

##### `LogBinClassifier`
```python
class LogBinClassifier(nn.Module):
    def __init__(self, input_dim: int, n_bins: int = 11)
    
    def forward(self, x: Float[torch.Tensor, "batch d_model"]) -> Float[torch.Tensor, "batch n_bins"]:
        # Returns logits for each bin
```
Linear classifier for binned tokens remaining prediction with shape-annotated tensors.

##### `TrainingStats`
```python
class TrainingStats(BaseModel):
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
```
Pydantic model containing training statistics.

#### Helper Functions

##### `bin_targets()`
```python
def bin_targets(y: Int[torch.Tensor, "batch"], n_bins: int = 11) -> Int[torch.Tensor, "batch"]
```
Converts continuous target values to logarithmic bins using natural log: `floor(ln(y+1)).clamp(0, n_bins-1)`

##### `calculate_class_weights()`
```python
def calculate_class_weights(train_dataloader: DataLoader, n_bins: int = 11) -> Float[torch.Tensor, "n_bins"]
```
Calculates inverse frequency weights for balanced training.

### Example Usage

```python
from pipeline.train_probe import train_probe

model, stats = train_probe(
    activations_dir="/workspace/llm-progress-monitor/rollouts/activations",
    start_idx=0,
    end_idx=1000,
    layer_idx=10,
    n_bins=11,
    learning_rate=1e-4,
    n_epochs=3,
    save_path="/workspace/llm-progress-monitor/models/probe_layer10.pt"
)

print(f"Final train loss: {stats.final_train_loss:.4f}")
print(f"Final test loss: {stats.final_test_loss:.4f}")
print(f"Trained on {stats.train_samples} samples from {stats.train_sequences} sequences")
```

### Binning Strategy

The logarithmic binning uses natural logarithm (base e) and maps tokens remaining to bins as follows:

**Formula**: `bin = floor(ln(tokens_remaining + 1))`

**Bin Mapping**:
- Bin 0: 0 tokens (last token)
- Bin 1: 1 token  
- Bin 2: 2-3 tokens (e¹ ≈ 2.72)
- Bin 3: 4-7 tokens (e² ≈ 7.39)
- Bin 4: 8-20 tokens (e³ ≈ 20.09)
- Bin 5: 21-54 tokens (e⁴ ≈ 54.60)
- Bin 6: 55-148 tokens (e⁵ ≈ 148.41)
- Bin 7: 149-403 tokens (e⁶ ≈ 403.43)
- Bin 8: 404-1096 tokens (e⁷ ≈ 1096.63)
- Bin 9: 1097-2980 tokens (e⁸ ≈ 2980.96)
- Bin 10: 2981+ tokens (e⁹ ≈ 8103.08)

This handles the long tail of completion lengths while maintaining granularity for near-completion predictions.

---

## Type Definitions

**File**: `types.py`

### `PromptData`
```python
class PromptData(BaseModel):
    instruction: str  # The instruction/prompt text
    category: Optional[str] = None  # Optional category label
```
Input prompt format with runtime validation.

### `ChatMessage`
```python
class ChatMessage(TypedDict):
    role: str  # "user" or "assistant"
    content: str  # Message text
```
Type definition for chat messages.

### `RolloutResponse`
```python
class RolloutResponse(BaseModel):
    instruction: str  # The instruction/prompt text
    response: str  # The model's generated response
    char_length: int  # Character length of response
    tokens_length: int  # Token length of response
```
Rollout response format with runtime validation.

### `FormattedResponse`
```python
class FormattedResponse(TypedDict):
    instruction: str
    response: str
    chat_formatted: str  # Full chat-formatted text
    char_length: int
    tokens_length: int
```
Response with chat template applied.

---

## Data Formats

### Input: Prompts JSON
**Location**: e.g., `/workspace/llm-progress-monitor/splits/harmful_train.json`

```json
[
    {
        "instruction": "Explain how photosynthesis works",
        "category": "science"
    },
    {
        "instruction": "Write a story about a robot",
        "category": null
    }
]
```

**Schema**:
- `instruction` (required): The prompt text
- `category` (optional): Category label for filtering/analysis

### Stage 1 Output: Rollouts JSON
**Location**: e.g., `{save_path}/{model_name}-{thinking_mode}.json`

```json
[
    {
        "instruction": "Explain how photosynthesis works",
        "response": "Photosynthesis is the process by which...",
        "char_length": 523,
        "tokens_length": 147
    }
]
```

### Stage 2 Output: Activation Files
**Location**: `{activations_dir}/{idx}.pt`

Each file contains a tuple loaded via `torch.load()`:
```python
(token_ids, activations)
# token_ids: Tensor of shape (seq_len,)
# activations: Tensor of shape (n_layers, seq_len, d_model) or (1, seq_len, d_model)
```

**Dimensions**:
- `seq_len`: Number of tokens in this specific sequence (no padding)
- `n_layers`: Number of layers extracted (1 if `layer_idx` specified, else all)
- `d_model`: Hidden dimension of the model (e.g., 3584 for Qwen3-4B)

### Stage 4 Output: Probe Weights
**Location**: Specified by `save_path` parameter

Single tensor of shape `(n_bins, d_model)` saved via `torch.save()`.

---

## Common Modifications

### For Different Models

**Change Model**: Update `model_name` parameter in all stages.

**Different Chat Template**: The pipeline auto-detects via tokenizer. For custom templates:
- Modify `format_prompt_with_chat_template()` in `generate_prompt_rollouts.py`
- Modify `format_responses_with_chat_template()` in `store_activations.py`

**Different Thinking Support**: Models without thinking should use `thinking_mode=0`.

### For Different Tasks

**Different Prompts**: Replace the prompts JSON with your own data following the schema.

**Different Target Variable**: To predict something other than tokens remaining:
1. Modify `TokensRemainingDataset.__init__()` to compute different targets
2. Update `bin_targets()` if binning strategy needs to change
3. Rename classes appropriately

**Regression Instead of Classification**:
1. Change `LogBinClassifier` to output single value: `nn.Linear(input_dim, 1)`
2. Replace `CrossEntropyLoss` with `MSELoss` or `L1Loss`
3. Remove `bin_targets()` call in training loop

### For Different Architectures

**Activation Extraction Point**: To extract from different layers/modules:
- Modify lines 238-267 in `store_activations.py`
- Change `model.model.layers[i]` to your target module
- Example: `model.model.encoder.layer[i]` for encoder-decoder models

**Different Probe Architecture**:
- Modify `LogBinClassifier` in `train_probe.py`
- Example: Add hidden layers, use MLP instead of linear

### For Memory Constraints

**Reduce Batch Size**: Lower `batch_size` in Stages 2 and 4.

**Extract Single Layer**: Use `layer_idx` parameter in Stage 2.

**Process in Chunks**: Use `start_idx` and `end_idx` to process data in chunks.

**Reduce Workers**: Lower `num_save_workers` in Stage 2 if CPU-bound.

### For Performance

**Increase Batch Size**: Higher `batch_size` in Stage 2 and 4 (if memory allows).

**More Save Workers**: Increase `num_save_workers` in Stage 2 for faster I/O.

**Disable Logging**: Change logging level to `WARNING` or `ERROR`.

**Use Float16**: Change `dtype=torch.float16` in Stage 2 (may reduce precision).

---

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `vllm`: Fast LLM inference
- `nnsight`: Model introspection and activation extraction
- `torch`: Deep learning framework
- `transformers`: HuggingFace models and tokenizers
- `pydantic`: Runtime validation
- `scikit-learn`: Train/test splitting
- `jaxtyping`: Shape-annotated tensor types

---

## Further Reading

For theoretical background on probing and mechanistic interpretability:
- Linear probes: Alain & Bengio (2016) "Understanding intermediate layers using linear classifier probes"
- nnsight library: https://nnsight.net/
- vLLM: https://docs.vllm.ai/
- jaxtyping: https://github.com/google/jaxtyping

For questions or contributions, see the main project README.


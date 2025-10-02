# %%
from pipeline.generate_prompt_rollouts import generate_rollouts
from pipeline.store_activations import store_activations
from pipeline.load_activations import prepare_dataloaders

# %%
if __name__ == "__main__":
    generate_rollouts(
        model_name="google/gemma-2-2b-it",
        prompts_path="/workspace/llm-progress-monitor/splits/harmless_train.json",
        num_rollouts=100,
        max_tokens=10,
        save_path="/workspace/llm-progress-monitor/rollouts-test",
        thinking_mode=2,
    )

    '''store_activations(
        model_name="google/gemma-2-2b-it",
        rollouts_path="/workspace/llm-progress-monitor/rollouts-test/gemma-2-2b-it-True.json",
        activations_dir="/workspace/llm-progress-monitor/rollouts-test/activations",
        batch_size = 64
    )'''

    '''train_dataloader, test_dataloader, stats = prepare_dataloaders(
        activations_dir="/workspace/llm-progress-monitor/rollouts-test/activations",
        batch_size=32,
        test_size=0.2
    )'''
    
    '''print(f"Dataset stats: {stats}")
    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Test dataloader: {len(test_dataloader)} batches")'''
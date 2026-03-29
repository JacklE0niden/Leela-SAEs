
import torch
from lm_saes import (
    ActivationFactoryTarget,
    BufferShuffleConfig,
    DatasetConfig,
    GenerateActivationsSettings,
    LanguageModelConfig,
    generate_activations,
)


if __name__ == "__main__":
    settings = GenerateActivationsSettings(
        model=LanguageModelConfig(
            model_name="lc0/BT4-1024x15x32h",
            device="cuda",
            dtype="torch.float32",
        ),
        model_name="lc0/BT4-1024x15x32h",
        dataset=DatasetConfig(
            dataset_name_or_path="/path/to/dataset",
            is_dataset_on_disk=True,
        ),
        dataset_name="master",
        hook_points=[f"blocks.{layer}.hook_mlp_out" for layer in range(15)] + [f"blocks.{layer}.resid_mid_after_ln" for layer in range(15)],
        
        output_dir="/path/to/activations",
        total_tokens=801_000_000,
        context_size=64,
        n_samples_per_chunk=None,
        model_batch_size=4,
        target=ActivationFactoryTarget.ACTIVATIONS_1D,
        batch_size=2048 * 32,
        buffer_size=2048,
        buffer_shuffle=BufferShuffleConfig(
            perm_seed=42,
            generator_device="cuda",
        ),
        num_workers=16
    )
    generate_activations(settings)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


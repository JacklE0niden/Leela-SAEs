import torch
import os
import argparse

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    AnalyzeSAESettings,
    FeatureAnalyzerConfig,
    MongoDBConfig,
    SAEConfig,
    analyze_sae,
)
from lm_saes.utils.timer import timer

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze TC features.")
    parser.add_argument('--layer', type=int, default=14, help='Layer to analyze. (default: 14)')
    parser.add_argument('--k', type=int, default=64, help='top_k (default: 64)')
    parser.add_argument('--exp_factor', type=int, default=32, help='expansion factor (default: 32)')
    parser.add_argument("--n_tokens", type=int, default=100_000_000, help="Number of tokens to analyze.")
    return parser.parse_args()



if __name__ == "__main__":
    
    timer.enable()
    args = parse_args()
    layer = args.layer
    
    folder_name = f"k_{args.k}_e_{args.exp_factor}"
    
    settings = AnalyzeSAESettings(
        sae=SAEConfig.from_pretrained(
            f"/path/to/tc/L{layer}",
            device="cuda", 
            dtype=torch.float32,
        ),
        model_name="lc0/BT4-1024x15x32h",
        analyzer=FeatureAnalyzerConfig(
            total_analyzing_tokens=args.n_tokens,
            batch_size=8,
            enable_sampling=True,
            subsamples={
                "top_activations": {"proportion": 1.0, "n_samples": 16},
                "sampling_0.7": {"proportion": 0.7, "n_samples": 16},
                "sampling_0.5": {"proportion": 0.5, "n_samples": 16},
                "sampling_0.2": {"proportion": 0.2, "n_samples": 16},
            },
            non_activating_subsample=None,
        ),
        sae_name=f"BT4_tc_L{layer}M_k{args.k}_e{args.exp_factor}",
        sae_series="BT4",
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path="/path/to/activations",
                    type="activations",
                    name="master",
                    device="cuda",
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_2D,
            hook_points=[f"blocks.{layer}.resid_mid_after_ln"],
            batch_size=32,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        mongo=MongoDBConfig(
        ),
    )
    analyze_sae(settings)
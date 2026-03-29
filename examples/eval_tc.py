import math
import torch
import os
import argparse
from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    SAEConfig,
    EvalConfig,
    WandbConfig,
)
from lm_saes.runners.eval import EvaluateSAESettings, evaluate_sae
from lm_saes.utils.timer import timer

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SAE with configurable lr and layer'
    )
    parser.add_argument(
        '--layer', type=int, default=14, 
        help='Layer number (default: 14)'
    )
    return parser.parse_args()



if __name__ == "__main__":
    # Build the evaluation settings
    timer.enable()
    args = parse_args()
    layer = args.layer
    
    hook_point_in = f"blocks.{args.layer}.resid_mid_after_ln"
    hook_point_out = f"blocks.{args.layer}.hook_mlp_out"
    
    cfg = SAEConfig.from_pretrained(
        f'/path/to/tc/L{layer}',
        device="cuda",
        dtype=torch.float32,
    )
    settings = EvaluateSAESettings(
        sae=cfg,
        sae_name=f"L{layer}M",
        sae_series="TC",
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    name='master',
                    path=os.path.expanduser(
                        "/path/to/activations"
                    ),
                    device="cuda",
                    dtype=torch.float32,
                    num_workers=8,
                    prefetch=8,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[hook_point_in, hook_point_out],
            batch_size=4096,
            buffer_size=None,
            ignore_token_ids=[0],
        ),
        model=None,
        eval=EvalConfig(
            feature_sampling_window=100,
            total_eval_tokens=10_000_000,
            use_cached_activations=True,
            device="cuda",
        ),
        device_type="cuda",
    )

    evaluate_sae(settings) 
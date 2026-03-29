import math
import time
import os

import torch
import numpy as np

from lm_saes import (
    ActivationFactoryActivationsSource,
    ActivationFactoryConfig,
    ActivationFactoryTarget,
    InitializerConfig,
    SAEConfig,
    TrainerConfig,
    TrainSAESettings,
    MongoDBConfig,
    WandbConfig,
    train_sae,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--layer', type=int, default=14, help='layer (default: 14)')
parser.add_argument("--tp", type=int, default=1, help="Number of model parallel processes.")
parser.add_argument("--dp", type=int, default=1, help="Number of data parallel processes.")
parser.add_argument('--k', type=int, default=30, help='top_k (default: 30)')
parser.add_argument('--exp_factor', type=int, default=16, help='expasion factor (default: 16)')

args = parser.parse_args()
l=args.layer
lr=args.lr
exp_factor=args.exp_factor

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    # seed = int(time.time())
    seed = 42
    print(f"[INFO] Using seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hook_point_in = f"blocks.{args.layer}.resid_mid_after_ln"
    hook_point_out = f"blocks.{args.layer}.hook_mlp_out"
    
    settings = TrainSAESettings(
        sae=SAEConfig(
            hook_point_in=hook_point_in,
            hook_point_out=hook_point_out,
            d_model=1024,
            proj_data=True,
            expansion_factor=exp_factor,
            act_fn='topk',
            top_k=args.k,
            norm_activation="dataset-wise",
            sparsity_include_decoder_norm=True,
            dtype=torch.float32,
            device="cuda",
            use_auxk=True,
            k_aux=512,
        ),
        model_name="lc0/BT4-1024x15x32h",
        initializer=InitializerConfig(
            state="training",
            init_search=True,
            bias_init_method="geometric_median",
            init_encoder_with_decoder_transpose = False,
            decoder_uniform_bound = (1024 * 16) ** (-0.5),
            encoder_uniform_bound = 1024 ** (-0.5),
        ),
        trainer=TrainerConfig(
            optimizer_type="lazyadam",
            update_param_on_zero_grad=False,
            update_exp_avg_on_zero_grad=False,
            update_exp_avg_sq_on_zero_grad=False,
            use_batch_norm_mse=False,
            initial_k = 1024 / 2,
            k_warmup_steps = 0.1,
            update_decoder_lr_with_l0=False,
            lr=lr,
            lr_scheduler_name="constantwithwarmup",
            lr_warm_up_steps=500,
            lr_cool_down_steps=0.2,
            total_training_tokens=800_000_000,
            log_frequency=100,
            feature_sampling_window=1000,
            eval_frequency=1000000,
            n_checkpoints=0,
            check_point_save_mode="linear",
            exp_result_path=f"/path/to/tc/L{l}",
        ),
        wandb=WandbConfig(
            log_to_wandb=True,
            wandb_project="TC",
            exp_name=f"L{l}M",
            wandb_entity=""
        ),
        activation_factory=ActivationFactoryConfig(
            sources=[
                ActivationFactoryActivationsSource(
                    path=os.path.expanduser(
                        "/path/to/activations"
                    ),
                    sample_weights=1.0,
                    name="master",
                    device="cuda",
                    dtype=torch.float32,
                    prefetch=4,
                )
            ],
            target=ActivationFactoryTarget.ACTIVATIONS_1D,
            hook_points=[hook_point_in, hook_point_out],
            batch_size=32768,
            buffer_size=None,
            ignore_token_ids=[],
        ),
        sae_name=f"BT4_tc_L{l}M_k{args.k}_e{exp_factor}",
        sae_series="BT4-exp128",
        model_parallel_size=args.tp,
        data_parallel_size=args.dp,
    )
    train_sae(settings)

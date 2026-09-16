from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from feature_perturbation import parse_args, run_feature_perturbation

from lm_saes.circuit.bt4_eval_common import build_frozen_error_partial_crm_hooks

SCRIPT_DIR = Path(__file__).resolve().parent


def build_only_lorsa_hooks(model, fen):
    """Replace attention with frozen-error LoRSA, keep the original MLP path."""

    return build_frozen_error_partial_crm_hooks(
        model,
        fen,
        replace_attention=True,
        replace_mlp=False,
    )


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / "feature_perturbation_results" / args.combo_id)

    run_feature_perturbation(
        args,
        frozen_hook_builder=build_only_lorsa_hooks,
    )

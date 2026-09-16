from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from faithfulness_evaluation import (
    DEFAULT_LORSA_ROOT,
    DEFAULT_MEAN_ACTIVATION_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PROMPT_DATASET_PATH,
    DEFAULT_TC_ROOT,
    evaluate_replacement_vs_mean_ablation,
    load_bt4_model,
    load_lorsa_for_layer,
    load_transcoder_for_layer,
    parse_dtype,
    resolve_device,
    sample_fens_from_dataset,
    write_summary_text,
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "kl_results"


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_layers(raw_layers: list[int] | None) -> list[int]:
    layers = list(range(DEFAULT_NUM_LAYERS)) if not raw_layers else sorted(set(raw_layers))
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= DEFAULT_NUM_LAYERS]
    if invalid_layers:
        raise ValueError(
            f"Invalid layers {invalid_layers}. Expected each layer in [0, {DEFAULT_NUM_LAYERS - 1}]."
        )
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random FENs and evaluate, for each layer, the KL divergence between the "
            "original policy output and the outputs after replacement or mean ablation."
        )
    )
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--components", choices=("mlp", "attn"), nargs="+", default=["mlp", "attn"])
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tc-root", type=Path, default=DEFAULT_TC_ROOT)
    parser.add_argument("--tc-checkpoint-subpath", type=str, default=None)
    parser.add_argument("--lorsa-root", type=Path, default=DEFAULT_LORSA_ROOT)
    parser.add_argument("--lorsa-checkpoint-subpath", type=str, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_PROMPT_DATASET_PATH)
    parser.add_argument("--mean-activation-root", type=Path, default=DEFAULT_MEAN_ACTIVATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--zero-bos", action="store_true")
    parser.add_argument("--use-hidden-pre", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = _parse_layers(args.layers)
    components = list(dict.fromkeys(args.components))

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    prompts = sample_fens_from_dataset(
        args.dataset_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if len(prompts) < args.sample_size:
        print(
            f"[KL_eval] warning: sampled {len(prompts)} FENs, "
            f"less than requested {args.sample_size}"
        )

    output_dir = args.output_root / (
        f"kl_layers_{layers[0]}-{layers[-1]}_"
        f"{'-'.join(components)}_n{len(prompts)}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sampled_fens.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    print(
        f"[KL_eval] loading model on {device}; "
        f"evaluating layers={layers} components={components} with {len(prompts)} FENs"
    )
    model = load_bt4_model(args.model_name, device=device, dtype=dtype)

    summary_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    for layer in layers:
        print(f"[KL_eval] loading replacement modules for layer={layer}")
        transcoder = load_transcoder_for_layer(
            layer,
            tc_root=args.tc_root,
            checkpoint_subpath=args.tc_checkpoint_subpath,
            device=device,
            dtype=dtype,
        )
        lorsa = load_lorsa_for_layer(
            layer,
            lorsa_root=args.lorsa_root,
            checkpoint_subpath=args.lorsa_checkpoint_subpath,
            device=device,
            dtype=dtype,
        )

        for component in components:
            print(f"[KL_eval] evaluating layer={layer} component={component}")
            _, summary = evaluate_replacement_vs_mean_ablation(
                model=model,
                transcoder=transcoder,
                lorsa=lorsa,
                prompts=prompts,
                layer=layer,
                component=component,
                mean_activation_root=args.mean_activation_root,
                batch_size=args.batch_size,
                progress_every=args.progress_every,
                zero_bos=args.zero_bos,
                use_hidden_pre=args.use_hidden_pre,
            )

            summary["config"] = {
                "model_name": args.model_name,
                "layers": layers,
                "layer": layer,
                "components": components,
                "component": component,
                "requested_sample_size": args.sample_size,
                "actual_sample_size": len(prompts),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "dtype": args.dtype,
                "dataset_path": str(args.dataset_path),
                "tc_root": str(args.tc_root),
                "tc_checkpoint_subpath": args.tc_checkpoint_subpath,
                "lorsa_root": str(args.lorsa_root),
                "lorsa_checkpoint_subpath": args.lorsa_checkpoint_subpath,
                "mean_activation_root": str(args.mean_activation_root),
                "zero_bos": args.zero_bos,
                "use_hidden_pre": args.use_hidden_pre,
            }

            component_dir = output_dir / f"layer{layer}_{component}"
            component_dir.mkdir(parents=True, exist_ok=True)
            (component_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_summary_text(summary, component_dir / "summary.txt")
            all_summaries.append(summary)

            replacement_full_kl = summary["replacement_vs_original"]["full_distribution"]["kl_ref_to_candidate"]
            mean_full_kl = summary["mean_ablation_vs_original"]["full_distribution"]["kl_ref_to_candidate"]
            replacement_legal_kl = summary["replacement_vs_original"]["legal_distribution"]["kl_ref_to_candidate"]
            mean_legal_kl = summary["mean_ablation_vs_original"]["legal_distribution"]["kl_ref_to_candidate"]
            replacement_legal_margin_abs_delta = summary["replacement_vs_original"]["legal_distribution"][
                "top1_logit_margin_abs_delta"
            ]
            mean_legal_margin_abs_delta = summary["mean_ablation_vs_original"]["legal_distribution"][
                "top1_logit_margin_abs_delta"
            ]
            replacement_legal_rank_spearman = summary["replacement_vs_original"]["legal_distribution"][
                "logit_rank_spearman"
            ]
            mean_legal_rank_spearman = summary["mean_ablation_vs_original"]["legal_distribution"][
                "logit_rank_spearman"
            ]

            summary_rows.append(
                {
                    "layer": layer,
                    "component": component,
                    "num_prompts": len(prompts),
                    "replacement_full_kl_mean": replacement_full_kl["mean"],
                    "replacement_full_kl_std": replacement_full_kl["std"],
                    "replacement_full_kl_median": replacement_full_kl["median"],
                    "mean_ablation_full_kl_mean": mean_full_kl["mean"],
                    "mean_ablation_full_kl_std": mean_full_kl["std"],
                    "mean_ablation_full_kl_median": mean_full_kl["median"],
                    "replacement_legal_kl_mean": replacement_legal_kl["mean"],
                    "replacement_legal_kl_std": replacement_legal_kl["std"],
                    "replacement_legal_kl_median": replacement_legal_kl["median"],
                    "mean_ablation_legal_kl_mean": mean_legal_kl["mean"],
                    "mean_ablation_legal_kl_std": mean_legal_kl["std"],
                    "mean_ablation_legal_kl_median": mean_legal_kl["median"],
                    "replacement_top1_agreement_rate": summary["replacement_vs_original"]["top1_agreement_rate"],
                    "mean_ablation_top1_agreement_rate": summary["mean_ablation_vs_original"]["top1_agreement_rate"],
                    "replacement_legal_top1_logit_margin_abs_delta_mean": replacement_legal_margin_abs_delta["mean"],
                    "mean_ablation_legal_top1_logit_margin_abs_delta_mean": mean_legal_margin_abs_delta["mean"],
                    "replacement_legal_logit_rank_spearman_mean": replacement_legal_rank_spearman["mean"],
                    "mean_ablation_legal_logit_rank_spearman_mean": mean_legal_rank_spearman["mean"],
                    "replacement_better_full_kl_rate": summary["replacement_better_than_mean_ablation"][
                        "full_kl_rate"
                    ],
                    "replacement_better_legal_kl_rate": summary["replacement_better_than_mean_ablation"][
                        "legal_kl_rate"
                    ],
                    "replacement_better_legal_top1_logit_margin_abs_delta_rate": summary[
                        "replacement_better_than_mean_ablation"
                    ]["legal_top1_logit_margin_abs_delta_rate"],
                    "replacement_better_legal_logit_rank_spearman_rate": summary[
                        "replacement_better_than_mean_ablation"
                    ]["legal_logit_rank_spearman_rate"],
                }
            )

    _write_csv(summary_rows, output_dir / "layer_summary.csv")
    (output_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "layers": layers,
                "components": components,
                "requested_sample_size": args.sample_size,
                "actual_sample_size": len(prompts),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "dtype": args.dtype,
                "dataset_path": str(args.dataset_path),
                "tc_root": str(args.tc_root),
                "tc_checkpoint_subpath": args.tc_checkpoint_subpath,
                "lorsa_root": str(args.lorsa_root),
                "lorsa_checkpoint_subpath": args.lorsa_checkpoint_subpath,
                "mean_activation_root": str(args.mean_activation_root),
                "zero_bos": args.zero_bos,
                "use_hidden_pre": args.use_hidden_pre,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[KL_eval] finished. results saved to {output_dir}")


if __name__ == "__main__":
    main()

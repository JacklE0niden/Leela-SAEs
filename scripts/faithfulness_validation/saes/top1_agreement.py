from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from faithfulness_evaluation import (
    DEFAULT_LORSA_ROOT,
    DEFAULT_MEAN_ACTIVATION_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PROMPT_DATASET_PATH,
    DEFAULT_TC_ROOT,
    build_mean_ablation_hooks,
    build_single_layer_replacement_hooks,
    chunked,
    load_bt4_model,
    load_lorsa_for_layer,
    load_mean_activation,
    load_transcoder_for_layer,
    parse_dtype,
    predict_top_legal_move,
    resolve_device,
    run_policy_logits,
    sample_fens_from_dataset,
)

MODEL_NAME = DEFAULT_MODEL_NAME
TC_ROOT = DEFAULT_TC_ROOT
TC_CHECKPOINT_SUBPATH: str | None = None
LORSA_ROOT = DEFAULT_LORSA_ROOT
LORSA_CHECKPOINT_SUBPATH: str | None = None
DATASET_PATH = DEFAULT_PROMPT_DATASET_PATH
MEAN_ACTIVATION_ROOT = DEFAULT_MEAN_ACTIVATION_ROOT
OUTPUT_ROOT = Path(__file__).resolve().parent / "top1_agreement_results"
EVAL_MODES = ("replacement", "mean_ablation")


def _mean_bool(values: Sequence[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_component(component: str) -> str:
    mapping = {
        "mlp": "mlp",
        "transcoder": "mlp",
        "attn": "attn",
        "lorsa": "attn",
    }
    normalized = mapping.get(component.lower())
    if normalized is None:
        raise ValueError(f"Unsupported component: {component}")
    return normalized


def _parse_layers(layer: int | None, layers: Sequence[int] | None) -> list[int]:
    if layer is not None and layers:
        raise ValueError("Use either --layer or --layers, not both.")
    selected = [layer] if layer is not None else (list(layers) if layers else list(range(DEFAULT_NUM_LAYERS)))
    unique_layers = sorted(set(selected))
    invalid_layers = [value for value in unique_layers if value < 0 or value >= DEFAULT_NUM_LAYERS]
    if invalid_layers:
        raise ValueError(
            f"Invalid layers {invalid_layers}. Expected each layer in [0, {DEFAULT_NUM_LAYERS - 1}]."
        )
    return unique_layers


def _parse_components(component: str | None, components: Sequence[str] | None) -> list[str]:
    if component is not None and components:
        raise ValueError("Use either --component or --components, not both.")
    raw_components = [component] if component is not None else (list(components) if components else ["mlp", "attn"])
    normalized: list[str] = []
    for value in raw_components:
        mapped = _normalize_component(value)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def evaluate_top1_agreement(
    *,
    model,
    layer: int,
    component: str,
    prompts: Sequence[str],
    mode: str,
    transcoder=None,
    lorsa=None,
    batch_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "replacement":
        hooks = build_single_layer_replacement_hooks(
            layer=layer,
            component=component,
            transcoder=transcoder,
            lorsa=lorsa,
        )
    elif mode == "mean_ablation":
        mean_activation = load_mean_activation(
            layer,
            component,
            mean_activation_root=MEAN_ACTIVATION_ROOT,
        ).to(model.cfg.device, model.cfg.dtype)
        hooks = build_mean_ablation_hooks(
            layer=layer,
            component=component,
            mean_activation=mean_activation,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    rows: list[dict[str, Any]] = []
    agreement_flags: list[bool] = []
    batches = list(chunked(list(prompts), batch_size))

    progress = tqdm(
        batches,
        desc=f"{mode} L{layer} {component}",
        unit="batch",
        leave=False,
    )
    for batch_prompts in progress:
        original_logits = run_policy_logits(model, batch_prompts)
        intervened_logits = run_policy_logits(model, batch_prompts, fwd_hooks=hooks)

        for row_idx, fen in enumerate(batch_prompts):
            original_move = predict_top_legal_move(original_logits[row_idx], fen)
            intervened_move = predict_top_legal_move(intervened_logits[row_idx], fen)
            agrees = original_move is not None and intervened_move == original_move
            agreement_flags.append(agrees)
            rows.append(
                {
                    "sample_idx": len(rows),
                    "fen": fen,
                    "original_move": original_move,
                    "intervention_move": intervened_move,
                    "agrees_with_original": agrees,
                    "mode": mode,
                    "layer": layer,
                    "component": component,
                }
            )

        progress.set_postfix(samples=f"{len(rows)}/{len(prompts)}")

    summary = {
        "layer": layer,
        "component": component,
        "mode": mode,
        "num_samples": len(prompts),
        "top1_agreement_count": int(sum(agreement_flags)),
        "top1_agreement_rate": _mean_bool(agreement_flags),
    }
    return rows, summary


def write_summary_text(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"layer: {summary['layer']}",
        f"component: {summary['component']}",
        f"mode: {summary['mode']}",
        f"num_samples: {summary['num_samples']}",
        f"top1_agreement_count: {summary['top1_agreement_count']}",
        f"top1_agreement_rate: {summary['top1_agreement_rate']:.6f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run top-1 move agreement for each selected layer. Replacement-model "
            "inference runs first, followed by mean activation ablation."
        )
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--component", type=str, default=None)
    parser.add_argument("--components", type=str, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    return parser.parse_args()


def _config_dict(
    *,
    args: argparse.Namespace,
    layers: Sequence[int],
    components: Sequence[str],
    layer: int,
    component: str,
    mode: str,
    device: str,
    actual_num_samples: int,
) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "layers": list(layers),
        "layer": layer,
        "components": list(components),
        "component": component,
        "mode": mode,
        "num_samples": args.num_samples,
        "actual_num_samples": actual_num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
        "dtype": args.dtype,
        "dataset_path": str(DATASET_PATH),
        "tc_root": str(TC_ROOT),
        "tc_checkpoint_subpath": TC_CHECKPOINT_SUBPATH,
        "lorsa_root": str(LORSA_ROOT),
        "lorsa_checkpoint_subpath": LORSA_CHECKPOINT_SUBPATH,
        "mean_activation_root": str(MEAN_ACTIVATION_ROOT),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    layers = _parse_layers(args.layer, args.layers)
    components = _parse_components(args.component, args.components)

    prompts = sample_fens_from_dataset(
        DATASET_PATH,
        sample_size=args.num_samples,
        seed=args.seed,
    )
    if len(prompts) < args.num_samples:
        print(
            f"[top1_agreement] warning: sampled {len(prompts)} prompts, "
            f"less than requested {args.num_samples}"
        )

    output_dir = OUTPUT_ROOT / (
        f"layers_{layers[0]}-{layers[-1]}_{'-'.join(components)}_"
        f"n{len(prompts)}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[top1_agreement] loading model on {device}; "
        f"modes={list(EVAL_MODES)} layers={layers} components={components}"
    )
    model = load_bt4_model(MODEL_NAME, device=device, dtype=dtype)

    summary_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    for layer in layers:
        print(f"[top1_agreement] loading replacement modules for layer={layer}")
        transcoder = (
            load_transcoder_for_layer(
                layer,
                tc_root=TC_ROOT,
                checkpoint_subpath=TC_CHECKPOINT_SUBPATH,
                device=device,
                dtype=dtype,
            )
            if "mlp" in components
            else None
        )
        lorsa = (
            load_lorsa_for_layer(
                layer,
                lorsa_root=LORSA_ROOT,
                checkpoint_subpath=LORSA_CHECKPOINT_SUBPATH,
                device=device,
                dtype=dtype,
            )
            if "attn" in components
            else None
        )

        for mode in EVAL_MODES:
            for component in components:
                print(
                    f"[top1_agreement] evaluating layer={layer} "
                    f"component={component} mode={mode}"
                )
                rows, summary = evaluate_top1_agreement(
                    model=model,
                    layer=layer,
                    component=component,
                    prompts=prompts,
                    mode=mode,
                    transcoder=transcoder,
                    lorsa=lorsa,
                    batch_size=args.batch_size,
                )
                summary["config"] = _config_dict(
                    args=args,
                    layers=layers,
                    components=components,
                    layer=layer,
                    component=component,
                    mode=mode,
                    device=device,
                    actual_num_samples=len(prompts),
                )

                component_dir = output_dir / mode / f"layer{layer}_{component}"
                component_dir.mkdir(parents=True, exist_ok=True)
                _write_csv(rows, component_dir / "predictions.csv")
                (component_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                write_summary_text(summary, component_dir / "summary.txt")

                all_summaries.append(summary)
                summary_rows.append(
                    {
                        "layer": layer,
                        "component": component,
                        "mode": mode,
                        "num_samples": summary["num_samples"],
                        "top1_agreement_count": summary["top1_agreement_count"],
                        "top1_agreement_rate": summary["top1_agreement_rate"],
                    }
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    (output_dir / "sampled_fens.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    _write_csv(summary_rows, output_dir / "top1_agreement_summary.csv")
    (output_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[top1_agreement] finished. results saved to {output_dir}")


if __name__ == "__main__":
    main()

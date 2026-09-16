from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from tqdm import tqdm

from lm_saes.circuit.bt4_eval_common import (
    EvalCase,
    GraphFeature,
    build_encoder_direction,
    build_feature_delta_hook,
    build_frozen_error_crm_hooks,
    build_full_tensor_hook,
    build_position_vector_hook,
    build_random_direction,
    cosine_similarity,
    ensure_dir,
    load_bt4_bundle,
    load_cases,
    normalized_mse,
    run_setup_attribution_sparse,
    run_with_cache_hooks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = Path(
    os.environ.get("CHESS_DATASET_PATH", PROJECT_ROOT / "data" / "chess_master_data")
)

# Type aliases for stats
StatsDict = dict[str, tuple[torch.Tensor, int] | None]
LayerStatsDict = dict[int, StatsDict]
UpstreamStatsDict = dict[int, dict[int, StatsDict]]
FrozenHookBuilder = Callable[[Any, str], tuple[list[tuple[str, Any]], torch.Tensor]]

METRIC_NAMES = ["cosine_similarity", "mse", "normalized_mse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chess BT4 feature perturbation evaluation.")
    parser.add_argument("--cases", default=None, help="Path to JSON/JSONL/CSV/TXT cases file.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH), help="Fallback chess dataset path when --cases is not provided.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to ./feature_perturbation_results/<combo-id>/",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--combo-id", default="k_30_e_16")
    parser.add_argument("--sae-series", default="BT4-exp128")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--features-per-layer", type=int, default=16)
    parser.add_argument(
        "--upstream-layers",
        nargs="+",
        default=[],
        help="Layer ids or inclusive ranges, e.g. `0 1 2` or `0-13`.",
    )
    parser.add_argument("--delta-activation", type=float, default=0.01)
    parser.add_argument("--type", type=str, choices=["e", "r", "u"], nargs="+", default=["e"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.upstream_layers = parse_layer_specs(args.upstream_layers)
    return args


def parse_layer_specs(specs: Sequence[str]) -> list[int]:
    layers: list[int] = []
    for spec in specs:
        token = spec.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid upstream layer range: {token}")
            layers.extend(range(start, end + 1))
            continue
        layers.append(int(token))
    return sorted(set(layers))


def load_dataset_cases(dataset_path: str | os.PathLike[str], max_cases: int, seed: int) -> list[EvalCase]:
    import datasets

    dataset = datasets.load_from_disk(os.path.expanduser(str(dataset_path)))
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset["train"] if "train" in dataset else dataset[next(iter(dataset.keys()))]

    total = len(dataset)
    if total == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    sample_size = min(max_cases, total)
    rng = random.Random(seed)
    indices = list(range(total)) if sample_size == total else rng.sample(range(total), sample_size)

    cases: list[EvalCase] = []
    for index in indices:
        record = dataset[int(index)]
        fen = record.get("fen") if isinstance(record, dict) else None
        if not isinstance(fen, str) or not fen.strip():
            continue
        cases.append(EvalCase(fen=fen.strip(), label=f"dataset_{index:06d}"))
    if not cases:
        raise ValueError(f"No valid `fen` rows found in dataset: {dataset_path}")
    return cases


def load_eval_cases(args: argparse.Namespace) -> list[EvalCase]:
    if args.cases:
        return load_cases(args.cases, max_cases=args.max_cases)
    return load_dataset_cases(args.dataset_path, max_cases=args.max_cases, seed=args.seed)


def collect_eval_hook_names(model: Any, intervention_layer: int) -> list[str]:
    # We evaluate only the residual-stream states after each transformer block.
    # The last block output already corresponds to the model's final pre-logit
    # representation, so adding policy-head hooks here would duplicate the last
    # layer under a different name and make the saved matrices harder to read.
    return [f"blocks.{layer}.resid_post_after_ln" for layer in range(intervention_layer, model.cfg.n_layers)]


def _init_stats_dict() -> StatsDict:
    return {metric_name: None for metric_name in METRIC_NAMES}


def _update_stats(stats: StatsDict, metric_name: str, metrics: torch.Tensor) -> None:
    if stats[metric_name] is None:
        stats[metric_name] = (metrics, 1)
        return
    sum_tensor, count = stats[metric_name]
    stats[metric_name] = (sum_tensor + metrics, count + 1)


def _compute_mean(stats_dict: StatsDict) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for metric_name, data in stats_dict.items():
        if data is None:
            result[metric_name] = None
            continue
        sum_tensor, count = data
        result[metric_name] = sum_tensor / count
    return result


def _tensor_to_list(d: dict | torch.Tensor | None) -> dict | list | None:
    if isinstance(d, dict):
        return {k: _tensor_to_list(v) for k, v in d.items()}
    if isinstance(d, torch.Tensor):
        return d.cpu().tolist()
    if d is None:
        return None
    return d


def _mse(x: torch.Tensor, y: torch.Tensor) -> float:
    diff = x.detach().to(torch.float64) - y.detach().to(torch.float64)
    return float(torch.mean(diff**2).item())


def _evaluate_single_perturbation(
    *,
    model: Any,
    fen: str,
    hook_names: Sequence[str],
    frozen_hooks: Sequence[tuple[str, Any]],
    baseline_normal_cache: dict[str, torch.Tensor],
    baseline_crm_cache: dict[str, torch.Tensor],
    injection_hook: tuple[str, Any],
) -> dict[str, torch.Tensor]:
    _, pert_normal_cache = run_with_cache_hooks(model, fen, hook_names=hook_names, extra_hooks=[injection_hook])
    _, pert_crm_cache = run_with_cache_hooks(
        model,
        fen,
        hook_names=hook_names,
        extra_hooks=list(frozen_hooks) + [injection_hook],
    )

    results = {metric_name: [] for metric_name in METRIC_NAMES}
    for hook_name in hook_names:
        delta_normal = pert_normal_cache[hook_name] - baseline_normal_cache[hook_name]
        delta_crm = pert_crm_cache[hook_name] - baseline_crm_cache[hook_name]
        results["cosine_similarity"].append(cosine_similarity(delta_crm, delta_normal))
        results["mse"].append(_mse(delta_crm, delta_normal))
        results["normalized_mse"].append(normalized_mse(delta_crm, delta_normal))

    return {
        metric_name: torch.tensor(values, dtype=torch.float64)
        for metric_name, values in results.items()
    }


def _select_features(features: Sequence[Any], n_features: int, seed: int) -> list[Any]:
    candidates = [feature for feature in features if float(feature.activation_value) > 0]
    if not candidates:
        return []
    rng = random.Random(seed)
    if len(candidates) <= n_features:
        rng.shuffle(candidates)
        return list(candidates)
    indices = rng.sample(range(len(candidates)), n_features)
    return [candidates[index] for index in indices]


def _sparse_layer_features(
    sparse_tensor: torch.Tensor,
    *,
    feature_type: str,
    layer: int,
) -> list[GraphFeature]:
    sparse_tensor = sparse_tensor.coalesce()
    if sparse_tensor._nnz() == 0:
        return []

    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    layer_mask = indices[0] == layer
    if not bool(layer_mask.any().item()):
        return []

    positions = indices[1, layer_mask]
    feature_indices = indices[2, layer_mask]
    activations = values[layer_mask]

    positive_mask = activations > 0
    positions = positions[positive_mask]
    feature_indices = feature_indices[positive_mask]
    activations = activations[positive_mask]

    features: list[GraphFeature] = []
    for local_idx in range(len(feature_indices)):
        features.append(
            GraphFeature(
                node_index=-1,
                global_id=-1,
                feature_type=feature_type,
                layer=layer,
                position=int(positions[local_idx].item()),
                feature_idx=int(feature_indices[local_idx].item()),
                activation_value=float(activations[local_idx].item()),
            )
        )
    return features


def select_random_active_features(
    *,
    lorsa_sparse: torch.Tensor,
    tc_sparse: torch.Tensor,
    layer: int,
    n_features: int,
    seed: int,
) -> list[GraphFeature]:
    candidates = _sparse_layer_features(lorsa_sparse, feature_type="lorsa", layer=layer)
    candidates.extend(_sparse_layer_features(tc_sparse, feature_type="transcoder", layer=layer))
    return _select_features(candidates, n_features, seed)


def save_results(
    stats: LayerStatsDict | UpstreamStatsDict,
    experiment_name: str,
    args: argparse.Namespace,
    sample_count: int,
    n_layers: int,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 80)
    print(f"{experiment_name} Experiment Results")
    print(f"Total samples processed: {sample_count}")
    print("=" * 80)

    is_upstream = experiment_name == "upstream_feature"

    if is_upstream:
        upstream_stats: UpstreamStatsDict = stats  # type: ignore[assignment]
        mean_results: dict[int, dict[int, dict[str, torch.Tensor | None]]] = {}
        for upstream_layer in sorted(upstream_stats.keys()):
            mean_results[upstream_layer] = {}
            print(f"\n=== Upstream Layer {upstream_layer} ===")
            for intervention_layer in sorted(upstream_stats[upstream_layer].keys()):
                mean_results[upstream_layer][intervention_layer] = _compute_mean(
                    upstream_stats[upstream_layer][intervention_layer]
                )
                print(f"\n  Intervention Layer {intervention_layer}:")
                for metric_name in METRIC_NAMES:
                    values = mean_results[upstream_layer][intervention_layer][metric_name]
                    if values is not None:
                        print(f"    {metric_name}: {values.cpu().numpy()}")
    else:
        layer_stats: LayerStatsDict = stats  # type: ignore[assignment]
        mean_results: dict[int, dict[str, torch.Tensor | None]] = {}
        for layer in sorted(layer_stats.keys()):
            mean_results[layer] = _compute_mean(layer_stats[layer])
            print(f"\nLayer {layer}:")
            for metric_name in METRIC_NAMES:
                values = mean_results[layer][metric_name]
                if values is not None:
                    print(f"  {metric_name}: {values.cpu().numpy()}")

    results = {
        "data": _tensor_to_list(mean_results),
        "config": {
            "experiment_name": experiment_name,
            "combo_id": args.combo_id,
            "features_per_layer": args.features_per_layer,
            "delta_activation": args.delta_activation,
            "upstream_layers": args.upstream_layers if is_upstream else None,
            "sample_count": sample_count,
            "n_layers": n_layers,
        },
    }
    ensure_dir(output_dir)
    output_path = output_dir / f"{experiment_name}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


@torch.no_grad()
def run_feature_perturbation(
    args: argparse.Namespace,
    *,
    frozen_hook_builder: FrozenHookBuilder = build_frozen_error_crm_hooks,
    output_subdir: str = "feature_perturbation_results",
) -> None:
    cases = load_eval_cases(args)
    bundle = load_bt4_bundle(device=args.device, combo_id=args.combo_id, sae_series=args.sae_series)
    n_layers = int(bundle.model.cfg.n_layers)
    exp_types: list[str] = args.type
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent / output_subdir / args.combo_id
    )

    encoder_dir_stats: LayerStatsDict = {}
    random_dir_stats: LayerStatsDict = {}
    upstream_stats: UpstreamStatsDict = {layer: {} for layer in args.upstream_layers}

    for case_idx, case in enumerate(tqdm(cases, desc="Processing chess cases")):
        resolved_case = case
        _, lorsa_sparse, _, tc_sparse, _, _ = run_setup_attribution_sparse(bundle.model, resolved_case.fen)
        frozen_hooks, _ = frozen_hook_builder(bundle.model, resolved_case.fen)

        hook_names_by_layer: dict[int, list[str]] = {}
        baseline_normal_by_layer: dict[int, dict[str, torch.Tensor]] = {}
        baseline_crm_by_layer: dict[int, dict[str, torch.Tensor]] = {}

        def get_baseline_caches(intervention_layer: int) -> tuple[list[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            if intervention_layer not in hook_names_by_layer:
                hook_names = collect_eval_hook_names(bundle.model, intervention_layer)
                _, baseline_normal_cache = run_with_cache_hooks(bundle.model, resolved_case.fen, hook_names=hook_names)
                _, baseline_crm_cache = run_with_cache_hooks(
                    bundle.model,
                    resolved_case.fen,
                    hook_names=hook_names,
                    extra_hooks=frozen_hooks,
                )
                hook_names_by_layer[intervention_layer] = hook_names
                baseline_normal_by_layer[intervention_layer] = baseline_normal_cache
                baseline_crm_by_layer[intervention_layer] = baseline_crm_cache
            return (
                hook_names_by_layer[intervention_layer],
                baseline_normal_by_layer[intervention_layer],
                baseline_crm_by_layer[intervention_layer],
            )

        if "e" in exp_types or "r" in exp_types:
            for layer in range(n_layers):
                selected_features = select_random_active_features(
                    lorsa_sparse=lorsa_sparse,
                    tc_sparse=tc_sparse,
                    layer=layer,
                    n_features=args.features_per_layer,
                    seed=args.seed + case_idx * 1000 + layer,
                )
                if not selected_features:
                    continue

                hook_names, baseline_normal_cache, baseline_crm_cache = get_baseline_caches(layer)

                if "e" in exp_types and layer not in encoder_dir_stats:
                    encoder_dir_stats[layer] = _init_stats_dict()
                if "r" in exp_types and layer not in random_dir_stats:
                    random_dir_stats[layer] = _init_stats_dict()

                for local_idx, feature in enumerate(selected_features):
                    if "e" in exp_types: # select an encoder direction * delta_activation（扰动的强度）
                        encoder_delta = build_encoder_direction(feature, bundle, args.delta_activation)
                        encoder_hook = build_position_vector_hook(
                            f"blocks.{layer}.hook_attn_in",
                            feature.position,
                            encoder_delta,
                        )
                        encoder_metrics = _evaluate_single_perturbation(
                            model=bundle.model,
                            fen=resolved_case.fen,
                            hook_names=hook_names,
                            frozen_hooks=frozen_hooks,
                            baseline_normal_cache=baseline_normal_cache,
                            baseline_crm_cache=baseline_crm_cache,
                            injection_hook=encoder_hook,
                        )
                        for metric_name in METRIC_NAMES:
                            _update_stats(encoder_dir_stats[layer], metric_name, encoder_metrics[metric_name])

                    if "r" in exp_types:
                        random_delta = build_random_direction(
                            feature,
                            bundle,
                            args.delta_activation,
                            seed=args.seed + case_idx * 1000 + layer * 100 + local_idx,
                        )
                        random_hook = build_position_vector_hook(
                            f"blocks.{layer}.hook_attn_in",
                            feature.position,
                            random_delta,
                        )
                        random_metrics = _evaluate_single_perturbation(
                            model=bundle.model,
                            fen=resolved_case.fen,
                            hook_names=hook_names,
                            frozen_hooks=frozen_hooks,
                            baseline_normal_cache=baseline_normal_cache,
                            baseline_crm_cache=baseline_crm_cache,
                            injection_hook=random_hook,
                        )
                        for metric_name in METRIC_NAMES:
                            _update_stats(random_dir_stats[layer], metric_name, random_metrics[metric_name])

        if "u" in exp_types:
            for upstream_layer in args.upstream_layers:
                selected_features = select_random_active_features(
                    lorsa_sparse=lorsa_sparse,
                    tc_sparse=tc_sparse,
                    layer=upstream_layer,
                    n_features=args.features_per_layer,
                    seed=args.seed + case_idx * 1000 + upstream_layer,
                )
                if not selected_features:
                    continue

                for feature in selected_features:
                    source_hook = build_feature_delta_hook(bundle, feature, args.delta_activation)
                    for intervention_layer in range(upstream_layer + 1, n_layers):
                        hook_names, baseline_normal_cache, baseline_crm_cache = get_baseline_caches(intervention_layer)
                        target_hook = f"blocks.{intervention_layer}.resid_post_after_ln"
                        _, upstream_cache = run_with_cache_hooks(
                            bundle.model,
                            resolved_case.fen,
                            hook_names=[target_hook],
                            extra_hooks=list(frozen_hooks) + [source_hook],
                        )
                        delta_tensor = (
                            upstream_cache[target_hook]
                            - baseline_crm_cache[target_hook]
                        )
                        upstream_injection = build_full_tensor_hook(
                            target_hook,
                            delta_tensor,
                        )
                        upstream_stats[upstream_layer].setdefault(intervention_layer, _init_stats_dict())
                        upstream_metrics = _evaluate_single_perturbation(
                            model=bundle.model,
                            fen=resolved_case.fen,
                            hook_names=hook_names,
                            frozen_hooks=frozen_hooks,
                            baseline_normal_cache=baseline_normal_cache,
                            baseline_crm_cache=baseline_crm_cache,
                            injection_hook=upstream_injection,
                        )
                        for metric_name in METRIC_NAMES:
                            _update_stats(
                                upstream_stats[upstream_layer][intervention_layer],
                                metric_name,
                                upstream_metrics[metric_name],
                            )

    sample_count = len(cases)
    if "e" in exp_types and encoder_dir_stats:
        save_results(encoder_dir_stats, "encoder_direction", args, sample_count, n_layers, output_dir)
    if "r" in exp_types and random_dir_stats:
        save_results(random_dir_stats, "random_direction", args, sample_count, n_layers, output_dir)
    if "u" in exp_types and upstream_stats:
        save_results(upstream_stats, "upstream_feature", args, sample_count, n_layers, output_dir)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    run_feature_perturbation(args)

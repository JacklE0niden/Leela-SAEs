from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from faithfulness_evaluation import (
    DEFAULT_LORSA_ROOT,
    DEFAULT_MEAN_ACTIVATION_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LAYERS,
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
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "puzzle_results"
DEFAULT_PUZZLE_CSV = Path(
    os.environ.get("CHESS_PUZZLE_CSV", PROJECT_ROOT / "data" / "puzzles" / "after_move1.csv")
)


def read_puzzle_csv_reservoir(
    csv_path: Path,
    sample_size: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    reservoir: list[tuple[str, str]] = []
    seen_fens: set[str] = set()
    seen_valid = 0

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            fen = row[0].strip()
            moves_cell = row[1].strip()
            if not fen or not moves_cell or fen in seen_fens:
                continue
            first_move = moves_cell.split()[0].strip().lower()
            if not first_move:
                continue
            seen_fens.add(fen)
            item = (fen, first_move)
            seen_valid += 1
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                idx = rng.randrange(seen_valid)
                if idx < sample_size:
                    reservoir[idx] = item
    return reservoir


def aggregate_accuracy(correct_flags: Sequence[bool]) -> dict[str, float | int]:
    total = len(correct_flags)
    correct = int(sum(correct_flags))
    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct / total) if total else 0.0,
    }


def evaluate_puzzles(
    *,
    model,
    layer: int,
    component: str,
    samples: Sequence[tuple[str, str]],
    transcoder=None,
    lorsa=None,
    mean_activation_root: str | Path = DEFAULT_MEAN_ACTIVATION_ROOT,
    batch_size: int = 64,
    progress_every: int = 20,
    zero_bos: bool = False,
    use_hidden_pre: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replacement_hooks = build_single_layer_replacement_hooks(
        layer=layer,
        component=component,
        transcoder=transcoder,
        lorsa=lorsa,
        zero_bos=zero_bos,
        use_hidden_pre=use_hidden_pre,
    )
    mean_activation = load_mean_activation(
        layer,
        component,
        mean_activation_root=mean_activation_root,
    ).to(model.cfg.device, model.cfg.dtype)
    mean_ablation_hooks = build_mean_ablation_hooks(
        layer=layer,
        component=component,
        mean_activation=mean_activation,
    )

    rows: list[dict[str, Any]] = []
    original_correct: list[bool] = []
    replacement_correct: list[bool] = []
    mean_correct: list[bool] = []

    batches = list(chunked([fen for fen, _ in samples], batch_size))
    references = [move for _, move in samples]

    seen = 0
    for batch_idx, batch_fens in enumerate(batches, start=1):
        batch_refs = references[seen : seen + len(batch_fens)]
        seen += len(batch_fens)

        original_logits = run_policy_logits(model, batch_fens)
        replacement_logits = run_policy_logits(model, batch_fens, fwd_hooks=replacement_hooks)
        mean_logits = run_policy_logits(model, batch_fens, fwd_hooks=mean_ablation_hooks)

        for row_idx, (fen, reference_move) in enumerate(zip(batch_fens, batch_refs, strict=False)):
            original_move = predict_top_legal_move(original_logits[row_idx], fen)
            replacement_move = predict_top_legal_move(replacement_logits[row_idx], fen)
            mean_move = predict_top_legal_move(mean_logits[row_idx], fen)

            original_hit = original_move == reference_move
            replacement_hit = replacement_move == reference_move
            mean_hit = mean_move == reference_move

            original_correct.append(original_hit)
            replacement_correct.append(replacement_hit)
            mean_correct.append(mean_hit)

            rows.append(
                {
                    "sample_idx": len(rows),
                    "fen": fen,
                    "reference_move": reference_move,
                    "original_move": original_move,
                    "replacement_move": replacement_move,
                    "mean_ablation_move": mean_move,
                    "original_correct": original_hit,
                    "replacement_correct": replacement_hit,
                    "mean_ablation_correct": mean_hit,
                }
            )

        if progress_every > 0 and (
            batch_idx % progress_every == 0 or batch_idx == len(batches)
        ):
            print(
                f"[puzzle_eval] processed {batch_idx}/{len(batches)} batches "
                f"({len(rows)}/{len(samples)} puzzles)"
            )

    summary = {
        "layer": layer,
        "component": component,
        "num_samples": len(samples),
        "original": aggregate_accuracy(original_correct),
        "replacement": aggregate_accuracy(replacement_correct),
        "mean_ablation": aggregate_accuracy(mean_correct),
    }
    return rows, summary


def write_summary_text(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"layer: {summary['layer']}",
        f"component: {summary['component']}",
        f"num_samples: {summary['num_samples']}",
        "",
        "[original]",
        f"total: {summary['original']['total']}",
        f"correct: {summary['original']['correct']}",
        f"accuracy: {summary['original']['accuracy']:.6f}",
        "",
        "[replacement]",
        f"total: {summary['replacement']['total']}",
        f"correct: {summary['replacement']['correct']}",
        f"accuracy: {summary['replacement']['accuracy']:.6f}",
        "",
        "[mean_ablation]",
        f"total: {summary['mean_ablation']['total']}",
        f"correct: {summary['mean_ablation']['correct']}",
        f"accuracy: {summary['mean_ablation']['accuracy']:.6f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate puzzle accuracy for replacement vs mean ablation on one or many layers."
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--component", type=str, default=None)
    parser.add_argument("--components", type=str, nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tc-root", type=Path, default=DEFAULT_TC_ROOT)
    parser.add_argument("--tc-checkpoint-subpath", type=str, default=None)
    parser.add_argument("--lorsa-root", type=Path, default=DEFAULT_LORSA_ROOT)
    parser.add_argument("--lorsa-checkpoint-subpath", type=str, default=None)
    parser.add_argument("--puzzle-csv", type=Path, default=DEFAULT_PUZZLE_CSV)
    parser.add_argument("--mean-activation-root", type=Path, default=DEFAULT_MEAN_ACTIVATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--zero-bos", action="store_true")
    parser.add_argument("--use-hidden-pre", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    samples = read_puzzle_csv_reservoir(args.puzzle_csv, args.sample_size, rng)
    if len(samples) < args.sample_size:
        print(
            f"[puzzle_eval] warning: only sampled {len(samples)} puzzles, "
            f"less than requested {args.sample_size}"
        )

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    layers = _parse_layers(args.layer, args.layers)
    components = _parse_components(args.component, args.components)
    output_dir = args.output_root / (
        f"puzzle_layers_{layers[0]}-{layers[-1]}_{'-'.join(components)}_n{len(samples)}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[puzzle_eval] loading model on {device}; "
        f"evaluating layers={layers} components={components}"
    )
    model = load_bt4_model(args.model_name, device=device, dtype=dtype)
    summary_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    for layer in layers:
        print(f"[puzzle_eval] loading replacement modules for layer={layer}")
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
            print(f"[puzzle_eval] evaluating layer={layer} component={component}")
            rows, summary = evaluate_puzzles(
                model=model,
                layer=layer,
                component=component,
                samples=samples,
                transcoder=transcoder,
                lorsa=lorsa,
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
                "sample_size": args.sample_size,
                "actual_sample_size": len(samples),
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "dtype": args.dtype,
                "puzzle_csv": str(args.puzzle_csv),
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
                    "num_samples": len(samples),
                    "original_accuracy": summary["original"]["accuracy"],
                    "replacement_accuracy": summary["replacement"]["accuracy"],
                    "mean_ablation_accuracy": summary["mean_ablation"]["accuracy"],
                    "replacement_correct": summary["replacement"]["correct"],
                    "mean_ablation_correct": summary["mean_ablation"]["correct"],
                }
            )

    (output_dir / "sample_info.txt").write_text(
        "\n".join(
            [
                f"puzzle_csv: {args.puzzle_csv}",
                f"requested_sample_size: {args.sample_size}",
                f"actual_sample_size: {len(samples)}",
                f"seed: {args.seed}",
                f"layers: {layers}",
                f"components: {components}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(summary_rows, output_dir / "layer_summary.csv")
    (output_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[puzzle_eval] finished. results saved to {output_dir}")


if __name__ == "__main__":
    main()

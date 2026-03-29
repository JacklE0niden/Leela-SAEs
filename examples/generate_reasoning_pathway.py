from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.path_generation.generate_reasoning_pathway import (
    DEFAULT_DEVICE,
    DEFAULT_LORSA_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_TC_ROOT,
    generate_path_csvs,
)


DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reasoning-pathway CSVs for a given FEN.",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=DEFAULT_FEN,
        help="FEN string to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reasoning_pathways"),
        help="Output directory (CSV + optional JSON will be written under a FEN-specific subfolder).",
    )

    parser.add_argument("--top-k-moves", type=int, default=1, help="Number of top moves to trace.")
    parser.add_argument("--n-features", type=int, default=200, help="Number of top features to keep per move.")
    parser.add_argument(
        "--reduction-ratio",
        type=float,
        default=0.1,
        help="Threshold for keeping interactions (more strict -> fewer rows).",
    )
    parser.add_argument("--steering-factor", type=float, default=0.0, help="Steering scale applied during analysis.")
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="Activation threshold used when selecting features.",
    )
    parser.add_argument(
        "--max-features-per-type",
        type=int,
        default=None,
        help="Optional cap on number of features per type (transcoder/lorsa).",
    )
    parser.add_argument(
        "--max-steering-features",
        type=int,
        default=None,
        help="Optional cap on number of steering features.",
    )

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device: cuda/cpu.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model name for TransformerLens.")
    parser.add_argument(
        "--tc-root",
        type=Path,
        default=DEFAULT_TC_ROOT,
        help="Root directory containing pretrained transcoders (expects L0..L14 subfolders).",
    )
    parser.add_argument(
        "--lorsa-root",
        type=Path,
        default=DEFAULT_LORSA_ROOT,
        help="Root directory containing pretrained LoRSAs (expects L0..L14 subfolders).",
    )
    parser.add_argument(
        "--save-analysis-json",
        action="store_true",
        help="Also save infl_all_feature.json alongside the generated CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    result: dict[str, Any] = generate_path_csvs(
        fen=args.fen,
        output_dir=args.output_dir,
        top_k_moves=args.top_k_moves,
        n_features=args.n_features,
        reduction_ratio=args.reduction_ratio,
        steering_factor=args.steering_factor,
        activation_threshold=args.activation_threshold,
        max_features_per_type=args.max_features_per_type,
        max_steering_features=args.max_steering_features,
        device=args.device,
        model_name=args.model_name,
        tc_root=args.tc_root,
        lorsa_root=args.lorsa_root,
        save_analysis_json=args.save_analysis_json,
    )

    printable = {
        "fen": result.get("fen"),
        "fen_output_dir": str(result.get("fen_output_dir")),
        "generated_csvs": {k: str(v) for k, v in result.get("generated_csvs", {}).items()},
        "analysis_json": str(result["analysis_json"]) if result.get("analysis_json") is not None else None,
        "selected_feature_counts": result.get("selected_feature_counts", {}),
        "move_probabilities": result.get("move_probabilities", {}),
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


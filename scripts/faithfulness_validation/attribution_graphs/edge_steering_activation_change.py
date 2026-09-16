from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy import stats
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent for parent in HERE.parents if (parent / "src").exists() and (parent / "server").exists()
)
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "server"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FeatureNode = tuple[int, int, int, str]  # layer, position, feature id, feature type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Steer each source feature in attribution-graph edges and compare the graph "
            "edge weight with the resulting downstream feature activation change."
        )
    )
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        default=HERE / "circuits_raw_influence",
        help="Directory containing attribution-graph trace JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "edge_steering_activation_change",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--combo-id", default="k_30_e_16")
    parser.add_argument("--sae-series", default="BT4-exp128")
    parser.add_argument(
        "--steering-scale",
        type=float,
        default=0.0,
        help="Source feature multiplier: 0 ablates it, 1 leaves it unchanged.",
    )
    parser.add_argument("--max-circuits", type=int, default=None)
    parser.add_argument(
        "--max-edges-per-circuit",
        type=int,
        default=None,
        help="Optional top-K graph edges by absolute weight.",
    )
    return parser.parse_args()


def node_id_to_feature(node_id: str) -> FeatureNode | None:
    parts = node_id.split("_")
    if len(parts) != 3:
        return None
    try:
        sublayer, feature_id, position = map(int, parts)
    except ValueError:
        return None  # error, embedding, and logit nodes are not steerable features
    if not (0 <= sublayer < 30 and feature_id >= 0 and position >= 0):
        return None
    feature_type = "lorsa" if sublayer % 2 == 0 else "transcoder"
    return sublayer // 2, position, feature_id, feature_type


def is_causally_ordered(source: FeatureNode, target: FeatureNode) -> bool:
    source_layer, _, _, source_type = source
    target_layer, _, _, target_type = target
    return source_layer < target_layer or (
        source_layer == target_layer
        and source_type == "lorsa"
        and target_type == "transcoder"
    )


def read_fen(payload: dict[str, Any], path: Path) -> str:
    metadata = payload.get("metadata", {})
    fen = metadata.get("prompt") if isinstance(metadata, dict) else None
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError(f"No metadata.prompt FEN in {path}")
    return fen.strip()


def read_feature_edges(
    payload: dict[str, Any], max_edges: int | None
) -> list[tuple[str, str, FeatureNode, FeatureNode, float]]:
    # If a graph contains duplicate directed links, retain the largest |weight|.
    deduplicated: dict[tuple[str, str], tuple[FeatureNode, FeatureNode, float]] = {}
    for link in payload.get("links", payload.get("edges", [])):
        source_id = str(link.get("source", ""))
        target_id = str(link.get("target", ""))
        source = node_id_to_feature(source_id)
        target = node_id_to_feature(target_id)
        if source is None or target is None or not is_causally_ordered(source, target):
            continue
        try:
            weight = float(link["weight"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(weight):
            continue
        key = source_id, target_id
        previous = deduplicated.get(key)
        if previous is None or abs(weight) > abs(previous[2]):
            deduplicated[key] = source, target, weight

    edges = [
        (source_id, target_id, source, target, value[2])
        for (source_id, target_id), value in deduplicated.items()
        for source, target in [(value[0], value[1])]
    ]
    edges.sort(key=lambda edge: abs(edge[4]), reverse=True)
    return edges if max_edges is None else edges[:max_edges]


def pearson(x: list[float], y: list[float]) -> tuple[float, float]:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), float("nan")
    result = stats.pearsonr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def evaluate_circuit(
    circuit_path: Path,
    payload: dict[str, Any],
    precomputed: dict[str, Any],
    steering_scale: float,
    max_edges: int | None,
) -> list[dict[str, Any]]:
    # Keep this import lazy so CLI inspection works on machines that do not have
    # the model runtime dependency (leela_interp) installed.
    from feature_and_steering.interact import analyze_node_activation_impact

    edges = read_feature_edges(payload, max_edges)
    edges_by_source: dict[FeatureNode, list[tuple[str, str, FeatureNode, float]]] = defaultdict(list)
    for source_id, target_id, source, target, weight in edges:
        edges_by_source[source].append((source_id, target_id, target, weight))

    rows: list[dict[str, Any]] = []
    fen = read_fen(payload, circuit_path)
    metadata = payload.get("metadata", {})
    for source, source_edges in tqdm(
        edges_by_source.items(),
        desc=circuit_path.name,
        leave=False,
    ):
        targets = [edge[2] for edge in source_edges]
        result = analyze_node_activation_impact(
            steering_nodes=[source],
            target_nodes=targets,
            steering_scale=steering_scale,
            cache=precomputed["cache"],
            model=precomputed["model"],
            tc_activations=precomputed["tc_activations"],
            lorsa_activations=precomputed["lorsa_activations"],
            tc_WDs=precomputed["tc_WDs"],
            lorsa_WDs=precomputed["lorsa_WDs"],
            transcoders=precomputed["transcoders"],
            lorsas=precomputed["lorsas"],
        )
        target_results = result.get("target_nodes", [])
        if len(target_results) != len(source_edges):
            raise RuntimeError(
                f"Target result count mismatch in {circuit_path}: "
                f"expected {len(source_edges)}, got {len(target_results)}"
            )

        steering_details = result.get("steering_details", [])
        source_activation = (
            float(steering_details[0].get("activation_value", 0.0))
            if steering_details
            else 0.0
        )
        for (source_id, target_id, target, weight), target_result in zip(
            source_edges, target_results
        ):
            original = float(target_result["original_activation"])
            modified = float(target_result["modified_activation"])
            change = modified - original
            drop = original - modified
            rows.append(
                {
                    "circuit_path": str(circuit_path),
                    "fen": fen,
                    "target_move": metadata.get("target_move") or metadata.get("move_uci"),
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "source_layer": source[0],
                    "source_pos": source[1],
                    "source_feature_id": source[2],
                    "source_feature_type": source[3],
                    "target_layer": target[0],
                    "target_pos": target[1],
                    "target_feature_id": target[2],
                    "target_feature_type": target[3],
                    "edge_weight": weight,
                    "abs_edge_weight": abs(weight),
                    "steering_scale": steering_scale,
                    "source_activation": source_activation,
                    "target_activation_before": original,
                    "target_activation_after": modified,
                    "target_activation_change": change,
                    "target_activation_drop": drop,
                    "abs_target_activation_change": abs(change),
                    "target_activation_ratio": (
                        modified / original if original != 0.0 else float("nan")
                    ),
                }
            )
    return rows


def summarize(circuit_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = [float(row["edge_weight"]) for row in rows]
    drops = [float(row["target_activation_drop"]) for row in rows]
    abs_weights = [abs(value) for value in weights]
    abs_changes = [abs(float(row["target_activation_change"])) for row in rows]
    signed_r, signed_p = pearson(weights, drops)
    abs_r, abs_p = pearson(abs_weights, abs_changes)
    return {
        "circuit_path": str(circuit_path),
        "n_edges": len(rows),
        "n_sources": len({row["source_node_id"] for row in rows}),
        "pearson_edge_weight_vs_activation_drop": signed_r,
        "p_edge_weight_vs_activation_drop": signed_p,
        "pearson_abs_edge_weight_vs_abs_activation_change": abs_r,
        "p_abs_edge_weight_vs_abs_activation_change": abs_p,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    circuit_paths = sorted(args.circuits_dir.rglob("trace_*.json"))
    if args.max_circuits is not None:
        circuit_paths = circuit_paths[: args.max_circuits]
    if not circuit_paths:
        raise FileNotFoundError(f"No trace JSON files found in {args.circuits_dir}")

    # Match the loading path used by Self_Cross_Connection/compute_connections.py.
    from lm_saes.circuit.bt4_eval_common import load_bt4_bundle
    from reasoning_path.path_evaluation.feature_infl import precompute_activations_and_weights

    bundle = load_bt4_bundle(
        device=args.device,
        combo_id=args.combo_id,
        sae_series=args.sae_series,
    )
    transcoders = {int(key): value for key, value in bundle.transcoders.items()}
    lorsas = bundle.lorsas

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for circuit_path in tqdm(circuit_paths, desc="Circuits"):
        try:
            payload = json.loads(circuit_path.read_text(encoding="utf-8"))
            fen = read_fen(payload, circuit_path)
            precomputed = precompute_activations_and_weights(
                fen, bundle.model, transcoders, lorsas
            )
            rows = evaluate_circuit(
                circuit_path,
                payload,
                precomputed,
                args.steering_scale,
                args.max_edges_per_circuit,
            )
            all_rows.extend(rows)
            summaries.append(summarize(circuit_path, rows))
        except Exception as exc:
            traceback.print_exc()
            failures.append({"circuit_path": str(circuit_path), "error": str(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "edge_effects.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output_dir / "edge_effects.csv", all_rows)
    write_csv(args.output_dir / "per_circuit_summary.csv", summaries)

    weights = [float(row["edge_weight"]) for row in all_rows]
    drops = [float(row["target_activation_drop"]) for row in all_rows]
    signed_r, signed_p = pearson(weights, drops)
    abs_r, abs_p = pearson(
        [abs(value) for value in weights],
        [abs(float(row["target_activation_change"])) for row in all_rows],
    )
    summary = {
        "circuits_dir": str(args.circuits_dir),
        "steering_scale": args.steering_scale,
        "n_circuits_requested": len(circuit_paths),
        "n_circuits_succeeded": len(summaries),
        "n_circuits_failed": len(failures),
        "n_edges": len(all_rows),
        "pearson_edge_weight_vs_activation_drop": signed_r,
        "p_edge_weight_vs_activation_drop": signed_p,
        "pearson_abs_edge_weight_vs_abs_activation_change": abs_r,
        "p_abs_edge_weight_vs_abs_activation_change": abs_p,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

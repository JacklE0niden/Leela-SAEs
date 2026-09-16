from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CIRCUITS_INPUT = Path(
    os.environ.get("BT4_CIRCUITS_DIR", PROJECT_ROOT / "circuit_trace_results")
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output_pruned_json"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute replacement_score and completeness_score directly from circuit JSON files "
            "for a specific circuit directory or for all leaf directories under a random-data results root."
        )
    )
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        default=DEFAULT_CIRCUITS_INPUT,
        help="A specific circuit directory, or a root directory containing many circuit subdirectories.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-files-per-group", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def load_circuit_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("graph_data"), dict):
        payload = payload["graph_data"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported JSON payload in {path}")
    if "nodes" not in payload or "links" not in payload:
        raise ValueError(f"Circuit JSON must contain `nodes` and `links`: {path}")
    return payload


def node_feature_type(node: dict[str, Any]) -> str:
    return str(node.get("feature_type", "")).strip().lower()


def is_logit_node(node: dict[str, Any]) -> bool:
    return node_feature_type(node) == "logit"


def is_token_node(node: dict[str, Any]) -> bool:
    return "embedding" in node_feature_type(node)


def is_error_node(node: dict[str, Any]) -> bool:
    return "error" in node_feature_type(node)


def build_index(nodes: list[dict[str, Any]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for node_idx, node in enumerate(nodes):
        node_id = node.get("node_id") or node.get("jsNodeId")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Node missing node_id/jsNodeId: {node}")
        index[node_id] = node_idx
    return index


def build_incoming_edges(
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> tuple[list[list[tuple[int, float]]], list[float]]:
    node_index = build_index(nodes)
    incoming_edges: list[list[tuple[int, float]]] = [[] for _ in nodes]
    row_abs_sums = [0.0 for _ in nodes]

    for link in links:
        source_id = link.get("source")
        target_id = link.get("target")
        weight = float(link.get("weight", 0.0))
        if source_id not in node_index or target_id not in node_index:
            continue
        src = node_index[source_id]
        dst = node_index[target_id]
        incoming_edges[dst].append((src, weight))
        row_abs_sums[dst] += abs(weight)

    return incoming_edges, row_abs_sums


def compute_logit_weights(nodes: list[dict[str, Any]]) -> list[float]:
    weights = [0.0 for _ in nodes]
    logit_indices = [idx for idx, node in enumerate(nodes) if is_logit_node(node)]
    for idx in logit_indices:
        token_prob = float(nodes[idx].get("token_prob", 0.0) or 0.0)
        if token_prob > 0:
            weights[idx] = token_prob

    if any(weight > 0 for weight in weights):
        return weights

    target_logits = [idx for idx in logit_indices if bool(nodes[idx].get("is_target_logit", False))]
    if target_logits:
        for idx in target_logits:
            weights[idx] = 1.0
        return weights

    if logit_indices:
        uniform = 1.0 / len(logit_indices)
        for idx in logit_indices:
            weights[idx] = uniform
    return weights


def propagate_influence(
    incoming_edges: list[list[tuple[int, float]]],
    row_abs_sums: list[float],
    logit_weights: list[float],
    *,
    max_iter: int,
) -> list[float]:
    current = [0.0 for _ in logit_weights]
    for dst, incoming in enumerate(incoming_edges):
        if logit_weights[dst] == 0.0:
            continue
        denom = row_abs_sums[dst]
        if denom <= 1e-12:
            continue
        for src, weight in incoming:
            current[src] += logit_weights[dst] * (abs(weight) / denom)

    total = [value for value in current]
    for _ in range(max_iter):
        if not current or max(abs(value) for value in current) <= 1e-12:
            break
        next_current = [0.0 for _ in current]
        for dst, incoming in enumerate(incoming_edges):
            if current[dst] == 0.0:
                continue
            denom = row_abs_sums[dst]
            if denom <= 1e-12:
                continue
            for src, weight in incoming:
                next_current[src] += current[dst] * (abs(weight) / denom)
        total = [acc + delta for acc, delta in zip(total, next_current)]
        current = next_current
    else:
        raise RuntimeError(f"Influence computation failed to converge after {max_iter} iterations")

    return total


def compute_non_error_fractions(
    nodes: list[dict[str, Any]],
    incoming_edges: list[list[tuple[int, float]]],
    row_abs_sums: list[float],
) -> list[float]:
    error_mask = [is_error_node(node) for node in nodes]
    fractions = [1.0 for _ in nodes]
    for dst, incoming in enumerate(incoming_edges):
        denom = row_abs_sums[dst]
        if denom <= 1e-12:
            fractions[dst] = 1.0
            continue
        error_fraction = 0.0
        for src, weight in incoming:
            if error_mask[src]:
                error_fraction += abs(weight) / denom
        fractions[dst] = 1.0 - error_fraction
    return fractions


def compute_scores_from_circuit_json(payload: dict[str, Any], *, max_iter: int) -> dict[str, float | int]:
    nodes = payload.get("nodes", [])
    links = payload.get("links", [])
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Circuit JSON must contain list-valued `nodes` and `links`.")

    incoming_edges, row_abs_sums = build_incoming_edges(nodes, links)
    logit_weights = compute_logit_weights(nodes)
    node_influence = propagate_influence(incoming_edges, row_abs_sums, logit_weights, max_iter=max_iter)

    token_influence = sum(node_influence[idx] for idx, node in enumerate(nodes) if is_token_node(node))
    error_influence = sum(node_influence[idx] for idx, node in enumerate(nodes) if is_error_node(node))
    replacement_denominator = token_influence + error_influence
    replacement_score = token_influence / replacement_denominator if replacement_denominator > 1e-12 else float("nan")

    non_error_fractions = compute_non_error_fractions(nodes, incoming_edges, row_abs_sums)
    output_influence = [node_score + logit_weight for node_score, logit_weight in zip(node_influence, logit_weights)]
    output_total = sum(output_influence)
    completeness_score = (
        sum(fraction * influence for fraction, influence in zip(non_error_fractions, output_influence)) / output_total
        if output_total > 1e-12
        else float("nan")
    )

    return {
        "n_nodes": len(nodes),
        "n_edges": len(links),
        "n_logit_nodes": sum(1 for node in nodes if is_logit_node(node)),
        "n_token_nodes": sum(1 for node in nodes if is_token_node(node)),
        "n_error_nodes": sum(1 for node in nodes if is_error_node(node)),
        "n_feature_nodes": sum(
            1 for node in nodes if not is_logit_node(node) and not is_token_node(node) and not is_error_node(node)
        ),
        "replacement_score": float(replacement_score),
        "completeness_score": float(completeness_score),
    }


def summarize_scores(rows: list[dict[str, object]]) -> dict[str, object]:
    replacement_values = [float(row["replacement_score"]) for row in rows]
    completeness_values = [float(row["completeness_score"]) for row in rows]
    return {
        "n_circuits": len(rows),
        "mean_replacement_score": sum(replacement_values) / len(replacement_values) if replacement_values else float("nan"),
        "mean_completeness_score": sum(completeness_values) / len(completeness_values) if completeness_values else float("nan"),
    }


def list_leaf_circuit_dirs(root: Path) -> list[Path]:
    leaf_dirs: set[Path] = set()
    for json_path in root.rglob("*.json"):
        leaf_dirs.add(json_path.parent)
    return sorted(leaf_dirs)


def resolve_group_dirs(circuits_dir: Path) -> tuple[Path, list[Path]]:
    direct_jsons = sorted(circuits_dir.glob("*.json"))
    if direct_jsons:
        return circuits_dir.parent, [circuits_dir]
    return circuits_dir, list_leaf_circuit_dirs(circuits_dir)


def output_relative_group_dir(circuits_dir: Path, group_dir: Path, output_base_root: Path) -> Path:
    if group_dir == circuits_dir:
        parent_name = group_dir.parent.name
        if parent_name.startswith("results_"):
            return Path(parent_name.removeprefix("results_")) / group_dir.name
        return Path(group_dir.name)
    return group_dir.relative_to(output_base_root)


def score_one_group(group_dir: Path, *, max_files_per_group: int | None, max_iter: int) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    circuit_paths = sorted(group_dir.glob("*.json"))
    if max_files_per_group is not None:
        circuit_paths = circuit_paths[:max_files_per_group]

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for idx, path in enumerate(circuit_paths, start=1):
        print(f"  [{idx}/{len(circuit_paths)}] scoring {path.name}")
        try:
            payload = load_circuit_payload(path)
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            scores = compute_scores_from_circuit_json(payload, max_iter=max_iter)
            rows.append(
                {
                    "file_path": str(path),
                    "slug": metadata.get("slug", path.stem),
                    "fen": metadata.get("prompt"),
                    "move_uci": metadata.get("target_move"),
                    "random_trace_index": metadata.get("random_trace_index"),
                    **scores,
                }
            )
        except Exception as exc:
            failures.append({"file_path": str(path), "error": str(exc)})
    return rows, failures


def main() -> None:
    args = parse_args()
    circuits_dir = args.circuits_dir.resolve()
    output_root = ensure_dir(args.output_root.resolve())
    output_base_root, group_dirs = resolve_group_dirs(circuits_dir)
    if args.max_groups is not None:
        group_dirs = group_dirs[: args.max_groups]
    if not group_dirs:
        raise FileNotFoundError(f"No circuit groups found under {circuits_dir}")

    all_group_rows: list[dict[str, object]] = []
    for group_idx, group_dir in enumerate(group_dirs, start=1):
        rel_group = output_relative_group_dir(circuits_dir, group_dir, output_base_root)
        print(f"[{group_idx}/{len(group_dirs)}] scoring group {rel_group}")
        rows, failures = score_one_group(
            group_dir,
            max_files_per_group=args.max_files_per_group,
            max_iter=args.max_iter,
        )
        group_output_dir = ensure_dir(output_root / rel_group)
        save_csv(group_output_dir / "circuit_scores_cases.csv", rows)
        summary = {
            "group_dir": str(group_dir),
            "relative_group_dir": str(rel_group),
            **summarize_scores(rows),
            "n_failures": len(failures),
            "failures": failures,
        }
        save_json(group_output_dir / "circuit_scores_summary.json", summary)
        all_group_rows.append(
            {
                "relative_group_dir": str(rel_group),
                "n_circuits": summary["n_circuits"],
                "mean_replacement_score": summary["mean_replacement_score"],
                "mean_completeness_score": summary["mean_completeness_score"],
                "n_failures": summary["n_failures"],
            }
        )

    save_csv(output_root / "all_groups_summary.csv", all_group_rows)
    save_json(
        output_root / "all_groups_summary.json",
        {
            "circuits_dir": str(circuits_dir),
            "n_groups": len(all_group_rows),
            "groups": all_group_rows,
        },
    )


if __name__ == "__main__":
    main()

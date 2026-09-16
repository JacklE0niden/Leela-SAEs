from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lm_saes.circuit.bt4_eval_common import (  # noqa: E402
    DEFAULT_VJP_BATCH_SIZE,
    EvalCase,
    build_graph_for_case,
    ensure_dir,
    graph_score_payload,
    load_bt4_bundle,
    load_cases,
    save_csv,
    save_json,
    slugify_case,
)

DEFAULT_DATASET_PATH = Path(
    os.environ.get("CHESS_DATASET_PATH", REPO_ROOT / "data" / "chess_master_data")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BT4 circuit sufficiency scores.")
    parser.add_argument("--cases", default=None, help="Path to JSON/JSONL/CSV/TXT cases file.")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        help="Fallback chess dataset path when --cases is not provided.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to ./output.")
    parser.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--combo-id", default="k_30_e_16", help="BT4 SAE combo id.")
    parser.add_argument("--sae-series", default="BT4-exp128")
    parser.add_argument("--side", default="both", choices=["q", "k", "both"])
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--max-feature-nodes", type=int, default=1024)
    parser.add_argument("--max-n-logits", type=int, default=1)
    parser.add_argument("--desired-logit-prob", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vjp-batch-size", type=int, default=DEFAULT_VJP_BATCH_SIZE)
    parser.add_argument("--order-mode", default="positive", choices=["positive", "negative", "abs", "move_pair"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "output")

    cases = load_eval_cases(args)
    bundle = load_bt4_bundle(device=args.device, combo_id=args.combo_id, sae_series=args.sae_series)

    rows: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        slug = slugify_case(case, idx)
        print(f"[{idx + 1}/{len(cases)}] tracing {slug}")
        graph, _, resolved_case = build_graph_for_case(
            bundle=bundle,
            case=case,
            side=args.side,
            slug=slug,
            max_feature_nodes=args.max_feature_nodes,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            vjp_batch_size=args.vjp_batch_size,
            order_mode=args.order_mode,
            save_activation_info=False,
        )
        scores = graph_score_payload(graph)
        rows.append(
            {
                "case_index": idx,
                "label": resolved_case.label,
                "fen": resolved_case.fen,
                "move_uci": resolved_case.move_uci,
                "side": args.side,
                "n_selected_features": int(len(graph.selected_features)),
                **scores,
            }
        )

    replacement_values = [float(row["replacement_score"]) for row in rows]
    completeness_values = [float(row["completeness_score"]) for row in rows]
    summary = {
        "n_cases": len(rows),
        "combo_id": args.combo_id,
        "sae_series": args.sae_series,
        "side": args.side,
        "max_feature_nodes": args.max_feature_nodes,
        "mean_replacement_score": sum(replacement_values) / len(replacement_values) if replacement_values else float("nan"),
        "mean_completeness_score": sum(completeness_values) / len(completeness_values) if completeness_values else float("nan"),
        "cases_csv": str(output_dir / "circuit_scores_cases.csv"),
    }

    save_csv(output_dir / "circuit_scores_cases.csv", rows)
    save_json(output_dir / "circuit_scores_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

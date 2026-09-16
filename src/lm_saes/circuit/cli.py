"""Command-line entry point for BT4 sparse circuit tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS_PATH = REPO_ROOT / "ui" / "src" / "config" / "circuit-trace-defaults.json"
with DEFAULTS_PATH.open(encoding="utf-8") as defaults_file:
    TRACE_DEFAULTS = json.load(defaults_file)

BT4_DEFAULT_SAE_COMBO = "k_30_e_16"
BT4_SAE_COMBOS = tuple(f"k_{k}_e_{e}" for k in (30, 60, 90) for e in (16, 32))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuit-trace",
        description="Trace a BT4 move through Lorsa and Transcoder features and save the circuit as JSON.",
    )
    parser.add_argument("--fen", required=True, help="Position in Forsyth-Edwards Notation.")
    parser.add_argument("--move", required=True, dest="move_uci", help="Target move in UCI form, for example e2e4.")
    parser.add_argument(
        "--negative-move",
        dest="negative_move_uci",
        help="Comparison move in UCI form; required when --order-mode=both.",
    )
    parser.add_argument(
        "--combo",
        default=BT4_DEFAULT_SAE_COMBO,
        choices=sorted(BT4_SAE_COMBOS),
        help="Transcoder/Lorsa checkpoint combo.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--tc-root", type=Path, help="Override the selected combo's Transcoder directory.")
    parser.add_argument("--lorsa-root", type=Path, help="Override the selected combo's Lorsa directory.")
    parser.add_argument("--side", default=TRACE_DEFAULTS["side"], choices=("q", "k", "both"))
    parser.add_argument(
        "--order-mode",
        default="positive",
        choices=("abs", "positive", "negative", "both"),
        help="Attribution ordering. 'both' compares --move against --negative-move.",
    )
    parser.add_argument("--max-feature-nodes", type=int, default=int(TRACE_DEFAULTS["max_feature_nodes"]))
    parser.add_argument("--node-threshold", type=float, default=float(TRACE_DEFAULTS["node_threshold"]))
    parser.add_argument("--edge-threshold", type=float, default=float(TRACE_DEFAULTS["edge_threshold"]))
    parser.add_argument("--batch-size", type=int, default=int(TRACE_DEFAULTS["batch_size"]))
    parser.add_argument("--vjp-batch-size", type=int, default=int(TRACE_DEFAULTS["vjp_batch_size"]))
    parser.add_argument(
        "--mixed-precision-edges",
        action=argparse.BooleanOptionalAction,
        default=bool(TRACE_DEFAULTS["mixed_precision_edges"]),
    )
    parser.add_argument(
        "--save-activation-info",
        action=argparse.BooleanOptionalAction,
        default=bool(TRACE_DEFAULTS["save_activation_info"]),
    )
    parser.add_argument("--max-act-times", type=int, default=None)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", default="mechinterp")
    parser.add_argument("--sae-series", default="BT4-exp128")
    parser.add_argument("--output", type=Path, help="Output JSON path. A timestamped path is used by default.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def default_output_path(fen: str, move_uci: str, order_mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fen_id = hashlib.sha256(fen.encode("utf-8")).hexdigest()[:10]
    filename = f"trace_{order_mode}_{move_uci}_{fen_id}_{timestamp}.json"
    return REPO_ROOT / "circuit_trace_results" / filename


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.order_mode == "both" and not args.negative_move_uci:
        parser.error("--negative-move is required when --order-mode=both")

    sae_root = Path(os.environ.get("BT4_SAE_ROOT", REPO_ROOT / "result_BT4"))
    tc_root = args.tc_root or sae_root / "tc" / args.combo
    lorsa_root = args.lorsa_root or sae_root / "lorsa" / args.combo
    output_path = args.output or default_output_path(args.fen, args.move_uci, args.order_mode)
    backend_order_mode = "move_pair" if args.order_mode == "both" else args.order_mode
    side = "both" if args.order_mode == "both" else args.side

    # Keep --help lightweight: importing the model stack happens only for a real trace.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from server.circuits_service import run_circuit_trace

    graph_data = run_circuit_trace(
        prompt=args.fen,
        move_uci=args.move_uci,
        negative_move_uci=args.negative_move_uci,
        device=args.device,
        tc_base_path=str(tc_root),
        lorsa_base_path=str(lorsa_root),
        side=side,
        max_feature_nodes=args.max_feature_nodes,
        batch_size=args.batch_size,
        vjp_batch_size=args.vjp_batch_size,
        mixed_precision_edges=args.mixed_precision_edges,
        order_mode=backend_order_mode,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        sae_series=args.sae_series,
        act_times_max=args.max_act_times,
        save_activation_info=args.save_activation_info,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        log_level=args.log_level,
        sae_combo_id=args.combo,
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Circuit trace saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

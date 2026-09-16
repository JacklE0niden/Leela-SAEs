#!/usr/bin/env python3
"""Render top activation boards for a circuit-taxonomy evidence feature."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import chess
import chess.svg


ACTIVATED_COLOR = "#ff6b6b99"
SOURCE_COLOR = "#4dabf799"
BOTH_COLOR = "#9775fa99"
TARGET_COLOR = "#ffd43b77"
MOVE_ARROW_COLOR = "#2f9e44cc"
Z_ARROW_COLOR = "#1c7ed6aa"


def load_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        items: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            items.append(item)
        return items

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    raise ValueError("Input must be a JSON object, JSON array of objects, or JSONL objects")


def parse_square(name: str | None) -> chess.Square | None:
    if not name:
        return None
    try:
        return chess.parse_square(name)
    except ValueError:
        return None


def square_name(square: chess.Square | None) -> str:
    return chess.square_name(square) if square is not None else "?"


def piece_role(board: chess.Board, square_name_value: str, side_to_move: str | None) -> str:
    square = parse_square(square_name_value)
    if square is None:
        return f"{square_name_value}: invalid square"
    piece = board.piece_at(square)
    if piece is None:
        return f"{square_name_value}: empty"

    if side_to_move == "w":
        side = "Own" if piece.color == chess.WHITE else "Opponent"
    elif side_to_move == "b":
        side = "Own" if piece.color == chess.BLACK else "Opponent"
    else:
        side = "White" if piece.color == chess.WHITE else "Black"

    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    return f"{square_name_value}: {side} {names[piece.piece_type]}"


def feature_matches(
    item: dict[str, Any],
    feature: str | None,
    dictionary_name: str | None,
    feature_index: int | None,
    feature_index_in_circuit: int | None,
) -> bool:
    if feature:
        if "#" in feature:
            dictionary, index = feature.rsplit("#", 1)
            return item.get("dictionary_name") == dictionary and str(item.get("feature_index")) == index
        return (
            str(item.get("feature_index")) == feature
            or str(item.get("feature_index_in_circuit")) == feature
            or str(item.get("node_id")) == feature
            or str(item.get("label")) == feature
        )

    if dictionary_name is not None and item.get("dictionary_name") != dictionary_name:
        return False
    if feature_index is not None and item.get("feature_index") != feature_index:
        return False
    if feature_index_in_circuit is not None and item.get("feature_index_in_circuit") != feature_index_in_circuit:
        return False
    return dictionary_name is not None or feature_index is not None or feature_index_in_circuit is not None


def find_feature(
    items: list[dict[str, Any]],
    feature: str | None,
    dictionary_name: str | None,
    feature_index: int | None,
    feature_index_in_circuit: int | None,
) -> dict[str, Any]:
    matches = [
        item
        for item in items
        if feature_matches(item, feature, dictionary_name, feature_index, feature_index_in_circuit)
    ]
    if not matches:
        raise ValueError("No feature matched the requested identifier")
    if len(matches) > 1:
        ids = ", ".join(
            f"{item.get('dictionary_name')}#{item.get('feature_index')}[{item.get('feature_index_in_circuit')}]"
            for item in matches[:8]
        )
        raise ValueError(f"Feature identifier is ambiguous; matched {len(matches)} items: {ids}")
    return matches[0]


def sample_fill(sample: dict[str, Any]) -> dict[chess.Square, str]:
    activated = {
        sq
        for sq in (parse_square(entry.get("square")) for entry in sample.get("top_activated_squares", []))
        if sq is not None
    }
    sources = {
        sq
        for sq in (parse_square(entry.get("source_square")) for entry in sample.get("top_z_pairs", []))
        if sq is not None
    }
    targets = {
        sq
        for sq in (parse_square(entry.get("target_square")) for entry in sample.get("top_z_pairs", []))
        if sq is not None
    }

    fill: dict[chess.Square, str] = {}
    for square in targets:
        fill[square] = TARGET_COLOR
    for square in sources:
        fill[square] = SOURCE_COLOR
    for square in activated:
        fill[square] = BOTH_COLOR if square in sources else ACTIVATED_COLOR
    return fill


def sample_arrows(sample: dict[str, Any], max_z_arrows: int, max_move_arrows: int) -> list[chess.svg.Arrow]:
    arrows: list[chess.svg.Arrow] = []
    for move_entry in sample.get("top_moves", [])[:max_move_arrows]:
        uci = move_entry.get("uci")
        if not isinstance(uci, str) or len(uci) < 4:
            continue
        tail = parse_square(uci[:2])
        head = parse_square(uci[2:4])
        if tail is not None and head is not None:
            arrows.append(chess.svg.Arrow(tail, head, color=MOVE_ARROW_COLOR))

    for pair in sample.get("top_z_pairs", [])[:max_z_arrows]:
        source = parse_square(pair.get("source_square"))
        target = parse_square(pair.get("target_square"))
        if source is not None and target is not None:
            arrows.append(chess.svg.Arrow(source, target, color=Z_ARROW_COLOR))
    return arrows


def orientation_for(sample: dict[str, Any], mode: str) -> chess.Color:
    if mode == "white":
        return chess.WHITE
    if mode == "black":
        return chess.BLACK
    return chess.BLACK if sample.get("side_to_move") == "b" else chess.WHITE


def move_text(sample: dict[str, Any]) -> str:
    moves = []
    for move in sample.get("top_moves", [])[:5]:
        uci = move.get("uci")
        prob = move.get("prob")
        if isinstance(prob, (int, float)):
            moves.append(f"{uci} ({prob:.3f})")
        else:
            moves.append(str(uci))
    return ", ".join(moves) if moves else "none"


def z_pair_text(sample: dict[str, Any], limit: int) -> str:
    pairs = []
    for pair in sample.get("top_z_pairs", [])[:limit]:
        source = pair.get("source_square")
        target = pair.get("target_square")
        value = pair.get("value")
        if isinstance(value, (int, float)):
            pairs.append(f"{source}->{target} ({value:.3f})")
        else:
            pairs.append(f"{source}->{target}")
    return ", ".join(pairs) if pairs else "none"


def activation_text(sample: dict[str, Any]) -> str:
    entries = []
    for entry in sample.get("top_activated_squares", []):
        square = entry.get("square")
        value = entry.get("value")
        if isinstance(value, (int, float)):
            entries.append(f"{square} ({value:.3f})")
        else:
            entries.append(str(square))
    return ", ".join(entries) if entries else "none"


def render_sample(sample: dict[str, Any], max_z_arrows: int, max_move_arrows: int, orientation: str) -> str:
    board = chess.Board(sample["fen"])
    side_to_move = sample.get("side_to_move")
    activated_names = [entry.get("square") for entry in sample.get("top_activated_squares", [])]
    roles = [piece_role(board, square, side_to_move) for square in activated_names if isinstance(square, str)]

    svg = chess.svg.board(
        board,
        size=430,
        orientation=orientation_for(sample, orientation),
        fill=sample_fill(sample),
        arrows=sample_arrows(sample, max_z_arrows=max_z_arrows, max_move_arrows=max_move_arrows),
    )

    value = sample.get("value", sample.get("wdl_value"))
    value_text = f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"
    header = (
        f"sample {sample.get('sample_index')} | context {sample.get('context_idx')} | "
        f"side_to_move={side_to_move} | value={value_text}"
    )

    return f"""
    <section class="sample">
      <h2>{html.escape(header)}</h2>
      <div class="sample-body">
        <div class="board">{svg}</div>
        <dl>
          <dt>FEN</dt><dd><code>{html.escape(sample.get("fen", ""))}</code></dd>
          <dt>Activated</dt><dd>{html.escape(activation_text(sample))}</dd>
          <dt>Activated piece roles</dt><dd>{html.escape("; ".join(roles) if roles else "none")}</dd>
          <dt>Top moves</dt><dd>{html.escape(move_text(sample))}</dd>
          <dt>Top z-pairs</dt><dd>{html.escape(z_pair_text(sample, max_z_arrows))}</dd>
        </dl>
      </div>
    </section>
    """


def safe_id(feature: dict[str, Any]) -> str:
    dictionary = str(feature.get("dictionary_name", "feature")).replace("/", "_")
    index = str(feature.get("feature_index", feature.get("feature_index_in_circuit", "unknown")))
    return f"{dictionary}_{index}".replace(" ", "_")


def render_html(
    feature: dict[str, Any],
    max_samples: int,
    max_z_arrows: int,
    max_move_arrows: int,
    orientation: str,
) -> str:
    samples = feature.get("top_activation_samples", [])[:max_samples]
    title = f"{feature.get('dictionary_name')} #{feature.get('feature_index')}"
    sample_html = "\n".join(
        render_sample(
            sample,
            max_z_arrows=max_z_arrows,
            max_move_arrows=max_move_arrows,
            orientation=orientation,
        )
        for sample in samples
        if isinstance(sample, dict) and sample.get("fen")
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)} boards</title>
<style>
  body {{ margin: 24px; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f8fa; }}
  h1 {{ margin: 0 0 8px; font-size: 22px; }}
  h2 {{ margin: 0 0 12px; font-size: 15px; }}
  code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
  .meta, .legend {{ margin: 0 0 16px; color: #52606d; }}
  .legend span {{ display: inline-flex; align-items: center; margin-right: 16px; gap: 6px; }}
  .swatch {{ width: 14px; height: 14px; border: 1px solid #9aa5b1; display: inline-block; }}
  .sample {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin: 16px 0; }}
  .sample-body {{ display: grid; grid-template-columns: minmax(320px, 430px) minmax(280px, 1fr); gap: 18px; align-items: start; }}
  .board svg {{ max-width: 100%; height: auto; display: block; }}
  dl {{ margin: 0; display: grid; grid-template-columns: 120px 1fr; gap: 8px 12px; }}
  dt {{ font-weight: 650; color: #334e68; }}
  dd {{ margin: 0; }}
  @media (max-width: 820px) {{
    .sample-body {{ grid-template-columns: 1fr; }}
    dl {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="meta">layer={html.escape(str(feature.get("layer")))} | type={html.escape(str(feature.get("feature_type")))} | feature_index_in_circuit={html.escape(str(feature.get("feature_index_in_circuit")))}</p>
  <p class="legend">
    <span><i class="swatch" style="background:{ACTIVATED_COLOR}"></i>activated</span>
    <span><i class="swatch" style="background:{SOURCE_COLOR}"></i>z-source</span>
    <span><i class="swatch" style="background:{TARGET_COLOR}"></i>z-target</span>
    <span>green arrows: top moves</span>
    <span>blue arrows: z-pairs</span>
  </p>
  {sample_html}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path, help="Feature evidence JSON or JSONL")
    parser.add_argument("--feature", default=None, help="Feature id: dictionary#feature_index, feature_index, feature_index_in_circuit, node_id, or label")
    parser.add_argument("--dictionary-name", default=None)
    parser.add_argument("--feature-index", type=int, default=None)
    parser.add_argument("--feature-index-in-circuit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--max-z-arrows", type=int, default=3)
    parser.add_argument("--max-move-arrows", type=int, default=1)
    parser.add_argument("--orientation", choices=["side-to-move", "white", "black"], default="side-to-move")
    args = parser.parse_args()

    items = load_items(args.evidence)
    feature = find_feature(
        items,
        feature=args.feature,
        dictionary_name=args.dictionary_name,
        feature_index=args.feature_index,
        feature_index_in_circuit=args.feature_index_in_circuit,
    )
    output_path = args.out or args.evidence.with_name(f"{args.evidence.stem}.{safe_id(feature)}.boards.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(
            feature,
            max_samples=max(1, args.max_samples),
            max_z_arrows=max(0, args.max_z_arrows),
            max_move_arrows=max(0, args.max_move_arrows),
            orientation=args.orientation,
        ),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

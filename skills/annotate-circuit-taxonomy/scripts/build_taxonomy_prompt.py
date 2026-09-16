#!/usr/bin/env python3
"""Build a compact taxonomy-labeling prompt from feature evidence JSON/JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TAXONOMY_LABELS = [
    "Det",
    "Src",
    "Tgt",
    "Val",
    "Cap",
    "Pro",
    "Mov",
    "Tac",
    "Reg",
    "Spa",
    "Uninterpretable",
]


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


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path, help="Feature evidence JSON or JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of evidence items")
    args = parser.parse_args()

    items = load_items(args.evidence)
    if args.limit is not None:
        items = items[: args.limit]

    print("# Circuit taxonomy labeling task")
    print()
    print("Classify each feature into exactly one taxonomy label.")
    print("Allowed labels:", ", ".join(TAXONOMY_LABELS))
    print("Be conservative: choose Uninterpretable when evidence is weak or inconsistent.")
    print()
    print("Return JSONL only. Each line must include:")
    print("dictionary_name, feature_index, taxonomy, confidence, rationale, evidence_summary.")
    print("Keep rationale to one sentence.")
    print()
    print("## Evidence")
    for index, item in enumerate(items, start=1):
        print(f"{index}. {compact_json(item)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

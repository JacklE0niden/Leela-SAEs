from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from lm_saes.circuit.cli import main


def test_cli_writes_trace_json_and_translates_pair_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    fake_service = types.ModuleType("server.circuits_service")

    def fake_run_circuit_trace(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"metadata": {"slug": "test-trace"}, "nodes": [], "links": []}

    fake_service.run_circuit_trace = fake_run_circuit_trace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "server.circuits_service", fake_service)
    output = tmp_path / "trace.json"

    exit_code = main(
        [
            "--fen",
            "8/8/8/8/8/8/4K3/7k w - - 0 1",
            "--move",
            "e2e3",
            "--negative-move",
            "e2f2",
            "--order-mode",
            "both",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls[0]["order_mode"] == "move_pair"
    assert calls[0]["side"] == "both"
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"]["slug"] == "test-trace"


def test_cli_requires_negative_move_for_pair_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--fen",
                "8/8/8/8/8/8/4K3/7k w - - 0 1",
                "--move",
                "e2e3",
                "--order-mode",
                "both",
                "--output",
                str(tmp_path / "trace.json"),
            ]
        )

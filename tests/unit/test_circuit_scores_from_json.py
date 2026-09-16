import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "faithfulness_validation"
    / "circuit_scores"
    / "circuit_scores_from_json.py"
)
SPEC = importlib.util.spec_from_file_location("circuit_scores_from_json", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compute_scores_from_circuit_json = MODULE.compute_scores_from_circuit_json


def test_circuit_scores_separate_token_and_error_influence() -> None:
    payload = {
        "nodes": [
            {"node_id": "token", "feature_type": "embedding"},
            {"node_id": "error", "feature_type": "attn_error"},
            {"node_id": "feature", "feature_type": "lorsa"},
            {"node_id": "logit", "feature_type": "logit", "token_prob": 1.0},
        ],
        "links": [
            {"source": "token", "target": "feature", "weight": 3.0},
            {"source": "error", "target": "feature", "weight": 1.0},
            {"source": "feature", "target": "logit", "weight": 1.0},
        ],
    }

    scores = compute_scores_from_circuit_json(payload, max_iter=10)

    assert scores["replacement_score"] == pytest.approx(0.75)
    assert scores["completeness_score"] == pytest.approx(11 / 12)
    assert scores["n_feature_nodes"] == 1

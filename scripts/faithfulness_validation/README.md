# Faithfulness validation

This directory contains the runnable evaluation code used by the
`exp/60ICLR/Faithfulness_validation` experiments in the research repository.
Only source code is included here; notebooks, generated figures, model
checkpoints, circuit JSON data, and evaluation outputs are intentionally
excluded.

## Evaluation groups

- `saes/kl.py`: replacement-versus-mean-ablation distribution metrics.
- `saes/top1_agreement.py`: top-1 legal-move agreement after intervention.
- `saes/puzzle.py`: puzzle move accuracy for original, replacement, and mean-ablation models.
- `saes/feature_perturbation*.py`: feature-direction perturbation evaluations.
- `attribution_graphs/in_circuit_steering_vs_random.py`: graph features versus random baselines.
- `attribution_graphs/influence_in_circuit_steering_pearson.py`: attribution influence versus causal steering effects.
- `attribution_graphs/edge_steering_activation_change.py`: graph-edge weight versus downstream activation change.
- `circuit_scores/`: circuit sufficiency/completeness scoring from saved JSON or fresh traces.

Run scripts from the repository root with the project environment, for example:

```bash
uv run python scripts/faithfulness_validation/saes/kl.py --help
uv run python scripts/faithfulness_validation/circuit_scores/circuit_scores_from_json.py --help
```

## Data configuration

The open-source version does not contain private `/inspire/...` paths. Configure
local data with command-line arguments or these environment variables:

- `BT4_SAE_ROOT`: root containing the Transcoder and LoRSA checkpoints.
- `CHESS_DATASET_PATH`: Hugging Face chess dataset saved on disk.
- `CHESS_PUZZLE_CSV`: puzzle CSV containing FEN and solution-move columns.
- `BT4_CIRCUITS_DIR`: directory containing saved circuit JSON files.

Large evaluations require the BT4 model, matching SAE checkpoints, and a CUDA
device. The JSON-only circuit score scripts do not load the model and can run on
CPU.

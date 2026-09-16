# Tracing the Internal Computation of a Chess Transformer

This repository accompanies **Tracing the Internal Computation of a Chess Transformer**. It provides an end-to-end circuit-tracing framework for LC0 BT4: Transcoders and Low-Rank Sparse Attention (Lorsa) decompose MLP and attention computations into sparse features, and attribution graphs trace information flow from the chessboard through those features to move logits.

## What this repository provides

The circuit-tracing pipeline replaces LC0 BT4's MLP and attention computations with Transcoders and Lorsa features, then constructs complete attribution graphs connecting board inputs to move logits. Lorsa is adapted to LC0's Smolgen-biased bidirectional attention, while the bilinear policy head is traced through its query and key sides before the resulting graphs are merged.

The repository also includes tools for inspecting feature activations and Lorsa z-patterns, assigning chess-structured interpretations, and organizing graph features into semantic supernodes. The resulting graphs support both position-level case studies and aggregate analysis. In the paper, they reveal internally parallel, move-specific computations and progressive concentration of attribution on the selected move's source and target squares across layers.

## Circuit tracing quickstart

Circuit tracing has two supported entry points. Both use the same tracing implementation and produce graph JSON that can be reopened on the **Circuits** page.

### WebUI (recommended)

```bash
uv sync
cp server/.env.example server/.env
cp ui/.env.example ui/.env
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env
```

In a second terminal:

```bash
cd ui
bun install
bun run dev --port 5173
```

Open [http://localhost:5173/play-game#circuit-tracing](http://localhost:5173/play-game#circuit-tracing). The **Circuit Tracing** navigation item points to the tracing panel inside the Play page: load an SAE combo, enter or play to a position, provide a UCI move, and run a positive, negative, or paired trace.

### Command line

```bash
uv run circuit-trace \
  --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  --move e2e4 \
  --combo k_30_e_16 \
  --order-mode positive
```

The command writes a timestamped JSON file under `circuit_trace_results/`. Use `--output path/to/trace.json` to choose the destination. For a paired comparison, add `--order-mode both --negative-move <uci>`.

Before tracing, provide the BT4 base checkpoint and one Transcoder/Lorsa combo as described in [Checkpoint layout](#1-prepare-the-checkpoints). Circuit tracing normally requires a CUDA GPU.

## Example: Attribution graph for multiple strategic cues

<p align="center">
  <img src="figures/example.svg" alt="Attribution graph for BT4's Ne5 decision" width="700" />
</p>

This case study groups features from the complete attribution graph into semantic clusters. The graph indicates that BT4's `Ne5` decision integrates several strategic and tactical signals:

- `e5` is identified as protected by the pawn on `d4`.
- `Ne5` interacts with the `Qf7+` threat to create mating pressure.
- `Ne5` supports the subsequent development of `Bg2`.
- After `Ne5`, the `...Bb7` diagonal no longer attacks the knight.
- Features capture relevant opponent-response considerations, including the plausible reply `Qe7`.
- Other features encode files where an own rook or queen is initially blocked by a pawn but becomes exposed after a diagonal pawn capture, supporting the opening of an attacking file.

## Installation

From the repository root, run:

```bash
uv sync
```

For the visualization UI, install the frontend packages from `ui/`:

```bash
cd ui
bun install
cd ..
```

## Pretrained Transcoder and Lorsa weights

Layer-wise checkpoints trained on `lc0/BT4-1024x15x32h` are hosted on Hugging Face. Use these checkpoints to run circuit tracing, inspect sparse features, reproduce attribution-graph analyses, or export reasoning-path summaries without retraining.

| Component | Hugging Face repo |
|-----------|-------------------|
| **Transcoder (TC)** | [JacklE0niden/lc0-BT4-tc](https://huggingface.co/JacklE0niden/lc0-BT4-tc) |
| **Lorsa** | [JacklE0niden/lc0-BT4-lorsa](https://huggingface.co/JacklE0niden/lc0-BT4-lorsa) |

Each repo is organized by **combo** directories (e.g. `k_30_e_16`, `k_30_e_32`, …) with per-layer folders `L0` … `L14`. See the model cards for full layout and `huggingface_hub` examples ([tc](https://huggingface.co/JacklE0niden/lc0-BT4-tc), [lorsa](https://huggingface.co/JacklE0niden/lc0-BT4-lorsa)).

**Download one combo locally** (example: `k_30_e_16`) with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="JacklE0niden/lc0-BT4-tc",
    local_dir="result_BT4/tc",
    allow_patterns="k_30_e_16/*",
)
snapshot_download(
    repo_id="JacklE0niden/lc0-BT4-lorsa",
    local_dir="result_BT4/lorsa",
    allow_patterns="k_30_e_16/*",
)
```

Use these as the Transcoder / Lorsa roots (layers `L0`–`L14` live under each combo), e.g. `result_BT4/tc/k_30_e_16` and `result_BT4/lorsa/k_30_e_16`. Pass `--tc-root` / `--lorsa-root` to `examples/generate_reasoning_pathway.py` if they differ from the script defaults. The WebUI backend reads the same layout from `result_BT4/` by default; set `BT4_SAE_ROOT` to override it.

You still need the **BT4 base model** in TransformerLens format (`BT4.pt` under `models/lc0/`) to run the model. Set `LC0_BT4_CHECKPOINT` if it lives elsewhere. If you do not have it yet, build it from ONNX as described under [Training custom sparse replacement models](#training-custom-sparse-replacement-models), or obtain a compatible `BT4.pt` by other means.

## Circuit tracing tutorial

### 1. Prepare the checkpoints

For the default `k_30_e_16` trace, the repository expects this layout:

```text
models/lc0/BT4.pt
result_BT4/
├── tc/k_30_e_16/L0 ... L14
└── lorsa/k_30_e_16/L0 ... L14
```

You can keep the checkpoints elsewhere by adding these values to `server/.env`:

```dotenv
LC0_BT4_CHECKPOINT=models/lc0/BT4.pt
BT4_SAE_ROOT=result_BT4
PRELOAD_CIRCUIT_MODELS=false
```

`PRELOAD_CIRCUIT_MODELS=false` keeps backend startup fast. The selected checkpoint combo is loaded from the UI when you click **Load / Reload**. Set it to `true` only when you want the default combo loaded during server startup.

### Paths to configure after cloning

All runtime defaults are repository-relative. Put backend settings in `server/.env` and the frontend URL in `ui/.env`; do not edit Python or TypeScript source paths. The two bold rows are the only path settings required for circuit tracing.

| Setting | Repository-local default | Change it when… |
|---|---|---|
| **`LC0_BT4_CHECKPOINT`** | `models/lc0/BT4.pt` | the BT4 base checkpoint is stored elsewhere |
| **`BT4_SAE_ROOT`** | `result_BT4` | Transcoder/Lorsa combo folders are stored elsewhere |
| `CHESS_DATASET_PATH` | `data/chess_master_data` | using dataset-backed feature or tactic tools |
| `STOCKFISH_PATH` | `stockfish` from `PATH` | Stockfish is not installed on the executable search path |
| `BT4_ACTIVATION_ROOT` | `activations/BT4` | logit-lens mean-ablation activations are stored elsewhere |
| `LC0_T82_CHECKPOINT` | `models/lc0/T82.pt` | using the optional T82 model from another location |
| `SEARCHLESS_CHESS_MODEL_ROOT` | `models/searchless_chess` | using the optional searchless-chess implementation from another location |
| `MONGO_URI`, `MONGO_DB` | local MongoDB / `mechinterp` | enabling persistent Features, Dictionaries, and Bookmarks |
| `VITE_BACKEND_URL` | `http://localhost:3000` | the browser should call a backend on another host or port |

For searchless chess, the more specific `SEARCHLESS_CHESS_CHECKPOINT` and `SEARCHLESS_CHESS_BEHAVIORAL_CHECKPOINT` variables can override individual files below `SEARCHLESS_CHESS_MODEL_ROOT`. For LC0 models, `LC0_MODEL_ROOT` can override the shared `models/lc0` directory, while `LC0_BT4_CHECKPOINT` and `LC0_T82_CHECKPOINT` take precedence for individual checkpoints. See `server/.env.example` for a copy-ready configuration.

### 2. Start the backend and frontend

Copy the example environment files. The example uses `MONGO_MOCK=true`, so circuit tracing starts without an external database. Set `MONGO_MOCK=false` and configure `MONGO_URI` when you want persistent **Features**, **Dictionaries**, and **Bookmarks**:

```bash
cp server/.env.example server/.env
cp ui/.env.example ui/.env
```

Start the backend from the repository root:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env
```

In another terminal, start the UI:

```bash
cd ui
bun run dev --port 5173
```

Open <http://localhost:5173/play-game#circuit-tracing>. The root URL redirects to the circuit-tracing panel in **Play Game**, which is also exposed as **Circuit Tracing** in the navigation bar.

### 3. Trace a move in the UI

1. On **Play Game**, select an SAE combo and click **Load / Reload**. Loading all 15 Transcoder and Lorsa layers can take time and normally requires a CUDA GPU.
2. Enter a FEN or play to the position you want to analyze.
3. In **Circuit Trace Analysis**, enter moves in UCI form, such as `e2e4`.
4. Choose a trace mode:
   - **Positive Trace** follows features that promote the positive move.
   - **Negative Trace** follows features that suppress the negative move.
   - **Both Trace** compares a positive and negative move and requires both inputs.
5. Use **Settings** to control graph size and pruning. The shared defaults are `4096` maximum feature nodes, `0.8` node threshold, and `0.65` edge threshold.
6. Click a graph node to inspect its feature, activation board, and incoming/outgoing connections. Pin nodes with the platform modifier key, and save the raw graph JSON when you want a reusable artifact.

The trace request is synchronous, while the UI polls `/circuit_trace/logs` to show progress. Completed results are cached under `circuit_trace_results/` so they can be recovered after a backend restart.

### 4. Reopen and summarize a circuit

- **Circuits** uploads saved circuit JSON and provides detailed graph inspection, feature cards, activation overlays, and graph comparison.
- **Semantic Supernode Graph** opens a bundled example or a semantic-supernode JSON proposal and renders the sparse circuit as higher-level Det/Src/Tgt/Mov/Tac reasoning units.

### Circuit tracing API

The UI calls the same backend endpoints directly. After preloading the combo, a minimal positive trace is:

```bash
curl -X POST http://localhost:3000/circuit/preload_models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"lc0/BT4-1024x15x32h","sae_combo_id":"k_30_e_16"}'

curl -X POST http://localhost:3000/circuit_trace \
  -H 'Content-Type: application/json' \
  -d '{
    "fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "move_uci":"e2e4",
    "sae_combo_id":"k_30_e_16",
    "side":"both",
    "order_mode":"positive",
    "max_feature_nodes":4096,
    "node_threshold":0.8,
    "edge_threshold":0.65
  }'
```

## Other experiments

Explore `examples/` for training and analysis patterns. **MongoDB** is recommended for recording configurations and storing analyses. For advanced use, see `src/lm_saes/runners/`.

### Reasoning-path export

After installing the model and sparse checkpoints described above, export a compact reasoning-path summary with:

```bash
python examples/generate_reasoning_pathway.py
```

Use `--tc-root` and `--lorsa-root` when the checkpoints are stored outside the default `result_BT4/` layout. The supporting modules live under `src/reasoning_path/`.

### Training custom sparse replacement models

Use this path when you need custom Transcoder or Lorsa hyperparameters, training data, or ablations. First obtain the BT4 network from LCZero and convert the matching ONNX checkpoint:

```bash
python examples/weight_conversion.py
```

This writes `models/lc0/BT4.pt` by default. It creates the dense BT4 backbone; pretrained sparse replacement checkpoints are downloaded separately from the Hugging Face repositories above. Then generate activations and train the sparse replacements:

```bash
python examples/gen_tc_BT4.py
python examples/train_tc_BT4.py
```

Additional training and evaluation scripts live under `examples/`.

### Other WebUI tools

Analysis results are stored in **MongoDB**. You can browse learned dictionaries and related analyses in the WebUI. The main public workflow uses **Features**, **Dictionaries**, **Bookmarks**, **Circuits**, **Circuit Tracing**, and **Semantic Supernode Graph**. Start the FastAPI backend with:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env
```
Then copy `ui/.env.example` to `ui/.env`, adjust `VITE_BACKEND_URL` if needed, and start the frontend:

```bash
cd ui
bun run dev --port 5173
```

Open <http://localhost:5173/play-game#circuit-tracing> for the primary circuit-tracing workflow. The other navigation items expose learned dictionaries, individual features, saved circuits, and semantic supernode graphs.

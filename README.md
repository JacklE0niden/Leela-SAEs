# Tracing the Thought of a Grandmaster-level Chess-Playing Transformer

This repository contains the code for experiments and analyses in **Tracing the Thought of a Grandmaster-level Chess-Playing Transformer**. Its primary workflow is **circuit tracing**: decompose an LC0/BT4 move into a sparse attribution graph over Transcoder and Lorsa features, inspect the graph, and optionally organize it into semantic supernodes.


## Example: Reasoning Pathway of a Grandmaster-Level Movement by BT4

<p align="center">
  <img src="figures/example.svg" alt="Superhuman performance" width="700" />
</p>

**Interpretation of the reasoning pathway shown in the figure:**

- e5 is identified as protected by the pawn on d4
- Ne5 interacts with the Qf7+ threat to create mating pressure
- Ne5 supports subsequent Bg2 development
- After Ne5, the ...Bb7 diagonal no longer attacks the knight
- The pathway reflects anticipation of the response Qe7
- We find features encoding files where an own rook/queen is blocked by a pawn, but becomes exposed to threaten the opponent king/queen after a diagonal pawn capture. They serve to open up a file for the rook/queen.


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

## Pretrained Transcoder & Lorsa weights (recommended)

Layer-wise checkpoints trained on `lc0/BT4-1024x15x32h` are hosted on Hugging Face. **Use these when you only need to analyze or build reasoning pathways** without retraining.

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

You still need the **BT4 base model** in TransformerLens format (`BT4.pt` under `models/lc0/`) to run the model. Set `LC0_BT4_CHECKPOINT` if it lives elsewhere. If you do not have it yet, build it from ONNX as described under **B) Training your own sparse replacement models → BT4 base checkpoint from ONNX** (or obtain a compatible `BT4.pt` by other means).

---

## Quickstart (typical workflow)

### A) Using pretrained HF checkpoints (primary)

1. Install dependencies (`uv sync`, etc.).
2. Ensure the BT4 base checkpoint is available (`models/lc0/BT4.pt`; see **B) → BT4 base checkpoint from ONNX** if you need to build it).
3. Download Transcoder + Lorsa for one combo from [lc0-BT4-tc](https://huggingface.co/JacklE0niden/lc0-BT4-tc) and [lc0-BT4-lorsa](https://huggingface.co/JacklE0niden/lc0-BT4-lorsa).
4. Generate reasoning pathways, for example:

```bash
python examples/generate_reasoning_pathway.py
```

Adjust `--tc-root` / `--lorsa-root` if your directories differ from the script defaults.

### B) Training your own sparse replacement models

Use this path when you need **custom** Transcoders / Lorsa (hyperparameters, data, or ablations). Pretrained HF weights are still listed above for the common case.

Relevant scripts live under `examples/` and `src/path_generation/`. You will likely need to edit model name, layer index, output paths, and other settings.

#### BT4 base checkpoint from ONNX

`weight_conversion.py` builds the PyTorch **base model** checkpoint (`BT4.pt`) used by `HookedTransformer` / LC0 loading. This is **not** how you download pretrained sparse replacements (those are the Hugging Face repos above)—it only produces the dense BT4 backbone.

1. Obtain the BT4 network from LCZero (e.g. `BT4-1024x15x32h-swa-6147500.pb.gz` from [big-transformers](https://storage.lczero.org/files/networks-contrib/big-transformers/)) and export or obtain the matching **ONNX** expected by `examples/weight_conversion.py`.

2. Place the ONNX file at `models/lc0/BT4-1024x15x32h-swa-6147500.onnx`, or pass a different location with `--onnx-path`.

3. Run:

```bash
python examples/weight_conversion.py
```

This writes `models/lc0/BT4.pt`. If your layout differs, set `LC0_BT4_CHECKPOINT`; no source edit is required.

After `BT4.pt` is in place, continue with sparse replacement training:

#### 1) Generate activations

```bash
python examples/gen_tc_BT4.py
```

#### 2) Train Transcoder / Lorsa

```bash
python examples/train_tc_BT4.py
```

(Other Lorsa / evaluation scripts are also under `examples/`.)

#### 3) Generate reasoning pathways

```bash
python examples/generate_reasoning_pathway.py
```

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
| `CHESS_DATASET_PATH` | `data/chess_master_data` | using dataset-backed feature, tactic, or faithfulness tools |
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

Open <http://localhost:5173>. The root URL redirects to **Play Game**, the main circuit-tracing workflow.

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
- **Interaction Circuit** uploads interaction CSV files and analyzes how steering source features changes target-feature activations.

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

### What “Interaction Circuit” means

**Interaction Circuit is not the search/MCTS circuit tracer.** It consumes feature-interaction CSVs and uses `/interaction/analyze_node_interaction` to intervene on selected source nodes and measure target-node activation changes. The search-based implementation in this repository is `/play_game_with_search` plus `/search_trace/files/{filename}`; its former **Search Circuits** page is intentionally not exposed in the streamlined navigation because the primary product path is sparse attribution circuit tracing.

## Other experiments

Explore `examples/` for training and analysis patterns. **MongoDB** is recommended for recording configurations and storing analyses. For advanced use, see `src/lm_saes/runners/`.

The paper-only intervention batch utility does not contain a built-in experiment directory list. Supply each input directory explicitly, repeating `--folder` as needed:

```bash
python src/path_generation/generate_feature_interventions_from_json.py \
  --folder outputs/experiment-a \
  --folder outputs/experiment-b \
  --top-n 400
```

### Visualizing learned dictionaries and reasoning pathways

Analysis results are stored in **MongoDB**. You can browse learned dictionaries and related analyses in the WebUI. The streamlined navigation contains only **Features**, **Dictionaries**, **Bookmarks**, **Circuits**, **Play Game**, **Semantic Supernode Graph**, and **Interaction Circuit**. Start the FastAPI backend with:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env
```
Then copy `ui/.env.example` to `ui/.env`, adjust `VITE_BACKEND_URL` if needed, and start the frontend:

```bash
cd ui
bun run dev --port 5173
```

That's it! You can now go to http://localhost:5173 to visualize the learned dictionary and its features.

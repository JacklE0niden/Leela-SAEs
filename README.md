# Tracing the Thought of a Grandmaster-level Chess-Playing Transformer
This repository contains the code for experiments and analyses in **Tracing the Thought of a Grandmaster-level Chess-Playing Transformer**. It focuses on sparse representation learning for chess Transformers (e.g., LC0/BT4), including Transcoders / Lorsa, and on generating reasoning pathways.


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

If you want to use the visualization tools, you also need to install the required packages for the frontend:

```bash
bun install
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

Use these as the Transcoder / Lorsa roots (layers `L0`–`L14` live under each combo), e.g. `result_BT4/tc/k_30_e_16` and `result_BT4/lorsa/k_30_e_16`. Pass `--tc-root` / `--lorsa-root` to `examples/generate_reasoning_pathway.py` if they differ from the script defaults.

You still need the **BT4 base model** in TransformerLens format (`BT4.pt` under `models/lc0/`) to run the model. If you do not have it yet, build it from ONNX as described under **B) Training your own sparse replacement models → BT4 base checkpoint from ONNX** (or obtain a compatible `BT4.pt` by other means).

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

2. Place the ONNX file at `models/lc0/BT4-1024x15x32h-swa-6147500.onnx` (or edit `DEFAULT_ONNX_PATH` in `examples/weight_conversion.py`).

3. Run:

```bash
python examples/weight_conversion.py
```

This writes `models/lc0/BT4.pt`. If your layout differs, align `Project_root` / paths in `TransformerLens/transformer_lens/loading_from_pretrained.py` with your setup.

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

### Launch experiments & visualization

Explore `examples/` for training and analysis patterns. **MongoDB** is recommended for recording configurations and storing analyses. For advanced use, see `src/lm_saes/runners/`.

### Visualizing learned dictionaries and reasoning pathways

Analysis results are stored in **MongoDB**. You can browse learned dictionary and related analyses in WebUI frontend. The frontend also includes a page where you can **upload reasoning-pathway CSV files** to inspect reasoning pathways interactively. Start the FastAPI backend with:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env
```
Then, copy the ui/.env.example file to ui/.env and modify the BACKEND_URL to fit your server settings, and start the frontend by running the following command:

```bash
cd ui
bun dev --port 5173
```

That's it! You can now go to http://localhost:5173 to visualize the learned dictionary and its features.


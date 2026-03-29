# Tracing the Thought of a Grandmaster-level Chess-Playing Transformer
This repository contains the code for experiments and analyses in **Tracing the Thought of a Grandmaster-level Chess-Playing Transformer**. It focuses on sparse representation learning for chess Transformers (e.g., LC0/BT4), including Transcoders / Lorsa, and on generating reasoning pathways.


## Example: Reasoning Pathway of a Grandmaster-Level Movement by BT4

<p align="center">
  <img src="figures/superhuman.svg" alt="Superhuman performance" width="700" />
</p>

**Key insights from the reasoning pathway:**

- Protected by the pawn on d4  
- Combines with Qf7+ to create a mating threat  
- Supports the development of Bg2  
- Prevents ...Bb7 attacking the knight  
- Anticipates the opponent response Qe7


## Download the BT4 model and convert weights

1) Download the BT4 network file from LCZero:
- `BT4-1024x15x32h-swa-6147500.pb.gz` from [the LCZero "big-transformers" directory](https://storage.lczero.org/files/networks-contrib/big-transformers/)

2) Convert the BT4 ONNX model into a PyTorch checkpoint used by this repo.

Place your ONNX file at:

- `models/lc0/BT4-1024x15x32h-swa-6147500.onnx`

Then run:

```bash
python examples/weight_conversion.py
```

This will write:

- `models/lc0/BT4.pt`

Notes:
- The conversion script intentionally uses **repo-relative paths** (no machine-specific absolute paths) to keep the repo anonymous and reproducible.
- If your ONNX filename differs, edit `DEFAULT_ONNX_PATH` inside `examples/weight_conversion.py`.


## Installation

From the repository root, run:

```bash
uv sync
```

If you want to use the visualization tools, you also need to install the required packages for the frontend:

```bash
bun install
```

## Quickstart (typical workflow)

A typical workflow is:

- **Generate activations**
- **Train Transcoder / Lorsa**
- **Generate reasoning pathways**

Relevant scripts live under `examples/` and `src/path_generation/`. You will likely need to edit the model name, layer index, output paths, and other settings to match your setup.

### 1) Generate activations

Example scripts are in `examples/` (e.g., `examples/gen_tc_BT4.py`). After adjusting parameters, run:

```bash
python examples/gen_tc_BT4.py
```

### 2) Train Transcoder / Lorsa

Example scripts are in `examples/` (e.g., `examples/train_tc_BT4.py`):

```bash
python examples/train_tc_BT4.py
```

(If you have Lorsa / evaluation scripts, they are also under `examples/` and can be run similarly.)

### 3) Generate reasoning pathways

The reasoning-pathway generation entrypoint is:

```bash
python src/path_generation/generate_pathway.py
```

### 4) Launch an Experiment
Explore the examples to check the basic usage of training/analyzing SAEs in different configurations. Note a MongoDB is recommended for recording the model/dataset/SAE configurations and required for storing analyses. For more advanced usage, you may explore src/lm_saes/runners folder for the interface for generating activations and training & analyzing SAE variants, and directly write your own variant of training/analyzing script at the runner level.

Visualizing the Learned Dictionary
The analysis results will be saved using MongoDB, and you can use the provided visualization tools to visualize the learned dictionary. First, start the FastAPI server by running the following command:

uvicorn server.app:app --port 24577 --env-file server/.env
Then, copy the ui/.env.example file to ui/.env and modify the BACKEND_URL to fit your server settings (by default, it's http://localhost:24577), and start the frontend by running the following command:

cd ui
bun dev --port 24576
That's it! You can now go to http://localhost:24576 to visualize the learned dictionary and its features.

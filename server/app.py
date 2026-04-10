# NEW HEADER
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io
from functools import lru_cache
from typing import Any, Optional, Tuple, List, Dict
from pathlib import Path

import msgpack
import numpy as np
import plotly.graph_objects as go
import torch
try:
    import umap  # type: ignore
except ImportError:
    umap = None
    print("WARNING: umap-learn not found, UMAP endpoints will be disabled")
from datasets import Dataset
from fastapi import FastAPI, Response, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

try:
    from .constants import (
        BT4_MODEL_NAME,
        BT4_TC_BASE_PATH,
        BT4_LORSA_BASE_PATH,
        BT4_SAE_COMBOS,
        BT4_DEFAULT_SAE_COMBO,
        get_bt4_sae_combo,
    )
except ImportError:
    from constants import (
        BT4_MODEL_NAME,
        BT4_TC_BASE_PATH,
        BT4_LORSA_BASE_PATH,
        BT4_SAE_COMBOS,
        BT4_DEFAULT_SAE_COMBO,
        get_bt4_sae_combo,
    )
try:
    from torchvision import transforms
except ImportError:
    transforms = None
    print("WARNING: torchvision not found, image processing will be disabled")

from lm_saes.backend import LanguageModel
from lm_saes.config import MongoDBConfig, SAEConfig
from lm_saes.database import MongoClient
from lm_saes.resource_loaders import load_dataset_shard, load_model
from lm_saes.sae import SparseAutoEncoder
import subprocess
import json
import tempfile
import os
import time

from .umap_utils import compute_decoder_weights_umap_for_name, get_sae_decoder_weights_umap

import random
import chess

try:
    from transformer_lens import HookedTransformer
    HOOKED_TRANSFORMER_AVAILABLE = True
except ImportError:
    HookedTransformer = None
    HOOKED_TRANSFORMER_AVAILABLE = False
    print("WARNING: transformer_lens not found, HookedTransformer will not be available")

from lm_saes.lc0_mapping.lc0_mapping import (
    idx_to_uci_mappings,
    get_mapping_index,
)
from lm_saes.circuit.leela_board import LeelaBoard
from move_evaluation import evaluate_move_quality

# Interaction functions are now implemented directly in this file

try:
    from tactic_features import analyze_tactic_features, validate_fens
    from lm_saes import LowRankSparseAttention
    TACTIC_FEATURES_AVAILABLE = True
except ImportError:
    analyze_tactic_features = None
    validate_fens = None
    LowRankSparseAttention = None
    TACTIC_FEATURES_AVAILABLE = False
    print("WARNING: tactic_features not found, tactic analysis will not be available")

try:
    from activation import get_activated_features_at_position
    ACTIVATION_MODULE_AVAILABLE = True
except ImportError:
    get_activated_features_at_position = None
    ACTIVATION_MODULE_AVAILABLE = False
    print("WARNING: activation module not found, get_features_at_position endpoint will not be available")

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_TRACE_OUTPUT_DIR = Path("search_trace_outputs")

client = MongoClient(MongoDBConfig())
sae_series = os.environ.get("SAE_SERIES", "default")
tokenizer_only = os.environ.get("TOKENIZER_ONLY", "false").lower() == "true"
if tokenizer_only:
    print("WARNING: Tokenizer only mode is enabled, some features may not be available")

# Remove global caches in favor of LRU cache
# sae_cache: dict[str, SparseAutoEncoder] = {}
# lm_cache: dict[str, LanguageModel] = {}
# dataset_cache: dict[tuple[str, int, int], Dataset] = {}


@lru_cache(maxsize=8)
def get_model(name: str) -> LanguageModel:
    """Load and cache a language model.

    Args:
        name: Name of the model to load

    Returns:
        LanguageModel: The loaded model

    Raises:
        ValueError: If the model is not found
    """
    cfg = client.get_model_cfg(name)
    if cfg is None:
        raise ValueError(f"Model {name} not found")
    cfg.tokenizer_only = tokenizer_only
    return load_model(cfg)


@lru_cache(maxsize=16)
def get_dataset(name: str, shard_idx: int = 0, n_shards: int = 1) -> Dataset:
    """Load and cache a dataset shard.

    Args:
        name: Name of the dataset
        shard_idx: Index of the shard to load
        n_shards: Total number of shards

    Returns:
        Dataset: The loaded dataset shard

    Raises:
        AssertionError: If the dataset is not found
    """
    cfg = client.get_dataset_cfg(name)
    assert cfg is not None, f"Dataset {name} not found"
    return load_dataset_shard(cfg, shard_idx, n_shards)


@lru_cache(maxsize=8)
def get_sae(name: str) -> SparseAutoEncoder:
    """Load and cache a sparse autoencoder.

    Args:
        name: Name of the SAE to load

    Returns:
        SparseAutoEncoder: The loaded SAE

    Raises:
        AssertionError: If the SAE is not found
    """
    path = client.get_sae_path(name, sae_series)
    assert path is not None, f"SAE {name} not found"
    cfg = SAEConfig.from_pretrained(path)
    sae = SparseAutoEncoder.from_config(cfg)
    sae.eval()
    return sae


###############################################################################
###############################################################################

CURRENT_BT4_SAE_COMBO_ID: str = BT4_DEFAULT_SAE_COMBO


def _make_combo_cache_key(model_name: str, combo_id: str | None) -> str:
    """Generate cache/log key: different keys for different combos of the same model."""

    if not combo_id:
        return model_name
    return f"{model_name}::{combo_id}"


_hooked_models: Dict[str, Any] = {}
_transcoders_cache: Dict[str, Dict[int, SparseAutoEncoder]] = {}
_lorsas_cache: Dict[str, Any] = {}  # combo_key -> List[LowRankSparseAttention]
_replacement_models_cache: Dict[str, Any] = {}  # combo_key -> ReplacementModel
_single_sae_cache: Dict[str, Any] = {}  # cache_key -> SAE (Lorsa or Transcoder)

_loading_logs: Dict[str, list] = {}  # combo_key -> [log1, log2, ...]

import threading

_global_loading_lock = threading.Lock()
_hooked_model_loading_lock = threading.Lock()  # Lock specifically for HookedTransformer model loading
_hooked_model_loading_status: Dict[str, bool] = {}  # model_name -> is_loading
_hooked_model_loading_condition = threading.Condition(_hooked_model_loading_lock)  # Condition variable used to wait for loading completion

_loading_locks: Dict[str, threading.Lock] = {}  # combo_key -> Lock
_loading_status: Dict[str, dict] = {}  # combo_key -> {"is_loading": bool}
_cancel_loading: Dict[str, bool] = {}
def get_hooked_model(model_name: str = 'lc0/BT4-1024x15x32h'):
    """Get or load a HookedTransformer model.

    This currently only supports BT4 and uses a global cache plus loading locks.
    """
    global _hooked_models, _hooked_model_loading_lock, _hooked_model_loading_status, _hooked_model_loading_condition
    
    # Always use the BT4 model
    model_name = 'lc0/BT4-1024x15x32h'
    
    # First check the shared SAE preload cache (only for the model itself; SAE combos are ignored)
    if SAE_PRELOAD_SERVICE_AVAILABLE and get_cached_models is not None:
        cached_hooked_model, _, _, _ = get_cached_models(model_name)
        if cached_hooked_model is not None:
            print(f"✅ Retrieved HookedTransformer model from shared SAE preload cache: {model_name}")
            return cached_hooked_model
    
    # Use the condition variable and lock to guard the model loading process
    with _hooked_model_loading_condition:
        # Check local cache (another thread may have loaded the model while we were waiting)
        if model_name in _hooked_models:
            print(f"✅ Retrieved HookedTransformer model from local cache: {model_name}")
            return _hooked_models[model_name]
        
        # Check whether the model is already being loaded
        if _hooked_model_loading_status.get(model_name, False):
            print(f"⏳ Detected that model {model_name} is currently loading; waiting for completion...")
            # Wait until the model finishes loading (maximum 60 seconds)
            max_wait_time = 60
            start_time = time.time()
            while _hooked_model_loading_status.get(model_name, False) and (time.time() - start_time) < max_wait_time:
                _hooked_model_loading_condition.wait(timeout=1.0)
            
            # Check cache again
            if model_name in _hooked_models:
                print(f"✅ Retrieved HookedTransformer model from cache after waiting: {model_name}")
                return _hooked_models[model_name]
            elif (time.time() - start_time) >= max_wait_time:
                raise TimeoutError(f"Timed out while waiting for model {model_name} to load ({max_wait_time} seconds)")
            
            # If the model is still not available after waiting, continue with the loading process
            if model_name in _hooked_models:
                return _hooked_models[model_name]
        
        # Mark as loading
        _hooked_model_loading_status[model_name] = True
        print(f"🔍 Starting to load HookedTransformer model: {model_name} (first load)")
    
    # Perform the actual loading outside the lock (to avoid holding the lock for too long)
    try:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise ValueError("HookedTransformer is not available; please install transformer_lens")
        
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            dtype=torch.float32,
        ).eval()
        
        # After loading completes, use the condition variable to safely update the cache
        with _hooked_model_loading_condition:
            _hooked_models[model_name] = model
            
            # If the shared SAE preload service is available, also update the shared cache
            if SAE_PRELOAD_SERVICE_AVAILABLE and set_cached_models is not None:
                # set_cached_models usually also takes transcoders and LORSAs; here we only cache the model
                _global_hooked_models[model_name] = model
            
            # Mark loading as finished
            _hooked_model_loading_status[model_name] = False
            
            # Notify all waiting threads
            _hooked_model_loading_condition.notify_all()
        
        print(f"✅ HookedTransformer model {model_name} loaded and cached successfully")
        return model
        
    except Exception as e:
        # If loading fails, clear loading state
        with _hooked_model_loading_condition:
            _hooked_model_loading_status[model_name] = False
            _hooked_model_loading_condition.notify_all()
        raise e


def get_cached_sae(sae_path: str, is_lorsa: bool, device: str = "cuda"):
    """Get or load a single SAE instance (with global caching)."""
    global _single_sae_cache
    
    # Use the path as the cache key
    cache_key = f"{sae_path}::{is_lorsa}::{device}"
    
    # Check local cache
    if cache_key not in _single_sae_cache:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise ValueError("HookedTransformer is not available; please install transformer_lens")
        
        print(f"🔍 Loading SAE: {sae_path} (type: {'Lorsa' if is_lorsa else 'Transcoder'})")
        
        if is_lorsa:
            from lm_saes import LowRankSparseAttention
            sae = LowRankSparseAttention.from_pretrained(
                sae_path,
                device=device,
            )
        else:
            sae = SparseAutoEncoder.from_pretrained(
                sae_path,
                dtype=torch.float32,
                device=device,
            )
        
        _single_sae_cache[cache_key] = sae
        print(f"✅ SAE loaded successfully: {sae_path}")
    
    return _single_sae_cache[cache_key]

def get_cached_transcoders_and_lorsas(
    model_name: str,
    sae_combo_id: str | None = None,
) -> Tuple[Optional[Dict[int, SparseAutoEncoder]], Optional[List[LowRankSparseAttention]]]:
    """Get cached transcoders and LORSAs, preferring the shared SAE preload cache."""

    combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
    cache_key = _make_combo_cache_key(model_name, combo_id)

    # First check the shared SAE preload cache
    if SAE_PRELOAD_SERVICE_AVAILABLE and get_cached_models is not None:
        _, cached_transcoders, cached_lorsas, _ = get_cached_models(cache_key)
        if cached_transcoders is not None and cached_lorsas is not None:
            return cached_transcoders, cached_lorsas

    # Fallback to local cache
    global _transcoders_cache, _lorsas_cache
    return _transcoders_cache.get(cache_key), _lorsas_cache.get(cache_key)


def get_available_models():
    """Get the available model list (BT4 only)."""
    return [
        {"name": "lc0/BT4-1024x15x32h", "display_name": "BT4-1024x15x32h"},
    ]


def make_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def trim_minimum(
    origins: list[dict[str, Any] | None],
    feature_acts_indices: np.ndarray,
    feature_acts_values: np.ndarray,
) -> tuple[list[dict[str, Any] | None], np.ndarray, np.ndarray]:
    """Trim multiple arrays to the length of the shortest non-None array.

    Args:
        origins: Origins
        feature_acts_indices: Feature acts indices
        feature_acts_values: Feature acts values

    Returns:
        list: List of trimmed arrays
    """
    # Check whether this is a chess model (by checking whether origins contain FEN data)
    has_fen_data = any(
        origin is not None and origin.get("key") == "fen"
        for origin in origins
        if origin is not None
    )

    if has_fen_data:
        # For chess models, force the minimum length to be at least 64 (number of squares on the board)
        min_length = max(64, feature_acts_indices[-1] + 10)
    else:
        # For other models, use the original logic
        min_length = min(len(origins), feature_acts_indices[-1] + 10)

    feature_acts_indices_mask = feature_acts_indices <= min_length
    return (
        origins[: int(min_length)],
        feature_acts_indices[feature_acts_indices_mask],
        feature_acts_values[feature_acts_indices_mask],
    )


@app.exception_handler(AssertionError)
async def assertion_error_handler(request, exc):
    return Response(content=str(exc), status_code=400)


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_error_handler(request, exc):
    print("CUDA Out of memory. Clearing cache.")
    # Clear LRU caches
    get_model.cache_clear()
    get_dataset.cache_clear()
    get_sae.cache_clear()
    return Response(content="CUDA Out of memory", status_code=500)


@app.get("/dictionaries")
def list_dictionaries():
    return client.list_saes(sae_series=sae_series, has_analyses=True)


@app.get("/dictionaries/{name}/decoder-weights-umap")
def get_dictionary_decoder_weights_umap(name: str) -> dict[str, Any]:
    """Get a 2D UMAP embedding over SAE decoder weights for a dictionary.

    For BT4 chess SAEs (Lorsa / Transcoder), this uses the same combo-based
    loading logic as ``analyze_fen_for_feature`` so that we correctly handle
    both ``W_D`` (Transcoder) and ``W_O`` (Lorsa) decoders. For other
    dictionaries, it falls back to the generic Mongo-backed ``get_sae`` path.
    """
    import re

    try:
        is_bt4_name = name.startswith("BT4_")
        if is_bt4_name:
            layer_match = re.search(r"L(\d+)", name)
            if not layer_match:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot extract layer index from SAE name {name}",
                )
            layer = int(layer_match.group(1))

            is_lorsa_name = "lorsa" in name.lower()
            is_tc_name = "tc" in name.lower() or "transcoder" in name.lower()

            combo_id: str | None = None
            combo_match = re.search(r"k(\d+)_e(\d+)", name)
            if combo_match:
                k_val = combo_match.group(1)
                e_val = combo_match.group(2)
                combo_id = f"k_{k_val}_e_{e_val}"
            else:
                for test_combo_id, test_combo_cfg in BT4_SAE_COMBOS.items():
                    if is_lorsa_name:
                        template = test_combo_cfg.get("lorsa_sae_name_template", "")
                    else:
                        template = test_combo_cfg.get("tc_sae_name_template", "")

                    if template:
                        template_with_layer = template.format(layer=layer)
                        if template_with_layer in name or name.startswith(
                            template_with_layer.split("{")[0]
                        ):
                            combo_id = test_combo_id
                            break

                if combo_id is None:
                    combo_id = BT4_DEFAULT_SAE_COMBO

            combo_cfg = get_bt4_sae_combo(combo_id)

            if is_lorsa_name:
                base_path = combo_cfg["lorsa_base_path"]
                sae_path = f"{base_path}/L{layer}"
                if not os.path.exists(sae_path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Lorsa not found at {sae_path} for layer {layer}",
                    )
                sae = get_cached_sae(sae_path, is_lorsa=True, device=device)
            elif is_tc_name:
                base_path = combo_cfg["tc_base_path"]
                sae_path = f"{base_path}/L{layer}"
                if not os.path.exists(sae_path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Transcoder not found at {sae_path} for layer {layer}",
                    )
                sae = get_cached_sae(sae_path, is_lorsa=False, device=device)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to determine SAE type; name should contain 'lorsa' or 'tc'/'transcoder'",
                )

            umap_data = get_sae_decoder_weights_umap(sae)
        else:
            # Generic path for non-BT4 dictionaries via Mongo-backed configs.
            umap_data = compute_decoder_weights_umap_for_name(name, get_sae)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    embedding = umap_data.get("embedding")
    feature_ids = umap_data.get("feature_ids")

    if isinstance(embedding, np.ndarray):
        embedding_serializable = embedding.tolist()
    else:
        embedding_serializable = embedding

    return {
        "embedding": embedding_serializable,
        "feature_ids": feature_ids,
    }


###############################################################################
# BT4 SAE combo APIs
###############################################################################


@app.get("/sae/combos")
def list_sae_combos() -> Dict[str, Any]:
    """
    Return the available BT4 SAE combos and the default combo.

    These combos are defined in `exp/38mongoanalyses/combos.txt`, and the
    frontend can only select from this set.
    """

    combos = [
        {
            "id": cfg["id"],
            "label": cfg["label"],
            "tc_base_path": cfg["tc_base_path"],
            "lorsa_base_path": cfg["lorsa_base_path"],
        }
        for cfg in BT4_SAE_COMBOS.values()
    ]

    return {
        "default_id": BT4_DEFAULT_SAE_COMBO,
        "current_id": CURRENT_BT4_SAE_COMBO_ID,
        "combos": combos,
    }


@app.get("/images/{dataset_name}")
def get_image(dataset_name: str, context_idx: int, image_idx: int, shard_idx: int = 0, n_shards: int = 1):
    dataset = get_dataset(dataset_name, shard_idx, n_shards)
    data = dataset[int(context_idx)]

    image_key = next((key for key in ["image", "images"] if key in data), None)
    if image_key is None:
        return Response(content="Image not found", status_code=404)

    if len(data[image_key]) <= image_idx:
        return Response(content="Image not found", status_code=404)

    image_tensor = data[image_key][image_idx]

    # Convert tensor to PIL Image and then to bytes
    image = transforms.ToPILImage()(image_tensor.to(torch.uint8))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")


@app.get("/dictionaries/{name}/features/{feature_index}")
def get_feature(
    name: str,
    feature_index: str | int,
    feature_analysis_name: str | None = None,
):
    # Parse feature_index if it's a string
    if isinstance(feature_index, str) and feature_index != "random":
        try:
            feature_index = int(feature_index)
        except ValueError:
            return Response(
                content=f"Feature index {feature_index} is not a valid integer",
                status_code=400,
            )
    print(f'{feature_analysis_name = }')
    print(f'{name = }')
    # Get feature data
    feature = (
        client.get_random_alive_feature(
            sae_name=name,
            sae_series=sae_series,
            name=feature_analysis_name,
        )
        if feature_index == "random"
        else client.get_feature(
            sae_name=name,
            sae_series=sae_series,
            index=feature_index)
    )
    
    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in SAE {name}",
            status_code=404,
        )

    analysis = next(
        (
            a for a in feature.analyses
            if a.name == feature_analysis_name or feature_analysis_name is None
        ),
        None,
    )
    if analysis is None:
        return Response(
            content=f"Feature analysis {feature_analysis_name} not found in SAE {name}"
            if feature_analysis_name is not None
            else f"No feature analysis found in SAE {name}",
            status_code=404,
        )

    def process_sample(
        *,
        sparse_feature_acts,
        context_idx,
        dataset_name,
        model_name,
        shard_idx=None,
        n_shards=None,
    ):
        """Process a sample to extract and format feature data.

        Args:
            sparse_feature_acts: Sparse feature activations,
                optional z pattern activations
            decoder_norms: Decoder norms
            context_idx: Context index in the dataset
            dataset_name: Name of the dataset
            model_name: Name of the model
            shard_idx: Index of the dataset shard, defaults to 0
            n_shards: Total number of shards, defaults to 1

        Returns:
            dict: Processed sample data
        """
        model = get_model(model_name)
        
        data = get_dataset(dataset_name, shard_idx, n_shards)[context_idx.item()]

        # Get origins for the features
        origins = model.trace({k: [v] for k, v in data.items()})[0]

        # Process image data if present
        image_key = next(
            (key for key in ["image", "images"] if key in data),
            None,
        )
        if image_key is not None:
            image_urls = [
                f"/images/{dataset_name}?context_idx={context_idx}&"
                f"shard_idx={shard_idx}&n_shards={n_shards}&"
                f"image_idx={img_idx}"
                for img_idx in range(len(data[image_key]))
            ]
            del data[image_key]
            data["images"] = image_urls

        # Trim to matching lengths
        (
            feature_acts_indices,
            feature_acts_values,
            z_pattern_indices,
            z_pattern_values,
        ) = sparse_feature_acts

        origins, feature_acts_indices, feature_acts_values = trim_minimum(
            origins,
            feature_acts_indices,
            feature_acts_values,
        )
        assert (
            origins is not None
            and feature_acts_indices is not None
            and feature_acts_values is not None
        ), "Origins and feature acts must not be None"

        # Detect whether this is a chess model (multiple checks)
        has_fen_data = any(
            origin is not None and origin.get("key") == "fen"
            for origin in origins
            if origin is not None
        )

        # Determine whether this is a chess model from the model or dataset name
        is_chess_model = (
            has_fen_data
            or "chess" in model_name.lower()
            or "lc0" in model_name.lower()
            or "chess" in dataset_name.lower()
            or "lc0" in dataset_name.lower()
        )

        if is_chess_model:
            # For chess models, create a dense activation array of length 64
            dense_feature_acts = np.zeros(64)

            # Enforce dtypes
            feature_acts_indices = np.asarray(feature_acts_indices, dtype=np.int64)
            feature_acts_values = np.asarray(feature_acts_values, dtype=np.float32)

            # Optionally filter out invalid indices
            valid_mask = (feature_acts_indices >= 0) & (feature_acts_indices < 64)
            feature_acts_indices = feature_acts_indices[valid_mask]
            feature_acts_values = feature_acts_values[valid_mask]

            # Then either loop with zip or write in a vectorized way
            for idx, val in zip(feature_acts_indices, feature_acts_values):
                dense_feature_acts[idx] = val

            # Ensure FEN data exists
            if "fen" not in data:
                # If there is no FEN in the data, try to extract it from origins
                fen_origins = [
                    origin
                    for origin in origins
                    if origin is not None and origin.get("key") == "fen"
                ]
                if fen_origins:
                    # Use the range from the first FEN origin to slice from the text
                    fen_origin = fen_origins[0]
                    if "range" in fen_origin and "text" in data:
                        start, end = fen_origin["range"]
                        data["fen"] = data["text"][start:end]
                    else:
                        # If there is no range information, fall back to the full text
                        data["fen"] = data.get("text", "")
                else:
                    # If there is no FEN information at all, create a default starting position
                    data["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        else:
            # For other models, use the original logic
            dense_feature_acts = np.zeros(len(origins))

            for i, (idx, val) in enumerate(zip(feature_acts_indices, feature_acts_values)):
                try:
                    # Make sure idx is a valid integer
                    if hasattr(idx, "item"):
                        idx = idx.item()
                    elif hasattr(idx, "__int__"):
                        idx = int(idx)
                    else:
                        idx = int(float(idx))

                    # Make sure val is a valid numeric value
                    if hasattr(val, "item"):
                        val = val.item()
                    elif hasattr(val, "__float__"):
                        val = float(val)
                    else:
                        val = float(val)

                    # Check index bounds
                    if 0 <= idx < len(origins):
                        dense_feature_acts[idx] = val

                except (ValueError, TypeError, IndexError):
                    continue

        # Process text data if present
        if "text" in data:
            text_ranges = [
                origin["range"] for origin in origins
                if origin is not None and origin["key"] == "text"
            ]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]

        # For chess models, use the FEN string as the text
        if is_chess_model:
            data["text"] = data.get("fen", "No FEN data")

        return {
            **data,
            "origins": origins,
            "feature_acts": dense_feature_acts,  # Dense activation array
            "feature_acts_indices": feature_acts_indices,
            "feature_acts_values": feature_acts_values,
            "z_pattern_indices": z_pattern_indices,
            "z_pattern_values": z_pattern_values,
        }
    
    def process_sparse_feature_acts(
        feature_acts_indices: np.ndarray,
        feature_acts_values: np.ndarray,
        z_pattern_indices: np.ndarray | None = None,
        z_pattern_values: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]]:
        """Process sparse feature acts.
        
        Args:
            feature_acts_indices: Feature acts indices
            feature_acts_values: Feature acts values
            z_pattern_indices: Z pattern indices
            z_pattern_values: Z pattern values
        
        TODO: This is really ugly, we should find a better way to do this.
        """

        if feature_acts_indices.size == 0 or feature_acts_indices.shape[1] == 0:
            return


        _, feature_acts_counts = np.unique(
            feature_acts_indices[0],
            return_counts=True,
        )

        _, z_pattern_counts = (
            np.unique(z_pattern_indices[0], return_counts=True)
            if z_pattern_indices is not None
            else (None, None)
        )

        feature_acts_sample_ranges = np.concatenate(
            [[0], np.cumsum(feature_acts_counts)]
        )

        z_pattern_sample_ranges = (
            np.concatenate([[0], np.cumsum(z_pattern_counts)])
            if z_pattern_counts is not None
            else None
        )

        feature_acts_sample_ranges = list(
            zip(feature_acts_sample_ranges[:-1], feature_acts_sample_ranges[1:])
        )

        if z_pattern_sample_ranges is not None:
            z_pattern_sample_ranges = list(
                zip(z_pattern_sample_ranges[:-1], z_pattern_sample_ranges[1:])
            )
            if len(feature_acts_sample_ranges) != len(z_pattern_sample_ranges):
                z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)
        else:
            z_pattern_sample_ranges = [(None, None)] * len(feature_acts_sample_ranges)

        for (feature_acts_start, feature_acts_end), (z_pattern_start, z_pattern_end) in zip(feature_acts_sample_ranges, z_pattern_sample_ranges):
            feature_acts_indices_i = feature_acts_indices[1, feature_acts_start:feature_acts_end]
            feature_acts_values_i = feature_acts_values[feature_acts_start:feature_acts_end]
            z_pattern_indices_i = z_pattern_indices[1:, z_pattern_start:z_pattern_end] if z_pattern_indices is not None else None
            z_pattern_values_i = z_pattern_values[z_pattern_start:z_pattern_end] if z_pattern_values is not None else None

            yield feature_acts_indices_i, feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i


    sample_groups = []
    for sampling in analysis.samplings:
        try:
            # Using zip to process correlated data instead of indexing
            samples = [
                process_sample(
                    sparse_feature_acts=sparse_feature_acts,
                    context_idx=context_idx,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    shard_idx=shard_idx,
                    n_shards=n_shards,
                )
                for sparse_feature_acts, context_idx, dataset_name, model_name, shard_idx, n_shards in zip(
                    process_sparse_feature_acts(
                        sampling.feature_acts_indices,
                        sampling.feature_acts_values,
                        sampling.z_pattern_indices,
                        sampling.z_pattern_values,
                    ),
                    sampling.context_idx,
                    sampling.dataset_name,
                    sampling.model_name,
                    sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.feature_acts_indices),
                    sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.feature_acts_indices),
                )
            ]
            

            sample_groups.append(
                {
                    "analysis_name": sampling.name,
                    "samples": samples,
                }
            )
        except Exception as e:
            # Return a 400 error response if processing this sampling fails
            return Response(
                content=f"Error while processing sampling '{sampling.name}': {str(e)}",
                status_code=400,
            )

    # Normalize interpretation so frontend always receives validation (array); autointerp may omit it
    interpretation = feature.interpretation
    if interpretation is not None and isinstance(interpretation, dict) and "validation" not in interpretation:
        interpretation = {**interpretation, "validation": []}

    # Prepare response
    response_data = {
        "feature_index": feature.index,
        "analysis_name": analysis.name,
        "interpretation": interpretation,
        "dictionary_name": feature.sae_name,
        "decoder_norms": analysis.decoder_norms,
        "decoder_similarity_matrices": analysis.decoder_similarity_matrices,
        "decoder_inner_product_matrices": analysis.decoder_inner_product_matrices,
        "act_times": analysis.act_times,
        "max_feature_act": analysis.max_feature_acts,
        "n_analyzed_tokens": analysis.n_analyzed_tokens,
        "sample_groups": sample_groups,
        "is_bookmarked": client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature.index),
    }

    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}")
def get_dictionary(name: str):
    # Get feature activation times
    feature_activation_times = client.get_feature_act_times(name, sae_series=sae_series)
    if feature_activation_times is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Create histogram of log activation times
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    # Get alive feature count
    alive_feature_count = client.get_alive_feature_count(name, sae_series=sae_series)
    if alive_feature_count is None:
        return Response(content=f"SAE {name} not found", status_code=404)

    # Prepare and return response
    response_data = {
        "dictionary_name": name,
        "feature_activation_times_histogram": [feature_activation_times_histogram],
        "alive_feature_count": alive_feature_count,
    }

    return Response(
        content=msgpack.packb(make_serializable(response_data)),
        media_type="application/x-msgpack",
    )


@app.get("/dictionaries/{name}/analyses")
def get_analyses(name: str):
    """Get all available analyses for a dictionary.

    Args:
        name: Name of the dictionary/SAE

    Returns:
        List of analysis names
    """
    # Get a random feature to check its available analyses
    feature = client.get_random_alive_feature(sae_name=name, sae_series=sae_series)
    if feature is None:
        return Response(content=f"Dictionary {name} not found", status_code=404)

    # Extract unique analysis names from feature
    analyses = list(set(analysis.name for analysis in feature.analyses))
    return analyses


@app.post("/dictionaries/{name}/features/{feature_index}/analyze_fen")
def analyze_fen_for_feature(name: str, feature_index: int, request: dict):
    fen = request.get("fen")
    if not fen:
        raise HTTPException(status_code=400, detail="FEN string must not be empty")
    
    try:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="HookedTransformer is not available; please install transformer_lens",
            )

        # Extract layer index and combo information from the SAE name
        import re
        layer_match = re.search(r"L(\d+)", name)
        if not layer_match:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot extract layer index from SAE name {name}",
            )
        layer = int(layer_match.group(1))
        
        # Determine whether this is a Lorsa or Transcoder SAE
        is_lorsa_name = 'lorsa' in name.lower()
        is_tc_name = 'tc' in name.lower() or 'transcoder' in name.lower()
        
        # Extract combo information from the SAE name (e.g. k30_e16 -> k_30_e_16),
        # or try to match against all known combos
        combo_id = None
        combo_match = re.search(r'k(\d+)_e(\d+)', name)
        if combo_match:
            k_val = combo_match.group(1)
            e_val = combo_match.group(2)
            combo_id = f"k_{k_val}_e_{e_val}"
        else:
            # If we cannot find combo info directly, try to infer it by matching
            # the SAE name against all known combo templates
            for test_combo_id, test_combo_cfg in BT4_SAE_COMBOS.items():
                if is_lorsa_name:
                    template = test_combo_cfg.get("lorsa_sae_name_template", "")
                else:
                    template = test_combo_cfg.get("tc_sae_name_template", "")

                # Try substituting the layer into the template and check for a match
                if template:
                    template_with_layer = template.format(layer=layer)
                    # Allow partial matches because there may be additional suffixes
                    if template_with_layer in name or name.startswith(
                        template_with_layer.split("{")[0]
                    ):
                        combo_id = test_combo_id
                        break

            # Fall back to the default combo if nothing matches
            if combo_id is None:
                combo_id = BT4_DEFAULT_SAE_COMBO
        
        # Get combo configuration
        combo_cfg = get_bt4_sae_combo(combo_id)
        
        # Get model
        model_name = "lc0/BT4-1024x15x32h"
        model = get_hooked_model(model_name)
        
        # Load the SAE according to the combo configuration (using cache)
        if is_lorsa_name:
            # Load Lorsa
            lorsa_base_path = combo_cfg["lorsa_base_path"]
            lorsa_path = f"{lorsa_base_path}/L{layer}"
            
            if not os.path.exists(lorsa_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Lorsa not found at {lorsa_path} for layer {layer}"
                )

            if not HOOKED_TRANSFORMER_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="HookedTransformer is not available; cannot load Lorsa",
                )

            # Load SAE from cache
            sae = get_cached_sae(lorsa_path, is_lorsa=True, device=device)
        elif is_tc_name:
            # Load Transcoder
            tc_base_path = combo_cfg["tc_base_path"]
            tc_path = f"{tc_base_path}/L{layer}"
            
            if not os.path.exists(tc_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Transcoder not found at {tc_path} for layer {layer}"
                )

            # Load SAE from cache
            sae = get_cached_sae(tc_path, is_lorsa=False, device=device)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unable to determine SAE type; name should contain 'lorsa' or 'tc'/'transcoder'",
            )
        
        # Run the model to obtain activations
        with torch.no_grad():
            # Determine which hook to read from
            if is_lorsa_name:
                # Lorsa uses hook_attn_in
                hook_name = f"blocks.{layer}.hook_attn_in"
            else:
                # Transcoder uses resid_mid_after_ln
                hook_name = f"blocks.{layer}.resid_mid_after_ln"
            
            _, cache = model.run_with_cache(fen, prepend_bos=False)            

            if cache is None or len(cache) == 0:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Cache is empty after running the model; cannot retrieve activations. "
                        f"Please check whether the FEN string is valid. FEN: {fen}"
                    ),
                )
            try:
                all_hooks = list(cache.keys())
            except Exception as e:
                all_hooks = []
            
            layer_hooks = [k for k in all_hooks if f"blocks.{layer}" in str(k)]
            if layer_hooks == []:
                # Check whether neighbouring layers have any hooks
                for test_layer in [layer - 1, layer + 1, 0, model.cfg.n_layers - 1]:
                    if 0 <= test_layer < model.cfg.n_layers:
                        test_hooks = [k for k in all_hooks if f"blocks.{test_layer}" in str(k)]
                        if test_hooks:
                            print(
                                f"   - Comparison: layer {test_layer} has {len(test_hooks)} hooks, "
                                f"examples: {test_hooks[:3]}"
                            )
                            break
            
            print(f"   - Expected hook name: {hook_name}")
            
            # Check whether the expected hook exists
            hook_exists = False
            try:
                hook_exists = hook_name in cache
                print(f"   - Does hook exist: {hook_exists}")
            except Exception as e:
                print(f"   - Error while checking hook existence: {e}")
            
            if not hook_exists:
                # Try to find similar hooks
                similar_hooks = [k for k in all_hooks if f"blocks.{layer}" in str(k)]
                # Also search for all hooks that contain "attn" or "resid" (for Lorsa and Transcoder)
                if is_lorsa_name:
                    attn_hooks = [k for k in all_hooks if f"blocks.{layer}" in str(k) and "attn" in str(k).lower()]
                    print(f"   - Hooks containing 'attn': {attn_hooks[:10]}")
                else:
                    resid_hooks = [k for k in all_hooks if f"blocks.{layer}" in str(k) and "resid" in str(k).lower()]
                    print(f"   - Hooks containing 'resid': {resid_hooks[:10]}")
                
                error_detail = (
                    f"Failed to find activations for layer {layer}. "
                    f"SAE type: {'Lorsa' if is_lorsa_name else 'Transcoder'}. "
                    f"Expected hook: {hook_name}. "
                    f"Total number of hooks: {len(all_hooks)}. "
                    f"Hooks containing 'blocks.{layer}': {similar_hooks[:20]}. "
                    f"Example hooks: {all_hooks[:20] if len(all_hooks) > 0 else 'none'}"
                )
                raise HTTPException(status_code=500, detail=error_detail)
            
            activations = cache[hook_name]  # shape: [batch, seq, ...], typically [1, seq_len, d_model]
            
            # Ensure that activations have the correct number of dimensions.
            # Both Lorsa and Transcoder encode methods expect a batch dimension.
            # If the batch dimension is missing, add one.
            if activations.dim() == 1:
                # [d_model] -> [1, d_model]
                activations = activations.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
            elif activations.dim() == 2:
                # [seq_len, d_model] -> [1, seq_len, d_model]
                activations = activations.unsqueeze(0)  # [1, seq_len, d_model]
            # If it is already 3D [batch, seq_len, d_model], use it directly
            
            print(f"   - Activations shape: {activations.shape}")
            
            # For the BT4 model, seq_len is usually 64 squares after FEN input
            seq_len = activations.shape[1] if activations.dim() >= 2 else activations.shape[0]
            print(f"   - Sequence length: {seq_len}")
        
        # Encode using the SAE
        # We already know whether this should be Lorsa or Transcoder,
        # but we also double-check the actual SAE type for safety.
        sae_type_str = str(type(sae))
        is_lorsa = is_lorsa_name or "LowRankSparseAttention" in sae_type_str

        if is_lorsa:
            # Lorsa encoding: get feature activations.
            # The Lorsa encode method expects input of shape [batch, seq_len, d_model].
            feature_acts = sae.encode(
                activations,  # Activations already include the batch dimension
                return_hidden_pre=False,
                return_attention_pattern=False,
            )

            print(f"   - Feature activations shape (after encoding): {feature_acts.shape}")

            # Remove batch dimension
            if feature_acts.dim() == 3:
                feature_acts = feature_acts[0]  # [seq_len, d_sae] - index instead of squeeze for safety
            elif feature_acts.dim() == 2:
                # Already [seq_len, d_sae]; nothing to do
                pass
            else:
                raise ValueError(f"Unexpected feature_acts shape: {feature_acts.shape}")

            # Get activations for the given feature index
            # feature_acts shape: [seq_len, d_sae]
            if feature_acts.dim() == 2:
                # Take activations across all positions, shape: [seq_len]
                feature_activation_values = feature_acts[:, feature_index].detach().cpu().numpy()
            else:
                feature_activation_values = (
                    feature_acts[feature_index].detach().cpu().unsqueeze(0).numpy()
                )

            # Build an array of activations for 64 squares
            seq_len = len(feature_activation_values)
            if seq_len == 64:
                activations_64 = feature_activation_values
            elif seq_len == 1:
                # If there is only one value, broadcast it to all 64 positions
                # This usually happens when the model output has only a single token
                activations_64 = np.full(64, feature_activation_values[0])
            else:
                # If the length is not 64, pad or truncate to 64
                activations_64 = np.zeros(64)
                min_len = min(seq_len, 64)
                activations_64[:min_len] = feature_activation_values[:min_len]

            # Use encode_z_pattern_for_head to compute the Z pattern for this feature.
            # This method computes the Z pattern for the specific head (feature_index),
            # instead of averaging across all heads.
            z_pattern_indices = None
            z_pattern_values = None
            try:
                # Make sure activations are on the correct device
                if activations.device != sae.cfg.device:
                    activations = activations.to(sae.cfg.device)

                # Compute the Z pattern for this feature using encode_z_pattern_for_head.
                # head_idx is feature_index (for Lorsa, each feature corresponds to one head).
                head_idx = torch.tensor([feature_index], device=activations.device)
                z_pattern = sae.encode_z_pattern_for_head(activations, head_idx)
                # z_pattern shape: [n_active_features, q_pos, k_pos], here [1, seq_len, seq_len]

                print(f"   - Z pattern shape: {z_pattern.shape}")

                # Get the Z pattern for all positions of this feature.
                # z_pattern[0] shape: [q_pos, k_pos], i.e. [seq_len, seq_len]
                z_pattern_2d = z_pattern[0]  # [seq_len, seq_len]

                # Find all active positions (non-zero activations)
                active_positions = np.where(activations_64 != 0)[0]

                if len(active_positions) > 0:
                    # For each active position, extract and aggregate its Z pattern
                    all_z_pattern_indices = []
                    all_z_pattern_values = []

                    for pos in active_positions:
                        if pos < z_pattern_2d.shape[0]:
                            # Get the Z pattern from this query position to all key positions
                            z_pattern_for_pos = (
                                z_pattern_2d[pos, :].detach().cpu().numpy()
                            )  # [seq_len]

                            # Find non-zero values (filter out very small values)
                            nonzero_mask = np.abs(z_pattern_for_pos) > 1e-6
                            if np.any(nonzero_mask):
                                nonzero_indices = np.where(nonzero_mask)[0]
                                nonzero_values = z_pattern_for_pos[nonzero_indices]

                                # Add [query_pos, key_pos] pairs
                                for key_pos, value in zip(nonzero_indices, nonzero_values):
                                    all_z_pattern_indices.append([int(pos), int(key_pos)])
                                    all_z_pattern_values.append(float(value))

                    if len(all_z_pattern_indices) > 0:
                        z_pattern_indices = all_z_pattern_indices
                        z_pattern_values = all_z_pattern_values
                        print(
                            f"   - Z pattern: found {len(z_pattern_indices)} non-zero connections"
                        )
                    else:
                        print("   - Z pattern: no non-zero connections found")
                else:
                    print("   - Z pattern: no active positions")

            except Exception as e:
                print(f"   - Error while computing Z pattern: {e}")
                import traceback

                traceback.print_exc()
                z_pattern_indices = None
                z_pattern_values = None

            # Build a sparse representation of activations (return only non-zero values)
            non_zero_mask = activations_64 != 0
            feature_acts_indices = np.where(non_zero_mask)[0].tolist()
            feature_acts_values = activations_64[non_zero_mask].tolist()

            return {
                "feature_acts_indices": feature_acts_indices,
                "feature_acts_values": feature_acts_values,
                "z_pattern_indices": z_pattern_indices,
                "z_pattern_values": z_pattern_values,
            }
        else:
            # Transcoder encoding also requires a batch dimension.
            # The Transcoder encode method expects input of shape [batch, seq_len, d_model].
            encode_result = sae.encode(activations)  # Use activations with batch dimension
            feature_acts = encode_result  # shape: [batch, seq_len, d_sae], usually [1, seq_len, d_sae]

            print(f"   - Feature activations shape (after encoding): {feature_acts.shape}")

            # Remove batch dimension
            if feature_acts.dim() == 3:
                feature_acts = feature_acts[0]  # [seq_len, d_sae]
            elif feature_acts.dim() == 2:
                # Already [seq_len, d_sae]; nothing to do
                pass
            else:
                raise ValueError(f"Unexpected feature_acts shape: {feature_acts.shape}")

            # Get activations for the given feature index
            # feature_acts shape: [seq_len, d_sae]
            if feature_acts.dim() == 2:
                feature_activation_values = feature_acts[:, feature_index].detach().cpu().numpy()
            else:
                feature_activation_values = (
                    feature_acts[feature_index].detach().cpu().unsqueeze(0).numpy()
                )

            # Build an array of activations for 64 squares
            seq_len = len(feature_activation_values)
            if seq_len == 64:
                activations_64 = feature_activation_values
            elif seq_len == 1:
                # If there is only one value, broadcast it to all 64 positions
                activations_64 = np.full(64, feature_activation_values[0])
            else:
                # If the length is not 64, pad or truncate to 64
                activations_64 = np.zeros(64)
                min_len = min(seq_len, 64)
                activations_64[:min_len] = feature_activation_values[:min_len]

            # Build sparse representation
            non_zero_mask = activations_64 != 0
            feature_acts_indices = np.where(non_zero_mask)[0].tolist()
            feature_acts_values = activations_64[non_zero_mask].tolist()

            return {
                "feature_acts_indices": feature_acts_indices,
                "feature_acts_values": feature_acts_values,
                "z_pattern_indices": None,
                "z_pattern_values": None,
            }
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error while analyzing FEN: {str(e)}")


@app.post("/activation/get_features_at_position")
def get_features_at_position(request: dict):
    """
    Get all active features at a given layer and board position.

    Args:
        request: A dictionary with the following fields:
            - fen: FEN string
            - layer: Layer index (0–14)
            - pos: Position index (0–63)
            - component_type: Component type, "attn" or "mlp"
            - model_name: Optional model name, defaults to "lc0/BT4-1024x15x32h"
            - sae_combo_id: Optional SAE combo ID, defaults to the current combo

    Returns:
        A dictionary containing:
        - "attn_features": if component_type is "attn", the active Lorsa features (list)
        - "mlp_features": if component_type is "mlp", the active Transcoder features (list)
        Each feature entry contains:
        - "feature_index": feature index
        - "activation_value": activation value
    """
    try:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="HookedTransformer is not available; please install transformer_lens",
            )

        fen = request.get("fen")
        layer = request.get("layer")
        pos = request.get("pos")
        component_type = request.get("component_type")
        model_name = request.get("model_name", "lc0/BT4-1024x15x32h")
        sae_combo_id = request.get("sae_combo_id")

        if not fen:
            raise HTTPException(status_code=400, detail="FEN string must not be empty")
        if layer is None:
            raise HTTPException(status_code=400, detail="Layer index must not be empty")
        if pos is None:
            raise HTTPException(status_code=400, detail="Position index must not be empty")
        if not component_type:
            raise HTTPException(
                status_code=400,
                detail="component_type is required and must be 'attn' or 'mlp'",
            )

        if component_type not in ["attn", "mlp"]:
            raise HTTPException(
                status_code=400,
                detail="component_type must be 'attn' or 'mlp'",
            )

        # Get model
        model = get_hooked_model(model_name)

        # Get transcoders and LORSAs
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(
            model_name, sae_combo_id
        )

        if cached_transcoders is None or cached_lorsas is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Transcoders/LORSAs are not loaded; please call "
                    "/sae/preload_combo to preload them first"
                ),
            )

        if not ACTIVATION_MODULE_AVAILABLE or get_activated_features_at_position is None:
            raise HTTPException(
                status_code=503,
                detail="The activation module is not available; cannot fetch active features",
            )

        # Call helper function to get the active features
        result = get_activated_features_at_position(
            model=model,
            transcoders=cached_transcoders,
            lorsas=cached_lorsas,
            fen=fen,
            layer=layer,
            pos=pos,
            component_type=component_type,
        )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to get active features: {str(e)}"
        )


@app.post("/dictionaries/{name}/features/{feature_index}/analyze_fen_all_positions")
def analyze_fen_all_positions(name: str, feature_index: int, request: dict):
    fen = request.get("fen")
    if not fen:
        raise HTTPException(status_code=400, detail="FEN string must not be empty")
    
    try:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="HookedTransformer is not available; please install transformer_lens",
            )
        
        import re
        layer_match = re.search(r'L(\d+)', name)
        if not layer_match:
            raise HTTPException(status_code=400, detail=f"Cannot extract layer index from SAE name {name}")
        layer = int(layer_match.group(1))
        
        is_lorsa_name = 'lorsa' in name.lower()
        is_tc_name = 'tc' in name.lower() or 'transcoder' in name.lower()
        
        combo_id = None
        combo_match = re.search(r'k(\d+)_e(\d+)', name)
        if combo_match:
            k_val = combo_match.group(1)
            e_val = combo_match.group(2)
            combo_id = f"k_{k_val}_e_{e_val}"
        else:
            for test_combo_id, test_combo_cfg in BT4_SAE_COMBOS.items():
                if is_lorsa_name:
                    template = test_combo_cfg.get("lorsa_sae_name_template", "")
                else:
                    template = test_combo_cfg.get("tc_sae_name_template", "")
                
                if template:
                    template_with_layer = template.format(layer=layer)
                    if template_with_layer in name or name.startswith(template_with_layer.split('{')[0]):
                        combo_id = test_combo_id
                        break
            
            if combo_id is None:
                combo_id = BT4_DEFAULT_SAE_COMBO
        
        combo_cfg = get_bt4_sae_combo(combo_id)
        
        model_name = "lc0/BT4-1024x15x32h"
        model = get_hooked_model(model_name)
        
        if is_lorsa_name:
            lorsa_base_path = combo_cfg["lorsa_base_path"]
            lorsa_path = f"{lorsa_base_path}/L{layer}"
            
            if not os.path.exists(lorsa_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Lorsa not found at {lorsa_path} for layer {layer}"
                )
            
            sae = get_cached_sae(lorsa_path, is_lorsa=True, device=device)
        elif is_tc_name:
            tc_base_path = combo_cfg["tc_base_path"]
            tc_path = f"{tc_base_path}/L{layer}"
            
            if not os.path.exists(tc_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Transcoder not found at {tc_path} for layer {layer}"
                )
            
            sae = get_cached_sae(tc_path, is_lorsa=False, device=device)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to determine SAE type; name should contain 'lorsa' or 'tc'/'transcoder'"
            )
        
        with torch.no_grad():
            if is_lorsa_name:
                hook_name = f"blocks.{layer}.hook_attn_in"
            else:
                hook_name = f"blocks.{layer}.resid_mid_after_ln"
            
            _, cache = model.run_with_cache(fen, prepend_bos=False)
            
            if hook_name not in cache:
                available_hooks = [k for k in cache.keys() if f"blocks.{layer}" in str(k)]
                raise HTTPException(
                    status_code=500,
                    detail=f"Cannot find activations for layer {layer}. SAE type: {'Lorsa' if is_lorsa_name else 'Transcoder'}. Expected hook: {hook_name}. Available hooks: {available_hooks[:10]}"
                )
            
            activations = cache[hook_name]  # shape: [batch, seq_len, d_model], typically [1, seq_len, d_model]
            
            if activations.dim() == 1:
                activations = activations.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
            elif activations.dim() == 2:
                activations = activations.unsqueeze(0)  # [1, seq_len, d_model]
            
            seq_len = activations.shape[1] if activations.dim() >= 2 else activations.shape[0]
            print(f"Analyzing all positions: FEN={fen}, Layer={layer}, Feature={feature_index}, SeqLen={seq_len}")
        
        sae_type_str = str(type(sae))
        is_lorsa = is_lorsa_name or 'LowRankSparseAttention' in sae_type_str
        
        if is_lorsa:
            feature_acts = sae.encode(
                activations,  # [1, seq_len, d_model]
                return_hidden_pre=False,
                return_attention_pattern=False
            )
            
            if feature_acts.dim() == 3:
                feature_acts = feature_acts[0]  # [seq_len, d_sae]
            
            z_pattern_2d = None
            try:
                if activations.device != sae.cfg.device:
                    activations = activations.to(sae.cfg.device)
                
                head_idx = torch.tensor([feature_index], device=activations.device)
                z_pattern = sae.encode_z_pattern_for_head(activations, head_idx)
                z_pattern_2d = z_pattern[0]  # [seq_len, seq_len]
            except Exception as e:
                print(f"Error calculating z_pattern: {e}")
                import traceback
                traceback.print_exc()
            
            positions_data = []
            for pos in range(min(seq_len, 64)):
                if feature_acts.dim() == 2:
                    pos_activations = feature_acts[pos, feature_index].detach().cpu().item()
                else:
                    pos_activations = feature_acts[feature_index].detach().cpu().item()
                
                activations_64 = np.zeros(64)
                if pos < 64:
                    activations_64[pos] = pos_activations
                
                z_pattern_indices = None
                z_pattern_values = None
                if z_pattern_2d is not None:
                    query_pos = pos
                    if query_pos < z_pattern_2d.shape[0]:
                        key_z_patterns = z_pattern_2d[query_pos, :].detach().cpu().numpy()  # [seq_len]
                        
                        nonzero_mask = np.abs(key_z_patterns) > 1e-6
                        nonzero_indices = np.where(nonzero_mask)[0]
                        if len(nonzero_indices) > 0:
                            z_pattern_indices = [[int(query_pos), int(k_pos)] for k_pos in nonzero_indices if k_pos < 64]
                            z_pattern_values = [float(key_z_patterns[k_pos]) for k_pos in nonzero_indices if k_pos < 64]
                
                positions_data.append({
                    "position": pos,
                    "activations": activations_64.tolist(),
                    "z_pattern_indices": z_pattern_indices,
                    "z_pattern_values": z_pattern_values,
                })
            
            for pos in range(seq_len, 64):
                positions_data.append({
                    "position": pos,
                    "activations": [0.0] * 64,
                    "z_pattern_indices": None,
                    "z_pattern_values": None,
                })
            
            return {
                "positions": positions_data,
                "total_positions": len(positions_data),
                "feature_index": feature_index,
                "layer": layer,
                "sae_type": "Lorsa" if is_lorsa else "Transcoder"
            }
        else:
            encode_result = sae.encode(activations)
            feature_acts = encode_result  # [1, seq_len, d_sae]
            
            if feature_acts.dim() == 3:
                feature_acts = feature_acts[0]  # [seq_len, d_sae]
            
            positions_data = []
            for pos in range(min(seq_len, 64)):
                if feature_acts.dim() == 2:
                    pos_activations = feature_acts[pos, feature_index].detach().cpu().item()
                else:
                    pos_activations = feature_acts[feature_index].detach().cpu().item()
                
                activations_64 = np.zeros(64)
                if pos < 64:
                    activations_64[pos] = pos_activations
                
                positions_data.append({
                    "position": pos,
                    "activations": activations_64.tolist(),
                    "z_pattern_indices": None,
                    "z_pattern_values": None,
                })
            
            for pos in range(seq_len, 64):
                positions_data.append({
                    "position": pos,
                    "activations": [0.0] * 64,
                    "z_pattern_indices": None,
                    "z_pattern_values": None,
                })
            
            return {
                "positions": positions_data,
                "total_positions": len(positions_data),
                "feature_index": feature_index,
                "layer": layer,
                "sae_type": "Transcoder"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing FEN all positions: {str(e)}")


@app.post("/dictionaries/{name}/features/{feature_index}/bookmark")
def add_bookmark(name: str, feature_index: int):
    """Add a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to bookmark

    Returns:
        Success response or error
    """
    try:
        success = client.add_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
        if success:
            return {"message": "Bookmark added successfully"}
        else:
            return Response(content="Feature is already bookmarked", status_code=409)
    except ValueError as e:
        return Response(content=str(e), status_code=404)


@app.delete("/dictionaries/{name}/features/{feature_index}/bookmark")
def remove_bookmark(name: str, feature_index: int):
    """Remove a bookmark for a feature.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature to remove bookmark from

    Returns:
        Success response or error
    """
    success = client.remove_bookmark(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    if success:
        return {"message": "Bookmark removed successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


@app.get("/dictionaries/{name}/features/{feature_index}/bookmark")
def check_bookmark(name: str, feature_index: int):
    """Check if a feature is bookmarked.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature

    Returns:
        Bookmark status
    """
    is_bookmarked = client.is_bookmarked(sae_name=name, sae_series=sae_series, feature_index=feature_index)
    return {"is_bookmarked": is_bookmarked}


@app.get("/bookmarks")
def list_bookmarks(sae_name: Optional[str] = None, sae_series: Optional[str] = None, limit: int = 100, skip: int = 0):
    """List bookmarks with optional filtering.

    Args:
        sae_name: Optional SAE name filter
        sae_series: Optional SAE series filter
        limit: Maximum number of bookmarks to return
        skip: Number of bookmarks to skip (for pagination)

    Returns:
        List of bookmarks
    """
    bookmarks = client.list_bookmarks(sae_name=sae_name, sae_series=sae_series, limit=limit, skip=skip)

    # Convert to dict for JSON serialization
    bookmark_data = []
    for bookmark in bookmarks:
        bookmark_dict = bookmark.model_dump()
        # Convert datetime to ISO string for JSON
        bookmark_dict["created_at"] = bookmark.created_at.isoformat()
        bookmark_data.append(bookmark_dict)

    return {
        "bookmarks": bookmark_data,
        "total_count": client.get_bookmark_count(sae_name=sae_name, sae_series=sae_series),
    }


@app.post("/dictionaries/{name}/features/{feature_index}/interpret")
def interpret_feature(
    name: str,
    feature_index: int,
    type: str,
    custom_interpretation: Optional[str] = None,
):
    """
    Handle feature interpretation: auto-generate (not implemented), save custom, or validate.
    
    Args:
        name: SAE name
        feature_index: Feature index
        type: Interpretation type ("auto" | "custom" | "validate")
        custom_interpretation: Custom interpretation text (required when type=\"custom\")
    
    Returns:
        Interpretation object as a dict
    """
    try:
        # Fetch feature
        feature = client.get_feature(
            sae_name=name,
            sae_series=sae_series,
            index=feature_index
        )
        
        if feature is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature {feature_index} not found in SAE {name}",
            )
        
        if type == "custom":
            # Save custom interpretation
            if not custom_interpretation:
                raise HTTPException(
                    status_code=400,
                    detail="custom_interpretation is required for type=custom",
                )
            
            # FastAPI should already have URL-decoded params; decode again just in case
            import urllib.parse
            decoded_interpretation = urllib.parse.unquote(custom_interpretation)
            
            print("📝 Received interpretation text:")
            print(f"   - Raw: {custom_interpretation}")
            print(f"   - Decoded: {decoded_interpretation}")
            
            # Build interpretation dict (only required fields, to match frontend optional schema)
            interpretation_dict = {
                "text": decoded_interpretation,
                "method": "custom",
                "validation": []
            }
            
            # Save to database
            try:
                client.update_feature(
                    sae_name=name,
                    sae_series=sae_series,
                    feature_index=feature_index,
                    update_data={"interpretation": interpretation_dict}
                )
            except Exception as update_error:
                print(f"Failed to update feature interpretation: {update_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save interpretation: {str(update_error)}"
                )
            
            return interpretation_dict
        
        elif type == "auto":
            raise HTTPException(
                status_code=501,
                detail="Automatic interpretation is not yet implemented. Please use custom interpretation.",
            )
        
        elif type == "validate":
            if not feature.interpretation:
                raise HTTPException(
                    status_code=400,
                    detail="No interpretation available to validate",
                )
            
            interp = feature.interpretation
            print(
                "📖 Reading interpretation text: "
                f"{interp.get('text', '') if isinstance(interp, dict) else getattr(interp, 'text', '')}"
            )
            
            if isinstance(interp, dict):
                result = {
                    "text": interp.get("text", ""),
                    "method": interp.get("method", "unknown"),
                    "validation": interp.get("validation", []),
                }
                if interp.get("passed") is not None:
                    result["passed"] = interp.get("passed")
                if interp.get("complexity") is not None:
                    result["complexity"] = interp.get("complexity")
                if interp.get("consistency") is not None:
                    result["consistency"] = interp.get("consistency")
                return result
            else:
                # If it is an object, try to access attributes
                result = {
                    "text": getattr(interp, "text", ""),
                    "method": getattr(interp, "method", "unknown"),
                    "validation": getattr(interp, "validation", []),
                }
                # Only add optional fields when not None
                passed = getattr(interp, "passed", None)
                if passed is not None:
                    result["passed"] = passed
                complexity = getattr(interp, "complexity", None)
                if complexity is not None:
                    result["complexity"] = complexity
                consistency = getattr(interp, "consistency", None)
                if consistency is not None:
                    result["consistency"] = consistency
                return result
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type: {type}. Must be 'auto', 'custom', or 'validate'"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process interpretation: {str(e)}"
        )


@app.put("/dictionaries/{name}/features/{feature_index}/bookmark")
def update_bookmark(name: str, feature_index: int, tags: Optional[list[str]] = None, notes: Optional[str] = None):
    """Update a bookmark with new tags or notes.

    Args:
        name: Name of the dictionary/SAE
        feature_index: Index of the feature
        tags: Optional new tags for the bookmark
        notes: Optional new notes for the bookmark

    Returns:
        Success response or error
    """
    success = client.update_bookmark(
        sae_name=name, sae_series=sae_series, feature_index=feature_index, tags=tags, notes=notes
    )
    if success:
        return {"message": "Bookmark updated successfully"}
    else:
        return Response(content="Bookmark not found", status_code=404)


# LC0 engine wrapper
class LC0Engine:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def play(self, chess_board):
        try:
            # Use the same interface as the notebook for inference
            fen = chess_board.fen()
            print(f"🔍 Processing FEN: {fen}")

            # Create a LeelaBoard instance to handle mapping
            lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
            lboard.pc_board = chess_board  # Use the existing board state

            with torch.no_grad():
                output, cache = self.model.run_with_cache(fen, prepend_bos=False)
                if isinstance(output, (list, tuple)) and len(output) >= 1:
                    policy_output = output[0]
                else:
                    policy_output = output
                if policy_output.dim() == 2:
                    policy_logits = policy_output[0]
                else:
                    policy_logits = policy_output

            legal_moves = list(chess_board.legal_moves)
            legal_uci_set = set(move.uci() for move in legal_moves)
            sorted_indices = torch.argsort(policy_logits, descending=True)

            top10 = []
            for idx in sorted_indices[:10].tolist():
                uci = lboard.idx2uci(idx)
                logit = float(policy_logits[idx].item())
                top10.append((uci, logit))
            
            print("🔍 Model output debug info:")
            print(f"   - policy_logits shape: {tuple(policy_logits.shape)}")
            print(f"   - number of legal moves: {len(legal_moves)}")
            print("   - Top 10 highest-probability moves (uci, logit):")
            print("     " + ", ".join([f"{uci}:{logit:.4f}" for uci, logit in top10]))

            # Try moves in descending policy order and choose the first legal move
            for rank, idx in enumerate(sorted_indices.tolist(), start=1):
                uci = lboard.idx2uci(idx)
                if uci in legal_uci_set:
                    move = chess.Move.from_uci(uci)
                    print(f"✅ Selected highest-probability legal move: {uci} (rank: {rank}, logit: {policy_logits[idx].item():.4f})")
                    return move

            # If no legal move is found, log and raise an error
            print("❌ Error: model did not find any legal move!")
            print(f"   - Current FEN: {fen}")
            print(f"   - Sample legal moves: {[m.uci() for m in legal_moves[:10]]}")
            print(f"   - Tried top {min(len(sorted_indices), 50)} highest-probability tokens")
            raise ValueError("Model did not find any legal move")

        except Exception as e:
            print(f"❌ LC0Engine.play() failed: {e}")
            raise e


@app.post("/play_game")
def play_game(request: dict):
    """
    Play against the model: given a FEN, return the model's suggested next move (UCI).
    
    Supported modes:
        1. Directly use the neural network policy output (use_search=False, default)
        2. Use MCTS search (use_search=True)
    
    Args:
        request: JSON body with:
            - fen: FEN string (required)
            - use_search: whether to use MCTS search (optional, default False)
            - search_params: search parameters (optional, used when use_search=True)
                - max_playouts: max playouts (default 100)
                - target_minibatch_size: minibatch size (default 8)
                - cpuct: UCT exploration coefficient (default 1.0)
                - max_depth: maximum search depth (default 10)
    """
    fen = request.get("fen")
    use_search = request.get("use_search", False)
    search_params = request.get("search_params", {})
    model_name = "lc0/BT4-1024x15x32h"
    
    save_trace = bool(request.get("save_trace", False))
    trace_output_dir = request.get("trace_output_dir") or str(SEARCH_TRACE_OUTPUT_DIR)
    trace_max_edges_raw = request.get("trace_max_edges", 1000)
    trace_max_edges = None if (trace_max_edges_raw == 0 or trace_max_edges_raw is None) else int(trace_max_edges_raw)

    if not fen:
        raise HTTPException(status_code=400, detail="FEN string must not be empty")
    
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid FEN string")
    
    try:
        # Check whether HookedTransformer is available
        if not HOOKED_TRANSFORMER_AVAILABLE:
            print("❌ Error: HookedTransformer is not available")
            raise HTTPException(
                status_code=503,
                detail="HookedTransformer is not available; please install transformer_lens",
            )
        
        if use_search:
            print(f"🔍 Using MCTS search mode: {fen[:50]}...")
            
            try:
                from search.model_interface import run_mcts_search, set_model_getter
                set_model_getter(get_hooked_model)
            except ImportError as e:
                print(f"❌ Failed to import search module: {e}")
                raise HTTPException(status_code=503, detail="MCTS search module not available")
            
            max_playouts = search_params.get("max_playouts", 100)
            target_minibatch_size = search_params.get("target_minibatch_size", 8)
            cpuct = search_params.get("cpuct", 1.0)
            max_depth = search_params.get("max_depth", 10)
            
            print(f"   Search params: max_playouts={max_playouts}, cpuct={cpuct}, max_depth={max_depth}")
            search_result = run_mcts_search(
                fen=fen,
                max_playouts=max_playouts,
                target_minibatch_size=target_minibatch_size,
                cpuct=cpuct,
                max_depth=max_depth,
                model_name=model_name,
            )
            
            best_move = search_result.get("best_move")
            if not best_move:
                raise ValueError("MCTS search did not find a legal move")
            
            print(f"✅ MCTS search completed: {best_move}, playouts={search_result.get('total_playouts')}")
            
            return {
                "move": best_move,
                "model_used": model_name,
                "search_used": True,
                "search_stats": {
                    "total_playouts": search_result.get("total_playouts"),
                    "max_depth_reached": search_result.get("max_depth_reached"),
                    "root_visits": search_result.get("root_visits"),
                    "top_moves": search_result.get("top_moves", [])[:5],  # return at most first 5
                }
            }
        else:
            model = get_hooked_model(model_name)
            engine = LC0Engine(model)
            move = engine.play(board)
            return {"move": move.uci(), "model_used": model_name, "search_used": False}
        
    except ValueError as e:
        print(f"❌ Model could not find a legal move: {e}")
        raise HTTPException(status_code=400, detail=f"Model could not find a legal move: {str(e)}")
    except Exception as e:
        print(f"❌ Error while computing move: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Move computation failed: {str(e)}")


@app.post("/play_game_with_search")
def play_game_with_search(request: dict):
    """
    Play against the model using MCTS search: given a FEN and search parameters,
    return the model's suggested next move (UCI).
    
    Request body:
        - fen: FEN string
        - max_playouts: max playouts (default 100)
        - target_minibatch_size: target minibatch size (default 8)
        - cpuct: UCT exploration coefficient (default 1.0)
        - max_depth: max search depth (default 10, 0 means unlimited)
        - low_q_exploration_enabled: whether to enable low-Q exploration (default False)
        - low_q_threshold: Q-value threshold for "low Q" (default 0.3)
        - low_q_exploration_bonus: base exploration bonus (default 0.1)
        - low_q_visit_threshold: visit threshold for "under-explored" (default 5)
    """
    fen = request.get("fen")
    # Always use the BT4 model
    model_name = "lc0/BT4-1024x15x32h"
    
    max_playouts = request.get("max_playouts", 100)
    target_minibatch_size = request.get("target_minibatch_size", 8)
    cpuct = request.get("cpuct", 1.0)
    max_depth = request.get("max_depth", 10)
    
    low_q_exploration_enabled = request.get("low_q_exploration_enabled", False)
    low_q_threshold = request.get("low_q_threshold", 0.3)
    low_q_exploration_bonus = request.get("low_q_exploration_bonus", 0.1)
    low_q_visit_threshold = request.get("low_q_visit_threshold", 5)
    
    save_trace = bool(request.get("save_trace", False))
    trace_slug = request.get("trace_slug")
    trace_output_dir = request.get("trace_output_dir") or str(SEARCH_TRACE_OUTPUT_DIR)
    trace_max_edges_raw = request.get("trace_max_edges", 1000)
    trace_max_edges = None if (trace_max_edges_raw == 0 or trace_max_edges_raw is None) else int(trace_max_edges_raw)
    
    if not fen:
        raise HTTPException(status_code=400, detail="FEN string must not be empty")
    
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid FEN string")
    
    try:
        # Check whether HookedTransformer is available
        if not HOOKED_TRANSFORMER_AVAILABLE:
            print("❌ Error: HookedTransformer is not available")
            raise HTTPException(
                status_code=503,
                detail="HookedTransformer is not available; please install transformer_lens",
            )
        
        from search import (
            SearchParams, Search, SimpleBackend, Node, SearchTracer,
            get_wl, get_d, get_m, get_policy,
            policy_tensor_to_move_dict, set_model_getter,
        )
        
        set_model_getter(get_hooked_model)
        
        def model_eval_fn(fen_str: str) -> dict:
            """Model evaluation function returning q, d, m, p."""
            wl = get_wl(fen_str, model_name)
            d = get_d(fen_str, model_name)
            m_tensor = get_m(fen_str, model_name)
            m_value = m_tensor.item() if hasattr(m_tensor, 'item') else float(m_tensor)
            
            policy_tensor = get_policy(fen_str, model_name)
            policy_dict = policy_tensor_to_move_dict(policy_tensor, fen_str)
            
            return {
                'q': wl,
                'd': d,
                'm': m_value,
                'p': policy_dict
            }
        
        params = SearchParams(
            max_playouts=max_playouts,
            target_minibatch_size=target_minibatch_size,
            cpuct=cpuct,
            max_depth=max_depth,
            low_q_exploration_enabled=low_q_exploration_enabled,
            low_q_threshold=low_q_threshold,
            low_q_exploration_bonus=low_q_exploration_bonus,
            low_q_visit_threshold=low_q_visit_threshold,
        )
        
        backend = SimpleBackend(model_eval_fn)
        root_node = Node(fen=fen)
        
        tracer = SearchTracer() if save_trace else None
        search = Search(
            root_node=root_node,
            backend=backend,
            params=params,
            tracer=tracer,
        )
        
        print(f"🔍 Starting MCTS search: max_playouts={max_playouts}, max_depth={max_depth}")
        search.run_blocking()
        
        best_move = search.get_best_move()
        total_playouts = search.get_total_playouts()
        current_max_depth = search.get_current_max_depth()
        
        if best_move is None:
            raise ValueError("Search did not find a legal move")
        
        print(f"✅ MCTS search completed: playouts={total_playouts}, depth={current_max_depth}, best_move={best_move.uci()}")
        
        trace_file_path = None
        if save_trace and tracer:
            trace_file_path = search.export_trace_json(
                output_dir=trace_output_dir,
                max_edges=trace_max_edges,
            )

        response_data = {
            "move": best_move.uci(),
            "model_used": model_name,
            "search_info": {
                "total_playouts": total_playouts,
                "max_depth_reached": current_max_depth,
                "max_depth_limit": max_depth,
            }
        }
        if trace_file_path:
            response_data["trace_file_path"] = trace_file_path
            response_data["trace_filename"] = Path(trace_file_path).name

        return response_data
        
    except ValueError as e:
        print(f"❌ Search could not find a legal move: {e}")
        raise HTTPException(status_code=400, detail=f"Search could not find a legal move: {str(e)}")
    except Exception as e:
        print(f"❌ Error while running search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search_trace/files/{filename}")
def download_search_trace_file(filename: str):
    """Download a saved MCTS search trace file."""
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    target_path = SEARCH_TRACE_OUTPUT_DIR / safe_name
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Trace file not found")
    return FileResponse(
        path=target_path,
        media_type="application/json",
        filename=safe_name,
    )


@app.post("/analyze/board")
def analyze_board(request: dict):
    """Analyze current position using HookedTransformer, and return win rate, draw rate and loss rate for the current player"""
    fen = request.get("fen")
    # force using BT4 model
    model_name = "lc0/BT4-1024x15x32h"
    
    if not fen:
        raise HTTPException(status_code=400, detail="FEN string must not be empty")
    try:
        if not HOOKED_TRANSFORMER_AVAILABLE:
            raise HTTPException(status_code=503, detail="HookedTransformer unavailable, please install transformer_lens")
        
        # use specified model (use cache to avoid duplicate loading)
        model = get_hooked_model(model_name)
        
        with torch.no_grad():
            output, _ = model.run_with_cache(fen, prepend_bos=False)
        
        # model output is a list containing three elements:
        # output[0]: logits, shape [1, 1858]
        # output[1]: WDL, shape [1, 3] - [current player win rate, draw rate, current player loss rate]
        # output[2]: other output, shape [1, 1]
        
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            wdl_tensor = output[1]  # get WDL output
            if wdl_tensor.shape == torch.Size([1, 3]):
                # WDL is already a probability distribution, no need to softmax
                current_player_win = wdl_tensor[0][0].item()  # current player win rate
                draw_prob = wdl_tensor[0][1].item()  # draw rate
                current_player_loss = wdl_tensor[0][2].item()  # current player loss rate
                
                # directly return current player win rate information, no softmax
                # [current player win rate, draw rate, opponent win rate]
                evaluation = [current_player_win, draw_prob, current_player_loss]
            else:
                print(f"WDL output shape incorrect: {wdl_tensor.shape}, expected [1, 3]")
                evaluation = [0.5, 0.2, 0.3]
        else:
            print(f"model output format incorrect, expected list or tuple containing at least 2 elements, got: {type(output)}")
            evaluation = [0.5, 0.2, 0.3]
        
        return {"evaluation": evaluation, "model_used": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"board analysis error: {str(e)}")


@app.get("/models")
def get_models():
    """get available models list"""
    return {"models": get_available_models()}


# import shared SAE combo preload service
try:
    try:
        from .sae_combo_service import (
            clear_cached_sae_resources,
            load_model_and_transcoders,
            get_cached_models,
            set_cached_models,
            _global_hooked_models,
            _global_transcoders_cache,
            _global_lorsas_cache,
            _global_replacement_models_cache,
        )
    except ImportError:
        from sae_combo_service import (
            clear_cached_sae_resources,
            load_model_and_transcoders,
            get_cached_models,
            set_cached_models,
            _global_hooked_models,
            _global_transcoders_cache,
            _global_lorsas_cache,
            _global_replacement_models_cache,
        )
    SAE_PRELOAD_SERVICE_AVAILABLE = True
    _hooked_models = _global_hooked_models
    _transcoders_cache = _global_transcoders_cache
    _lorsas_cache = _global_lorsas_cache
    _replacement_models_cache = _global_replacement_models_cache
except ImportError as e:
    clear_cached_sae_resources = None
    load_model_and_transcoders = None
    get_cached_models = None
    set_cached_models = None
    _global_hooked_models = {}
    _global_transcoders_cache = {}
    _global_lorsas_cache = {}
    _global_replacement_models_cache = {}
    SAE_PRELOAD_SERVICE_AVAILABLE = False
    print(f"WARNING: sae_combo_service not found, SAE combo preloading will not be available: {e}")

# import patching service
try:
    from patching import run_patching_analysis
    PATCHING_SERVICE_AVAILABLE = True
except ImportError:
    run_patching_analysis = None
    PATCHING_SERVICE_AVAILABLE = False
    print("WARNING: patching service not found, patching analysis will not be available")

# import intervention service
try:
    from intervention import run_feature_steering_analysis, run_multi_feature_steering_analysis
    INTERVENTION_SERVICE_AVAILABLE = True
except ImportError:
    run_feature_steering_analysis = None
    run_multi_feature_steering_analysis = None
    INTERVENTION_SERVICE_AVAILABLE = False
    print("WARNING: intervention service not found, steering analysis will not be available")

# import interaction service
try:
    from interaction import analyze_node_interaction_impl
    INTERACTION_SERVICE_AVAILABLE = True
except ImportError:
    analyze_node_interaction_impl = None
    INTERACTION_SERVICE_AVAILABLE = False
    print("WARNING: interaction service not found, node interaction analysis will not be available")

# import self-play service
try:
    from self_play import run_self_play, analyze_game_positions
    SELF_PLAY_SERVICE_AVAILABLE = True
except ImportError:
    run_self_play = None
    analyze_game_positions = None
    SELF_PLAY_SERVICE_AVAILABLE = False
    print("WARNING: self-play service not found, self-play functionality will not be available")

# import Logit Lens service
try:
    from logit_lens import IntegratedPolicyLens
    LOGIT_LENS_AVAILABLE = True
except ImportError:
    IntegratedPolicyLens = None
    LOGIT_LENS_AVAILABLE = False
    print("WARNING: logit_lens not found, logit lens functionality will not be available")

# global Logit Lens cache
_logit_lens_instances = {}

@app.post("/sae/preload_combo")
def preload_sae_combo(request: dict):
    global CURRENT_BT4_SAE_COMBO_ID, _loading_locks, _loading_status, _loading_logs, _cancel_loading
    global _transcoders_cache, _lorsas_cache, _replacement_models_cache, _global_loading_lock

    model_name = request.get("model_name", "lc0/BT4-1024x15x32h")
    
    import urllib.parse
    
    decoded_model_name = urllib.parse.unquote(model_name)
    if "%" in decoded_model_name:
        decoded_model_name = urllib.parse.unquote(decoded_model_name)
    
    requested_combo_id = request.get("sae_combo_id") or CURRENT_BT4_SAE_COMBO_ID

    # normalize combo configuration (if unknown ID is passed, fallback to default combo)
    combo_cfg = get_bt4_sae_combo(requested_combo_id)
    combo_id = combo_cfg["id"]
    # use decoded model_name to generate cache key
    combo_key = _make_combo_cache_key(decoded_model_name, combo_id)
    
    # if switch combo, first interrupt other combos that are currently loading
    if combo_id != CURRENT_BT4_SAE_COMBO_ID:
        # interrupt loading of all other combos
        for other_combo_key in list(_cancel_loading.keys()):
            if other_combo_key != combo_key:
                _cancel_loading[other_combo_key] = True
                print(f"mark interrupt loading: {other_combo_key}")
                # if this combo is currently loading, also record in logs
                if other_combo_key in _loading_logs:
                    _loading_logs[other_combo_key].append({
                        "timestamp": time.time(),
                        "message": f"loading interrupted (switch to new combo {combo_id})",
                    })

    try:
        if not SAE_PRELOAD_SERVICE_AVAILABLE or load_model_and_transcoders is None:
            raise HTTPException(status_code=503, detail="SAE preload service not available")

        # if switch combo, clear previous combo's SAE cache and try to release memory
        if combo_id != CURRENT_BT4_SAE_COMBO_ID:
            print(f"chess SAE combo switch: {CURRENT_BT4_SAE_COMBO_ID} -> {combo_id}, start clearing old cache")

            # clear all SAE caches, keeping the shared HookedTransformer model cache
            for cache_name, cache in [
                ("_transcoders_cache", _transcoders_cache),
                ("_lorsas_cache", _lorsas_cache),
                ("_replacement_models_cache", _replacement_models_cache),
            ]:
                try:
                    for cache_key, v in list(cache.items()):
                        # try to move SAE to CPU, then delete reference
                        if isinstance(v, dict):
                            for sae in v.values():
                                try:
                                    if hasattr(sae, "to"):
                                        sae.to("cpu")
                                except Exception:
                                    continue
                        elif isinstance(v, list):
                            for sae in v:
                                try:
                                    if hasattr(sae, "to"):
                                        sae.to("cpu")
                                except Exception:
                                    continue
                        del cache[cache_key]
                    print(f"   - cleared cache {cache_name}")
                except Exception as clear_err:
                    print(f"   error clearing cache {cache_name}: {clear_err}")
            
            # also clear the shared SAE preload cache
            if SAE_PRELOAD_SERVICE_AVAILABLE:
                try:
                    for cache_key in list(_global_transcoders_cache.keys()):
                        if cache_key != decoded_model_name:  # keep HookedTransformer's cache key (only model_name)
                            del _global_transcoders_cache[cache_key]
                    for cache_key in list(_global_lorsas_cache.keys()):
                        if cache_key != decoded_model_name:
                            del _global_lorsas_cache[cache_key]
                    for cache_key in list(_global_replacement_models_cache.keys()):
                        if cache_key != decoded_model_name:
                            del _global_replacement_models_cache[cache_key]
                    print("   - cleared shared SAE preload cache")
                except Exception as clear_err:
                    print(f"   error clearing shared SAE preload cache: {clear_err}")

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("   - called torch.cuda.empty_cache() to release memory")
            except Exception as e:
                print(f"   error calling empty_cache: {e}")

            # clear old patching analyzer
            try:
                from intervention import clear_patching_analyzer
                clear_patching_analyzer(CURRENT_BT4_SAE_COMBO_ID)
                print("   - cleared old patching analyzer")
            except (ImportError, Exception) as e:
                print(f"   error clearing patching analyzer: {e}")

            CURRENT_BT4_SAE_COMBO_ID = combo_id

        # create/get loading lock for current combo
        if combo_key not in _loading_locks:
            _loading_locks[combo_key] = threading.Lock()

        # check if already preloaded
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(decoded_model_name, combo_id)
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                print(f"✅ transcoders and lorsas already preloaded: {decoded_model_name} @ {combo_id}")
                return {
                    "status": "already_loaded",
                    "message": f"transcoders and lorsas of model {decoded_model_name} combo {combo_id} already preloaded",
                    "model_name": decoded_model_name,
                    "sae_combo_id": combo_id,
                    "n_layers": len(cached_lorsas),
                    "transcoders_count": len(cached_transcoders),
                    "lorsas_count": len(cached_lorsas),
                }

        # use global lock to ensure only one configuration is loaded at a time (avoid GPU memory being occupied by multiple configurations)
        # then use combo lock to avoid concurrent loading of the same combo
        with _global_loading_lock:
            with _loading_locks[combo_key]:
                # check again if already loaded (may have been loaded while waiting for the lock)
                cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(decoded_model_name, combo_id)
                if cached_transcoders is not None and cached_lorsas is not None:
                    if len(cached_transcoders) == 15 and len(cached_lorsas) == 15:
                        print(f"✅ transcoders and lorsas already preloaded (checked inside lock): {decoded_model_name} @ {combo_id}")
                        return {
                            "status": "already_loaded",
                            "message": f"transcoders and lorsas of model {decoded_model_name} combo {combo_id} already preloaded",
                            "model_name": decoded_model_name,
                            "sae_combo_id": combo_id,
                            "n_layers": len(cached_lorsas),
                            "transcoders_count": len(cached_transcoders),
                            "lorsas_count": len(cached_lorsas),
                        }

                # mark as loading, and clear interrupt flag (set inside global lock, ensure other requests can detect it)
                _loading_status[combo_key] = {"is_loading": True}
                _cancel_loading[combo_key] = False
                print(f"🔍 start preloading transcoders and lorsas: {decoded_model_name} @ {combo_id} (global lock acquired)")

                try:
                    # get HookedTransformer model
                    hooked_model = get_hooked_model(decoded_model_name)

                    # only support BT4
                    if "BT4" not in decoded_model_name:
                        raise HTTPException(status_code=400, detail="Unsupported Model!")

                    tc_base_path = combo_cfg["tc_base_path"]
                    lorsa_base_path = combo_cfg["lorsa_base_path"]
                    n_layers = 15

                    # initialize loading logs
                    if combo_key not in _loading_logs:
                        _loading_logs[combo_key] = []
                    loading_logs = _loading_logs[combo_key]
                    loading_logs.clear()
                    # add initial log
                    loading_logs.append({
                        "timestamp": time.time(),
                        "message": f"🔍 start preloading transcoders and lorsas: {decoded_model_name} @ {combo_id}",
                    })
                    print(f"📝 initialize loading logs list: combo_key={combo_key}, list ID={id(loading_logs)}")

                    # load transcoders and lorsas
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    # create cancel flag dictionary (passed by reference, can be checked in loop)
                    # use a wrapper function to check cancel flag periodically
                    def check_cancel():
                        return _cancel_loading.get(combo_key, False)
                    
                    cancel_flag = {"should_cancel": False, "combo_key": combo_key, "check_fn": check_cancel}
                    replacement_model, transcoders, lorsas = load_model_and_transcoders(
                        model_name=decoded_model_name,
                        device=device,
                        tc_base_path=tc_base_path,
                        lorsa_base_path=lorsa_base_path,
                        n_layers=n_layers,
                        hooked_model=hooked_model,
                        loading_logs=loading_logs,
                        cancel_flag=cancel_flag,
                        cache_key=combo_key,  # pass cache_key to distinguish different combos
                    )

                    print(f"📝 number of logs after loading: {len(loading_logs)}")

                    # cache transcoders and lorsas (update shared cache and local cache)
                    _transcoders_cache[combo_key] = transcoders
                    _lorsas_cache[combo_key] = lorsas
                    _replacement_models_cache[combo_key] = replacement_model

                    # if the preload service is available, also update the shared cache (use combo_key as cache key)
                    if SAE_PRELOAD_SERVICE_AVAILABLE and set_cached_models is not None:
                        set_cached_models(combo_key, hooked_model, transcoders, lorsas, replacement_model)

                    print(f"✅ preloading completed: {model_name} @ {combo_id}")
                    print(f"   - transcoders: {len(transcoders)} layers")
                    print(f"   - lorsas: {len(lorsas)} layers")

                    # add completion log
                    if combo_key in _loading_logs:
                        _loading_logs[combo_key].append(
                            {
                                "timestamp": time.time(),
                                "message": f"✅ preloading completed: {model_name} @ {combo_id}",
                            }
                        )
                        _loading_logs[combo_key].append(
                            {
                                "timestamp": time.time(),
                                "message": f"   - transcoders: {len(transcoders)} layers",
                            }
                        )
                        _loading_logs[combo_key].append(
                            {
                                "timestamp": time.time(),
                                "message": f"   - lorsas: {len(lorsas)} layers",
                            }
                        )

                    _loading_status[combo_key] = {"is_loading": False}

                    return {
                        "status": "loaded",
                        "message": f"successfully preloaded transcoders and lorsas of model {decoded_model_name} combo {combo_id}",
                        "model_name": decoded_model_name,
                        "sae_combo_id": combo_id,
                        "n_layers": n_layers,
                        "transcoders_count": len(transcoders),
                        "lorsas_count": len(lorsas),
                        "device": device,
                    }
                except InterruptedError as e:
                    # loading interrupted, clear partially loaded cache
                    _loading_status[combo_key] = {"is_loading": False}
                    _cancel_loading[combo_key] = False
                    # clear cache of this combo
                    if combo_key in _transcoders_cache:
                        del _transcoders_cache[combo_key]
                    if combo_key in _lorsas_cache:
                        del _lorsas_cache[combo_key]
                    if combo_key in _replacement_models_cache:
                        del _replacement_models_cache[combo_key]
                    if combo_key in _loading_logs:
                        _loading_logs[combo_key].append({
                            "timestamp": time.time(),
                            "message": f"🛑 loading interrupted and cleared cache: {str(e)}",
                        })
                    print(f"🛑 loading interrupted and cleared cache: {combo_key}")
                    raise HTTPException(status_code=499, detail=f"loading interrupted: {str(e)}")
                except Exception:
                    _loading_status[combo_key] = {"is_loading": False}
                    raise

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        if combo_key in _loading_logs:
            _loading_logs[combo_key].append(
                {
                    "timestamp": time.time(),
                    "message": f"❌ preloading failed: {str(e)}",
                }
            )
        if combo_key in _loading_status:
            _loading_status[combo_key] = {"is_loading": False}
        raise HTTPException(status_code=500, detail=f"preloading failed: {str(e)}")


@app.post("/sae/cancel_preload")
def cancel_sae_preload(request: dict):
    """
    interrupt ongoing model loading
    
    Args:
        request: request body containing model information
            - model_name: model name (optional, default: "lc0/BT4-1024x15x32h")
            - sae_combo_id: SAE combo ID (optional, if not provided, interrupt all ongoing combos)
    
    Returns:
        interrupt result
    """
    global _cancel_loading, _loading_status, _loading_logs
    global _transcoders_cache, _lorsas_cache, _replacement_models_cache
    
    model_name = request.get("model_name", "lc0/BT4-1024x15x32h")
    requested_combo_id = request.get("sae_combo_id")
    
    if requested_combo_id:
        # interrupt specified combo
        combo_cfg = get_bt4_sae_combo(requested_combo_id)
        combo_id = combo_cfg["id"]
        combo_key = _make_combo_cache_key(model_name, combo_id)
        
        if combo_key in _loading_status and _loading_status[combo_key].get("is_loading", False):
            _cancel_loading[combo_key] = True
            print(f"🛑 mark interrupt loading: {combo_key}")
            return {
                "status": "cancelled",
                "message": f"mark interrupt combo {combo_id} loading",
                "model_name": model_name,
                "sae_combo_id": combo_id,
            }
        else:
            return {
                "status": "not_loading",
                "message": f"combo {combo_id} is not currently loading",
                "model_name": model_name,
                "sae_combo_id": combo_id,
            }
    else:
        # interrupt all ongoing combos
        cancelled_keys = []
        for combo_key, status in _loading_status.items():
            if status.get("is_loading", False):
                _cancel_loading[combo_key] = True
                cancelled_keys.append(combo_key)
                print(f"🛑 mark interrupt loading: {combo_key}")
        
        return {
            "status": "cancelled" if cancelled_keys else "no_loading",
            "message": f"mark interrupt {len(cancelled_keys)} combos loading" if cancelled_keys else "no combos are currently loading",
            "cancelled_keys": cancelled_keys,
        }


@app.get("/sae/preload_logs")
def get_sae_preload_logs(
    model_name: str = "lc0/BT4-1024x15x32h",
    sae_combo_id: str | None = None,
):
    """
    get model loading logs
    
    Args:
        model_name: model name (query parameter, default: "lc0/BT4-1024x15x32h")
        sae_combo_id: SAE combo ID (query parameter, optional)
    
    Returns:
        loading logs list
    """

    global _loading_logs, _loading_status

    # URL decode, handle possible double encoding problem
    import urllib.parse

    decoded_model_name = urllib.parse.unquote(model_name)
    if "%" in decoded_model_name:
        decoded_model_name = urllib.parse.unquote(decoded_model_name)

    combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
    combo_cfg = get_bt4_sae_combo(combo_id)
    normalized_combo_id = combo_cfg["id"]
    combo_key = _make_combo_cache_key(decoded_model_name, normalized_combo_id)

    logs = _loading_logs.get(combo_key, [])
    is_loading = _loading_status.get(combo_key, {}).get("is_loading", False)

    return {
        "model_name": decoded_model_name,
        "sae_combo_id": normalized_combo_id,
        "logs": logs,
        "total_count": len(logs),
        "is_loading": is_loading,
    }



@app.post("/patching_analysis")
def patching_analysis(request: dict):
    """
    run patching analysis and return Token Predictions result
    
    Args:
        request: request body containing analysis parameters
            - fen: FEN string (required)
            - feature_type: feature type ('transcoder' or 'lorsa') (required)
            - layer: layer number (required)
            - pos: position (required)
            - feature: feature index (required)
    
    Returns:
        Token Predictions analysis result (JSON format)
    """
    try:
        # check if patching service is available
        if not PATCHING_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Patching service not available")
        
        # extract parameters
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        feature_type = request.get("feature_type")
        if feature_type not in ['transcoder', 'lorsa']:
            raise HTTPException(status_code=400, detail="feature_type must be 'transcoder' or 'lorsa'")
        
        layer = request.get("layer")
        if layer is None or not isinstance(layer, int):
            raise HTTPException(status_code=400, detail="layer must be an integer")
        
        pos = request.get("pos")
        if pos is None or not isinstance(pos, int):
            raise HTTPException(status_code=400, detail="pos must be an integer")
        
        feature = request.get("feature")
        if feature is None or not isinstance(feature, int):
            raise HTTPException(status_code=400, detail="feature must be an integer")
        
        print(f"🔍 Running patching analysis: {feature_type} L{layer} pos{pos} feature{feature}")
        
        # run patching analysis
        result = run_patching_analysis(
            fen=fen,
            feature_type=feature_type,
            layer=layer,
            pos=pos,
            feature=feature
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        print(f"✅ Patching analysis completed, found {result['statistics']['total_legal_moves']} legal moves")
        
        return result
        
    except Exception as e:
        print(f"❌ Patching analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Patching analysis failed: {str(e)}")


@app.get("/patching_analysis/status")
def patching_analysis_status():
    """check patching analysis service status"""
    return {
        "available": PATCHING_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/steering_analysis")
def steering_analysis(request: dict):
    """
    run steering analysis and return Token Predictions result, support adjustable steering_scale
    
    Args:
        request: request body containing analysis parameters
            - fen: FEN string (required)
            - feature_type: feature type ('transcoder' or 'lorsa') (required)
            - layer: layer number (required)
            - pos: position (required)
            - feature: feature index (required)
            - steering_scale: scaling factor (optional, default 1)
    
    Returns:
        Token Predictions analysis result (JSON format)
    """
    try:
        if not INTERVENTION_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Steering service not available")

        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")

        feature_type = request.get("feature_type")
        if feature_type not in ['transcoder', 'lorsa']:
            raise HTTPException(status_code=400, detail="feature_type must be 'transcoder' or 'lorsa'")

        layer = request.get("layer")
        if layer is None or not isinstance(layer, int):
            raise HTTPException(status_code=400, detail="layer must be an integer")

        pos = request.get("pos")
        if pos is None or not isinstance(pos, int):
            raise HTTPException(status_code=400, detail="pos must be an integer")

        feature = request.get("feature")
        if feature is None or not isinstance(feature, int):
            raise HTTPException(status_code=400, detail="feature must be an integer")

        steering_scale = request.get("steering_scale", 1)
        if not isinstance(steering_scale, (int, float)):
            raise HTTPException(status_code=400, detail="steering_scale must be a number")

        # get metadata from request
        metadata = request.get("metadata", {})

        print(f"🔍 run steering analysis: {feature_type} L{layer} pos{pos} feature{feature} scale{steering_scale}")
        print(f"📋 Metadata: {metadata}")

        result = run_feature_steering_analysis(
            fen=fen,
            feature_type=feature_type,
            layer=layer,
            pos=pos,
            feature=feature,
            steering_scale=steering_scale,
            metadata=metadata
        )

        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])

        print(f"✅ Steering analysis completed, found {result['statistics']['total_legal_moves']} legal moves")
        return result

    except Exception as e:
        print(f"❌ Steering analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Steering analysis failed: {str(e)}")


@app.post("/steering_analysis/multi")
def steering_analysis_multi(request: dict):
    """
    run multi feature steering analysis (each feature corresponds to one position) and return the result.

    Args:
        request:
            - fen: FEN string (required)
            - feature_type: 'transcoder' or 'lorsa' (required)
            - layer: int (required)
            - nodes: list[dict] (required), each node must contain:
                - pos: int
                - feature: int
                - steering_scale: float | int (optional, default 1)
            - metadata: dict (optional)

    Returns:
        the same structure as /steering_analysis, but ablation_info.nodes will contain information for each node.
    """
    try:
        if not INTERVENTION_SERVICE_AVAILABLE or run_multi_feature_steering_analysis is None:
            raise HTTPException(status_code=503, detail="Steering service not available")

        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")

        feature_type = request.get("feature_type")
        if feature_type not in ["transcoder", "lorsa"]:
            raise HTTPException(status_code=400, detail="feature_type must be 'transcoder' or 'lorsa'")

        layer = request.get("layer")
        if layer is None or not isinstance(layer, int):
            raise HTTPException(status_code=400, detail="layer must be an integer")

        nodes = request.get("nodes")
        if not isinstance(nodes, list) or len(nodes) == 0:
            raise HTTPException(status_code=400, detail="nodes must be a non-empty list")

        metadata = request.get("metadata", {})

        print(f"🔍 run multi steering analysis: {feature_type} L{layer}, nodes={len(nodes)}")
        result = run_multi_feature_steering_analysis(
            fen=fen,
            feature_type=feature_type,
            layer=layer,
            nodes=nodes,
            metadata=metadata,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Multi steering analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi steering analysis failed: {str(e)}")


@app.get("/steering_analysis/status")
def steering_analysis_status():
    """check steering analysis service status"""
    return {
        "available": INTERVENTION_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/self_play")
def start_self_play(request: dict):
    """
    start self-play and return game data
    
    Args:
        request: request body containing game parameters
            - initial_fen: initial FEN string (optional, default starting position)
            - max_moves: maximum move number (optional, default 10)
            - temperature: temperature parameter (optional, default 1.0)
    
    Returns:
        self-play game data (JSON format)
    """
    try:
        # check if self-play service is available
        if not SELF_PLAY_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Self-play service not available")
        
        # extract parameters
        initial_fen = request.get("initial_fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        max_moves = request.get("max_moves", 10)
        temperature = request.get("temperature", 1.0)
        
        # validate parameters
        if not isinstance(max_moves, int) or max_moves <= 0:
            raise HTTPException(status_code=400, detail="max_moves must be a positive integer")
        
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise HTTPException(status_code=400, detail="temperature must be a non-negative number")
        
        print(f"🎮 start self-play: {initial_fen[:50]}..., maximum moves: {max_moves}, temperature: {temperature}")
        
        # force using BT4 model
        model_name = "lc0/BT4-1024x15x32h"
        hooked_model = get_hooked_model(model_name)
        
        # run self-play
        game_result = run_self_play(
            initial_fen=initial_fen,
            max_moves=max_moves,
            temperature=temperature,
            model=hooked_model
        )
        
        print(f"✅ self-play completed, total {len(game_result['moves'])} moves")
        
        return game_result
        
    except Exception as e:
        print(f"❌ self-play failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-play failed: {str(e)}")


@app.post("/self_play/analyze")
def analyze_self_play_positions(request: dict):
    """
    analyze position sequence in self-play
    
    Args:
        request: request body containing position sequence
            - positions: list of FEN strings
    
    Returns:
        position analysis result (JSON format)
    """
    try:
        # check if self-play service is available
        if not SELF_PLAY_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Self-play service not available")
        
        # extract parameters
        positions = request.get("positions", [])
        
        if not isinstance(positions, list) or not positions:
            raise HTTPException(status_code=400, detail="positions must be a non-empty list of FEN strings")
        
        print(f"🔍 analyze position sequence, total {len(positions)} positions")
        
        # get cached HookedTransformer model
        hooked_model = get_hooked_model()
        
        # analyze position sequence
        analysis_result = analyze_game_positions(
            positions=positions,
            model=hooked_model
        )
        
        print(f"✅ position analysis completed")
        
        return {
            "positions_analysis": analysis_result,
            "total_positions": len(positions)
        }
        
    except Exception as e:
        print(f"❌ position analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Position analysis failed: {str(e)}")


@app.get("/self_play/status")
def self_play_status():
    """check self-play service status"""
    return {
        "available": SELF_PLAY_SERVICE_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/logit_lens/analyze")
def logit_lens_analyze(request: dict):
    """
    run Logit Lens analysis
    
    Args:
        request: request body containing analysis parameters
            - fen: FEN string (required)
            - target_move: target move UCI (optional)
            - topk_vocab: top k vocabulary (optional, default: 2000)
    
    Returns:
        Logit Lens analysis result (JSON format)
    """
    try:
        # check if Logit Lens service is available
        if not LOGIT_LENS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Logit Lens service not available")
        
        # extract parameters
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        # force using BT4 model
        model_name = "lc0/BT4-1024x15x32h"
        target_move = request.get("target_move")
        topk_vocab = request.get("topk_vocab", 2000)
        
        print(f"🔍 run Logit Lens analysis: FEN={fen[:50]}..., model={model_name}, target={target_move}")
        
        # get or create Logit Lens instance
        global _logit_lens_instances
        if model_name not in _logit_lens_instances:
            # get model
            hooked_model = get_hooked_model(model_name)
            # create Logit Lens instance
            _logit_lens_instances[model_name] = IntegratedPolicyLens(hooked_model)
        
        lens = _logit_lens_instances[model_name]
        
        # run analysis
        result = lens.analyze_single_fen(
            fen=fen,
            target_move=target_move,
            topk_vocab=topk_vocab
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        print(f"✅ Logit Lens analysis completed, analyzed {result['num_layers']} layers")
        
        return {
            **result,
            "model_used": model_name
        }
        
    except Exception as e:
        print(f"❌ Logit Lens analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Logit Lens analysis failed: {str(e)}")


@app.get("/logit_lens/status")
def logit_lens_status():
    """check Logit Lens service status"""
    return {
        "available": LOGIT_LENS_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


@app.post("/logit_lens/mean_ablation")
def logit_lens_mean_ablation(request: dict):
    """
    run Mean Ablation analysis
    
    Args:
        request: request body containing analysis parameters
            - fen: FEN string (required)
            - hook_types: hook types list (optional, default: ['attn_out', 'mlp_out'])
            - target_move: target move UCI (optional)
            - topk_vocab: top k vocabulary (optional, default: 2000)
    
    Returns:
        Mean Ablation analysis result (JSON format)
    """
    try:
        # check if Logit Lens service is available
        if not LOGIT_LENS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Logit Lens service not available")
        
        # extract parameters
        fen = request.get("fen")
        if not fen:
            raise HTTPException(status_code=400, detail="FEN string is required")
        
        # force using BT4 model
        model_name = "lc0/BT4-1024x15x32h"
        hook_types = request.get("hook_types", ['attn_out', 'mlp_out'])
        target_move = request.get("target_move")
        topk_vocab = request.get("topk_vocab", 2000)
        
        print(f"🔍 run Mean Ablation analysis: FEN={fen[:50]}..., model={model_name}, hooks={hook_types}, target={target_move}")
        
        # get or create Logit Lens instance
        global _logit_lens_instances
        if model_name not in _logit_lens_instances:
            # get model
            hooked_model = get_hooked_model(model_name)
            # create Logit Lens instance
            _logit_lens_instances[model_name] = IntegratedPolicyLens(hooked_model)
        
        lens = _logit_lens_instances[model_name]
        
        # run Mean Ablation analysis
        result = lens.analyze_mean_ablation(
            fen=fen,
            hook_types=hook_types,
            target_move=target_move,
            topk_vocab=topk_vocab
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        print(f"✅ Mean Ablation analysis completed, analyzed {result['num_layers']} layers, {len(result['hook_types'])} hook types")
        
        return {
            **result,
            "model_used": model_name
        }
        
    except Exception as e:
        print(f"❌ Mean Ablation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mean Ablation analysis failed: {str(e)}")


# new: move evaluation interface (based on Stockfish)
@app.post("/evaluate_move")
def evaluate_move(request: dict):
    """
    Evaluate a single move: given the previous-position FEN and the move UCI,
    return a 0-100 score, centipawn difference, WDL, etc.

    Body: { "fen": str, "move": str, "time_limit": float? }
    """
    fen = request.get("fen")
    move = request.get("move")
    time_limit = request.get("time_limit", 0.2)
    if not fen or not move:
        raise HTTPException(status_code=400, detail="Both 'fen' and 'move' are required")
    try:
        _ = chess.Board(fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN")

    res = evaluate_move_quality(fen, move, time_limit=time_limit)
    if res is None:
        raise HTTPException(status_code=400, detail="Evaluation failed or move is illegal")
    return res


# tactic features analysis interface
@app.post("/tactic_features/analyze")
async def analyze_tactic_features_api(
    file: UploadFile = File(...),
    n_random: int = Form(200),
    n_fens: int = Form(200),
    top_k_lorsa: int = Form(10),
    top_k_tc: int = Form(10),
    specific_layer: Optional[str] = Form(None),
    specific_layer_top_k: int = Form(20),
):
    """
    analyze tactic features: upload FEN file, compare with random FENs, find the most relevant features
    
    Args:
        file: uploaded txt file, each line is a FEN
        model_name: model name
        n_random: random FEN number (compatible with old parameters)
        n_fens: FEN number (new parameter, use priority)
        top_k_lorsa: display top k Lorsa features
        top_k_tc: display top k TC features
        specific_layer: specified layer number (optional), if provided, return the detailed features of the layer
        specific_layer_top_k: top k features of the specified layer
    """
    if not TACTIC_FEATURES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tactic features analysis not available")
    
    if not HOOKED_TRANSFORMER_AVAILABLE:
        raise HTTPException(status_code=503, detail="HookedTransformer is not available")
    
    try:
        # force using BT4 model
        model_name = "lc0/BT4-1024x15x32h"
        
        # ========== debug information: function started ==========
        print("=" * 80)
        print("🚀 start processing tactic features analysis request")
        print(f"📥 received original parameters:")
        print(f"   - model_name: {model_name} (force using BT4)")
        print(f"   - n_random: {n_random}")
        print(f"   - n_fens: {n_fens}")
        print(f"   - top_k_lorsa: {top_k_lorsa}")
        print(f"   - top_k_tc: {top_k_tc}")
        print(f"   - specific_layer (original): {specific_layer} (type: {type(specific_layer)})")
        print(f"   - specific_layer_top_k: {specific_layer_top_k}")
        print("=" * 80)
        
        # parse specific_layer parameter
        parsed_specific_layer = None
        print(f"🔍 start parsing specific_layer parameter...")
        print(f"   - specific_layer is None: {specific_layer is None}")
        if specific_layer is not None:
            print(f"   - specific_layer value: '{specific_layer}'")
            print(f"   - specific_layer.strip() after: '{specific_layer.strip() if isinstance(specific_layer, str) else specific_layer}'")
        
        if specific_layer is not None and isinstance(specific_layer, str) and specific_layer.strip():
            try:
                parsed_specific_layer = int(specific_layer.strip())
                print(f"✅ successfully parsed specified layer parameter: {parsed_specific_layer} (original value: '{specific_layer}')")
            except (ValueError, TypeError) as e:
                print(f"❌ failed to parse layer number parameter: {e}")
                print(f"⚠️ invalid layer number parameter: '{specific_layer}', will ignore specified layer analysis")
                parsed_specific_layer = None
        elif specific_layer is None:
            print(f"ℹ️ specific_layer parameter is not provided, will not perform specified layer analysis")
        else:
            print(f"⚠️ specific_layer parameter is empty string or invalid, will ignore")
        
        # use n_fens if provided, otherwise n_random
        actual_n_fens = n_fens if n_fens != 200 or n_random == 200 else n_random
        print(f"📊 actual used FEN number: {actual_n_fens}")
        
        print("final parsed result:")
        print(f"   - parsed_specific_layer: {parsed_specific_layer}")
        print(f"   - specific_layer_top_k: {specific_layer_top_k}")
        print(f"   - actual_n_fens: {actual_n_fens}")
        if parsed_specific_layer is not None:
            print(f"✅ will analyze specified layer: Layer {parsed_specific_layer}")
        else:
            print(f"ℹ️ will not perform specified layer analysis")
        print("=" * 80)
        
        # read file content
        contents = await file.read()
        text = contents.decode('utf-8')
        tactic_fens = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        if not tactic_fens:
            raise HTTPException(status_code=400, detail="file is empty or no valid FEN lines")
        
        # validate FEN format
        valid_fens, invalid_fens = validate_fens(tactic_fens)
        
        # limit FEN number: if the number of FENs in the file is greater than the set number, take the first n; otherwise, use all
        if len(valid_fens) > actual_n_fens:
            print(f"📊 file has {len(valid_fens)} valid FENs, take the first {actual_n_fens}")
            valid_fens = valid_fens[:actual_n_fens]
        else:
            print(f"📊 file has {len(valid_fens)} valid FENs, use all")
        
        if len(valid_fens) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"no valid FEN strings. invalid FEN examples: {invalid_fens[:5]}"
            )
        
        # load model (using cache)
        hooked_model = get_hooked_model(model_name)
        
        # check cached transcoders and lorsas
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(model_name)
        
        num_layers = 15
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == num_layers and len(cached_lorsas) == num_layers:
                print(f"✅ using cached transcoders and lorsas: {model_name}")
                transcoders = cached_transcoders
                lorsas = cached_lorsas
            else:
                # cache is incomplete, need to load
                print(f"⚠️ cache is incomplete, reload: {model_name}")
                transcoders = None
                lorsas = None
        else:
            transcoders = None
            lorsas = None
        
        # if cache is not available, load
        if transcoders is None or lorsas is None:
            if 'BT4' in model_name:
                tc_base_path = BT4_TC_BASE_PATH
                lorsa_base_path = BT4_LORSA_BASE_PATH
            else:
                raise ValueError("Unsupported Model!")
            
            transcoders = {}
            lorsas = []
            
            for layer in range(num_layers):
                # load Transcoder
                tc_path = f"{tc_base_path}/L{layer}"
                if os.path.exists(tc_path):
                    transcoders[layer] = SparseAutoEncoder.from_pretrained(
                        tc_path,
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Transcoder not found at {tc_path}"
                    )
                
                # load Lorsa
                lorsa_path = f"{lorsa_base_path}/L{layer}"
                if os.path.exists(lorsa_path):
                    lorsas.append(LowRankSparseAttention.from_pretrained(
                        lorsa_path,
                        device=device,
                    ))
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Lorsa not found at {lorsa_path}"
                    )
            
            # cache loaded transcoders and lorsas
            if SAE_PRELOAD_SERVICE_AVAILABLE and set_cached_models is not None:
                # need to create replacement_model to cache, here cache transcoders and lorsas first
                _global_transcoders_cache[model_name] = transcoders
                _global_lorsas_cache[model_name] = lorsas
                _global_hooked_models[model_name] = hooked_model
        
        # execute analysis
        print("=" * 80)
        print(f"🔬 start executing feature analysis")
        print(f"   - tactic FEN number: {len(valid_fens)}")
        print(f"   - random FEN number: {actual_n_fens}")
        print(f"   - model layers: {num_layers} layers (0-{num_layers-1})")
        if parsed_specific_layer is not None:
            print(f"   ✅ specified layer analysis is enabled:")
            print(f"      - layer number: Layer {parsed_specific_layer}")
            print(f"      - Top K: {specific_layer_top_k}")
            if parsed_specific_layer < 0 or parsed_specific_layer >= num_layers:
                print(f"      ⚠️ warning: layer number {parsed_specific_layer} is out of valid range!")
        else:
            print(f"   ℹ️ layer number is not specified, will only return top K features of all layers")
        print("=" * 80)
        
        result = analyze_tactic_features(
            tactic_fens=valid_fens,
            model=hooked_model,
            lorsas=lorsas,
            transcoders=transcoders,
            n_random=actual_n_fens,
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # sort and take top k
        lorsa_diffs = sorted(result["lorsa_diffs"], key=lambda x: x[2], reverse=True)[:top_k_lorsa]
        tc_diffs = sorted(result["tc_diffs"], key=lambda x: x[2], reverse=True)[:top_k_tc]
        
        # format result
        def format_diff(diff_tuple):
            layer, feature, diff, p_random, p_tactic, kind = diff_tuple
            return {
                "layer": layer,
                "feature": feature,
                "diff": float(diff),
                "p_random": float(p_random),
                "p_tactic": float(p_tactic),
                "kind": kind
            }
        
        response_data = {
            "valid_tactic_fens": result["valid_tactic_fens"],
            "invalid_tactic_fens": result["invalid_tactic_fens"],
            "random_fens": result["random_fens"],
            "tactic_fens": result["tactic_fens"],
            "top_lorsa_features": [format_diff(d) for d in lorsa_diffs],
            "top_tc_features": [format_diff(d) for d in tc_diffs],
            "invalid_fens_sample": result.get("invalid_fens_list", [])
        }
        
        # if layer number is specified, return the detailed features of the layer
        print("=" * 80)
        print(f"🔍 check if need to return specified layer features...")
        print(f"   - parsed_specific_layer: {parsed_specific_layer}")
        print(f"   - num_layers: {num_layers}")
        print(f"   - condition check: parsed_specific_layer is not None = {parsed_specific_layer is not None}")
        if parsed_specific_layer is not None:
            print(f"   - condition check: 0 <= {parsed_specific_layer} < {num_layers} = {0 <= parsed_specific_layer < num_layers}")
        
        if parsed_specific_layer is not None and 0 <= parsed_specific_layer < num_layers:
            print(f"✅ start filtering features of Layer {parsed_specific_layer}...")
            
            # print total number of all features (for debugging)
            total_lorsa_diffs = len(result["lorsa_diffs"])
            total_tc_diffs = len(result["tc_diffs"])
            print(f"   - total Lorsa features: {total_lorsa_diffs}")
            print(f"   - total TC features: {total_tc_diffs}")
            
            # filter out features of the specified layer
            specific_lorsa = [d for d in result["lorsa_diffs"] if d[0] == parsed_specific_layer]
            specific_tc = [d for d in result["tc_diffs"] if d[0] == parsed_specific_layer]
            
            print(f"📊 Layer {parsed_specific_layer} features statistics:")
            print(f"   - Lorsa features: {len(specific_lorsa)}")
            print(f"   - TC features: {len(specific_tc)}")
            
            if len(specific_lorsa) == 0:
                print(f"   ⚠️ warning: Layer {parsed_specific_layer} no Lorsa features found!")
            if len(specific_tc) == 0:
                print(f"   ⚠️ warning: Layer {parsed_specific_layer} no TC features found!")
            
            # sort and take top k
            specific_lorsa_sorted = sorted(specific_lorsa, key=lambda x: x[2], reverse=True)[:specific_layer_top_k]
            specific_tc_sorted = sorted(specific_tc, key=lambda x: x[2], reverse=True)[:specific_layer_top_k]
            
            print(f"   - sorted and take top {specific_layer_top_k}:")
            print(f"     * Lorsa: {len(specific_lorsa_sorted)}")
            print(f"     * TC: {len(specific_tc_sorted)}")
            
            # print detailed information of the first 3 features (for debugging)
            if len(specific_lorsa_sorted) > 0:
                print(f"   - Lorsa Top 3 features example:")
                for i, feat in enumerate(specific_lorsa_sorted[:3]):
                    print(f"     [{i+1}] Layer={feat[0]}, Feature={feat[1]}, Diff={feat[2]:.6f}")
            
            if len(specific_tc_sorted) > 0:
                print(f"   - TC Top 3 features example:")
                for i, feat in enumerate(specific_tc_sorted[:3]):
                    print(f"     [{i+1}] Layer={feat[0]}, Feature={feat[1]}, Diff={feat[2]:.6f}")
            
            response_data["specific_layer"] = parsed_specific_layer
            response_data["specific_layer_lorsa"] = [format_diff(d) for d in specific_lorsa_sorted]
            response_data["specific_layer_tc"] = [format_diff(d) for d in specific_tc_sorted]
            
            print(f"✅ added specified layer features to response data:")
            print(f"   - specific_layer: {response_data.get('specific_layer')}")
            print(f"   - specific_layer_lorsa: {len(response_data.get('specific_layer_lorsa', []))}")
            print(f"   - specific_layer_tc: {len(response_data.get('specific_layer_tc', []))}")
        elif parsed_specific_layer is not None:
            print(f"❌ specified layer number {parsed_specific_layer} is out of valid range (0-{num_layers-1})")
            print(f"    will ignore specified layer analysis")
        else:
            print(f"ℹ️ layer number is not specified, will skip specified layer feature filtering")
        
        print("=" * 80)
        print(f"📤 prepare to return response data:")
        print(f"   - Basic stats: valid_tactic_fens={response_data.get('valid_tactic_fens')}, tactic_fens={response_data.get('tactic_fens')}")
        print(f"   - Top Lorsa features: {len(response_data.get('top_lorsa_features', []))}")
        print(f"   - Top TC features: {len(response_data.get('top_tc_features', []))}")
        print(f"   - specified layer: {response_data.get('specific_layer', 'not specified')}")
        if response_data.get('specific_layer') is not None:
            print(f"   - specified layer Lorsa: {len(response_data.get('specific_layer_lorsa', []))}")
            print(f"   - specified layer TC: {len(response_data.get('specific_layer_tc', []))}")
        print("=" * 80)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"analysis failed: {str(e)}")


@app.get("/tactic_features/status")
def tactic_features_status():
    """check tactic features analysis service status"""
    return {
        "available": TACTIC_FEATURES_AVAILABLE,
        "hooked_transformer_available": HOOKED_TRANSFORMER_AVAILABLE
    }


# Graph Feature Diffing API
@app.get("/virtual_weight")
def get_virtual_weight(
    model_name: str = "lc0/BT4-1024x15x32h",
    sae_combo_id: str | None = None,
    feature_type: str = "tc",  # "tc" or "lorsa"
    layer_idx: int = 0,
    feature_idx: int = 0,
    k: int = 100,
    activation_type: str = "max",  # "max" or "mean"
    features_in_layer_filter: str | None = None,  # layer filter (e.g. "4,5,8-9")
    features_out_layer_filter: str | None = None,  # layer filter (e.g. "4,5,8-9")
):
    """
    get virtual weight of features (input and output)

    Args:
        model_name: model name
        sae_combo_id: SAE combo ID
        feature_type: feature type ("tc" or "lorsa")
        layer_idx: layer index
        feature_idx: feature index
        k: return top k number
        activation_type: activation type ("max" or "mean")
        features_in_layer_filter: input feature layer filter (e.g. "4,5,8-9" means only include layers 4, 5, 8, 9)
        features_out_layer_filter: output feature layer filter (e.g. "4,5,8-9" means only include layers 4, 5, 8, 9)
    Returns:
        dictionary containing input and output virtual weights
    """
    def parse_layer_filter(filter_str: str | None) -> list[int] | None:
        """
        parse layer filter string

        Args:
            filter_str: filter string (e.g. "4,5,8-9")

        Returns:
            layer index list, if None or empty string, return None to indicate no filtering
        """
        if not filter_str or not filter_str.strip():
            return None

        layers = []
        parts = filter_str.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                # handle range (e.g. "8-9")
                try:
                    start, end = map(int, part.split('-'))
                    if start > end:
                        continue
                    layers.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                # handle single number (e.g. "4")
                try:
                    layer = int(part)
                    layers.append(layer)
                except ValueError:
                    continue

        # remove duplicates and sort
        return sorted(list(set(layers)))

    try:
        # URL decode, handle possible encoding issues (consistent with /sae/preload_logs)
        import urllib.parse

        decoded_model_name = urllib.parse.unquote(model_name)
        if "%" in decoded_model_name:
            decoded_model_name = urllib.parse.unquote(decoded_model_name)
        
        # get SAE combo configuration
        combo_id = sae_combo_id or CURRENT_BT4_SAE_COMBO_ID
        combo_cfg = get_bt4_sae_combo(combo_id)
        normalized_combo_id = combo_cfg["id"]
        
        # use get_cached_transcoders_and_lorsas to get cached transcoders and lorsas
        # this function will first check the shared SAE preload cache, then check local cache
        # use decoded model_name
        cached_transcoders, cached_lorsas = get_cached_transcoders_and_lorsas(decoded_model_name, normalized_combo_id)
        
        if cached_transcoders is None or cached_lorsas is None:
            # provide more detailed error information, including requested combo ID and current server's combo ID
            # use decoded model_name to generate cache key
            cache_key = _make_combo_cache_key(decoded_model_name, normalized_combo_id)
            error_detail = (
                f"Transcoders/Lorsas not loaded, please call /sae/preload_combo to preload. "
                f"requested combo ID: {normalized_combo_id}, "
                f"cache key: {cache_key}, "
                f"current server's combo ID: {CURRENT_BT4_SAE_COMBO_ID}"
            )
            print(f"⚠️ /virtual_weight request failed: {error_detail}")
            print(f"   original model_name parameter: {model_name!r}")
            print(f"   decoded model_name: {decoded_model_name!r}")
            # print current cache key list to help debug
            if SAE_PRELOAD_SERVICE_AVAILABLE:
                print(f"   shared SAE preload cache keys: transcoders={list(_global_transcoders_cache.keys())}, lorsas={list(_global_lorsas_cache.keys())}")
                # check if similar cache keys exist (using original or decoded model_name)
                for key in list(_global_transcoders_cache.keys()) + list(_global_lorsas_cache.keys()):
                    if normalized_combo_id in key:
                        print(f"     found similar cache key: {key!r}")
            print(f"   local cache keys: transcoders={list(_transcoders_cache.keys())}, lorsas={list(_lorsas_cache.keys())}")
            # check if similar cache keys exist
            for key in list(_transcoders_cache.keys()) + list(_lorsas_cache.keys()):
                if normalized_combo_id in key:
                    print(f"     found similar cache key: {key!r}")
            raise HTTPException(
                status_code=503,
                detail=error_detail
            )
        
        # validate activation_type parameter
        if activation_type not in ["max", "mean"]:
            raise HTTPException(status_code=400, detail="activation_type must be 'max' or 'mean'")
        
        # load activations data (max or mean)
        tc_acts, lorsa_acts = load_max_activations(
            normalized_combo_id, device=device, get_bt4_sae_combo=get_bt4_sae_combo,
            activation_type=activation_type
        )

        # parse layer filter
        features_in_layer_filter_parsed = parse_layer_filter(features_in_layer_filter)
        features_out_layer_filter_parsed = parse_layer_filter(features_out_layer_filter)
        
        # validate parameters
        if layer_idx < 0 or layer_idx >= len(cached_transcoders):
            raise HTTPException(status_code=400, detail=f"layer_idx must be between 0 and {len(cached_transcoders)-1}")
        
        if feature_type == "tc":
            if feature_idx < 0 or feature_idx >= cached_transcoders[layer_idx].cfg.d_sae:
                raise HTTPException(
                    status_code=400,
                    detail=f"feature_idx must be between 0 and {cached_transcoders[layer_idx].cfg.d_sae-1}"
                )
            
            # compute TC global weight
            features_in = tc_virtual_weight_in(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_acts, lorsa_acts, k=k, layer_filter=features_in_layer_filter_parsed
            )
            features_out = tc_virtual_weight_out(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_acts, lorsa_acts, k=k, layer_filter=features_out_layer_filter_parsed
            )
        elif feature_type == "lorsa":
            if feature_idx < 0 or feature_idx >= cached_lorsas[layer_idx].cfg.d_sae:
                raise HTTPException(
                    status_code=400,
                    detail=f"feature_idx must be between 0 and {cached_lorsas[layer_idx].cfg.d_sae-1}"
                )
            
            # compute Lorsa global weight
            features_in = lorsa_virtual_weight_in(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_acts, lorsa_acts, k=k, layer_filter=features_in_layer_filter_parsed
            )
            features_out = lorsa_virtual_weight_out(
                cached_transcoders, cached_lorsas, layer_idx, feature_idx,
                tc_acts, lorsa_acts, k=k, layer_filter=features_out_layer_filter_parsed
            )
        else:
            raise HTTPException(status_code=400, detail="feature_type must be 'tc' or 'lorsa'")
        
        return {
            "feature_type": feature_type,
            "layer_idx": layer_idx,
            "feature_idx": feature_idx,
            "activation_type": activation_type,
            "feature_name": f"BT4_{feature_type}_L{layer_idx}{'M' if feature_type == 'tc' else 'A'}_k30_e16#{feature_idx}",
            "features_in": [{"name": name, "weight": weight} for name, weight in features_in],
            "features_out": [{"name": name, "weight": weight} for name, weight in features_out],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"compute global weight failed: {str(e)}")


@app.post("/interaction/analyze_node_interaction")
def analyze_node_interaction_api(request: dict):
    """
    Analyze node interaction (supports multiple steering nodes and multiple target nodes)

    Request body:
    {
        "model_name": "lc0/BT4-1024x15x32h",
        "sae_combo_id": "k_128_e_128",
        "fen": "8/p3kpp1/8/3R1r2/8/4P1Q1/PPr4n/6KR b - - 9 32",
        "steering_nodes": [  # can be a single node object or a list of nodes
            {
            "feature_type": "lorsa",
            "layer": 1,
            "feature": 3026,
            "pos": 48
            }
        ],
        "target_nodes": [  # can be a single node object or a list of nodes, all target nodes must be at a higher layer than all steering nodes
            {
            "feature_type": "transcoder",
            "layer": 3,
            "feature": 11305,
            "pos": 34
            }
        ],
        "steering_scale": 2.0
    }

    Returns:
        dictionary containing interaction analysis results:
        {
            "steering_scale": float,
            "steering_nodes_count": int,
            "steering_details": list,
            "target_nodes": [
                {
                    "target_node": str,
                    "original_activation": float,
                    "modified_activation": float,
                    "activation_ratio": float,
                    "activation_change": float
                },
                ...
            ]
        }
    """
    try:
        if analyze_node_interaction_impl is None:
            raise HTTPException(status_code=503, detail="Node interaction service not available")
        return analyze_node_interaction_impl(request)
    except HTTPException:
        # Re-raise HTTPException directly so FastAPI preserves status/detail
        raise
    except ValueError as e:
        # Map validation-style errors to 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Fallback: unexpected errors become 500
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"node interaction analysis failed: {str(e)}")


# add CORS middleware - must be after all route definitions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

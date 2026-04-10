"""
Shared SAE combo preload/cache helpers.

This module keeps the long-lived BT4 model, transcoder, LORSA, and
ReplacementModel caches used by interaction-style endpoints. It deliberately
does not include any circuit-tracing logic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from lm_saes import LowRankSparseAttention, ReplacementModel, SparseAutoEncoder


_global_hooked_models: Dict[str, HookedTransformer] = {}
_global_transcoders_cache: Dict[str, Dict[int, SparseAutoEncoder]] = {}
_global_lorsas_cache: Dict[str, List[LowRankSparseAttention]] = {}
_global_replacement_models_cache: Dict[str, ReplacementModel] = {}

_loading_lock = threading.Lock()
_is_loading: Dict[str, bool] = {}


def get_cached_models(
    cache_key: str,
) -> Tuple[
    Optional[HookedTransformer],
    Optional[Dict[int, SparseAutoEncoder]],
    Optional[List[LowRankSparseAttention]],
    Optional[ReplacementModel],
]:
    model_name = cache_key.split("::")[0] if "::" in cache_key else cache_key
    hooked_model = _global_hooked_models.get(model_name)
    transcoders = _global_transcoders_cache.get(cache_key)
    lorsas = _global_lorsas_cache.get(cache_key)
    replacement_model = _global_replacement_models_cache.get(cache_key)
    return hooked_model, transcoders, lorsas, replacement_model


def set_cached_models(
    cache_key: str,
    hooked_model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    replacement_model: ReplacementModel,
) -> None:
    model_name = cache_key.split("::")[0] if "::" in cache_key else cache_key
    _global_hooked_models[model_name] = hooked_model
    _global_transcoders_cache[cache_key] = transcoders
    _global_lorsas_cache[cache_key] = lorsas
    _global_replacement_models_cache[cache_key] = replacement_model


def clear_cached_sae_resources(*, keep_model_name: Optional[str] = None) -> None:
    for cache_key in list(_global_transcoders_cache.keys()):
        if keep_model_name is not None and cache_key == keep_model_name:
            continue
        for sae in _global_transcoders_cache[cache_key].values():
            try:
                if hasattr(sae, "to"):
                    sae.to("cpu")
            except Exception:
                continue
        del _global_transcoders_cache[cache_key]

    for cache_key in list(_global_lorsas_cache.keys()):
        if keep_model_name is not None and cache_key == keep_model_name:
            continue
        for sae in _global_lorsas_cache[cache_key]:
            try:
                if hasattr(sae, "to"):
                    sae.to("cpu")
            except Exception:
                continue
        del _global_lorsas_cache[cache_key]

    for cache_key in list(_global_replacement_models_cache.keys()):
        if keep_model_name is not None and cache_key == keep_model_name:
            continue
        del _global_replacement_models_cache[cache_key]


def load_model_and_transcoders(
    model_name: str,
    device: str,
    tc_base_path: str,
    lorsa_base_path: str,
    n_layers: int = 15,
    hooked_model: Optional[HookedTransformer] = None,
    loading_logs: Optional[list] = None,
    cancel_flag: Optional[dict] = None,
    cache_key: Optional[str] = None,
) -> Tuple[ReplacementModel, Dict[int, SparseAutoEncoder], List[LowRankSparseAttention]]:
    logger = logging.getLogger(__name__)

    if cache_key is None:
        cache_key = model_name

    def add_log(message: str) -> None:
        print(message)
        logger.info(message)
        if loading_logs is not None:
            loading_logs.append({"timestamp": time.time(), "message": message})

    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = (
        get_cached_models(cache_key)
    )
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"Use cached model, transcoders and lorsas: {model_name}")
                return cached_replacement_model, cached_transcoders, cached_lorsas

    with _loading_lock:
        cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = (
            get_cached_models(cache_key)
        )
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
                if cached_replacement_model is not None:
                    add_log(f"Use cached model, transcoders and lorsas (double check): {cache_key}")
                    return cached_replacement_model, cached_transcoders, cached_lorsas

        if _is_loading.get(cache_key, False):
            add_log(f"Model {cache_key} is being loaded by another thread, waiting...")

    wait_count = 0
    max_wait = 600
    while _is_loading.get(cache_key, False) and wait_count < max_wait:
        time.sleep(1)
        wait_count += 1
        if wait_count % 10 == 0:
            add_log(f"Waiting for model to load... ({wait_count} seconds)")

    cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = (
        get_cached_models(cache_key)
    )
    if cached_transcoders is not None and cached_lorsas is not None:
        if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
            if cached_replacement_model is not None:
                add_log(f"Use cached model, transcoders and lorsas (after waiting): {cache_key}")
                return cached_replacement_model, cached_transcoders, cached_lorsas

    with _loading_lock:
        cached_hooked_model, cached_transcoders, cached_lorsas, cached_replacement_model = (
            get_cached_models(cache_key)
        )
        if cached_transcoders is not None and cached_lorsas is not None:
            if len(cached_transcoders) == n_layers and len(cached_lorsas) == n_layers:
                if cached_replacement_model is not None:
                    add_log(f"Use cached model, transcoders and lorsas (final check): {cache_key}")
                    return cached_replacement_model, cached_transcoders, cached_lorsas

        _is_loading[cache_key] = True
        add_log(f"Get loading lock, start loading model: {cache_key}")

    try:
        add_log(f"Start loading model and transcoders: {model_name}")

        if hooked_model is not None:
            add_log("Use incoming HookedTransformer model")
            model = hooked_model
        elif cached_hooked_model is not None:
            add_log("Use cached HookedTransformer model")
            model = cached_hooked_model
        else:
            add_log("Load new HookedTransformer model...")
            model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                device=device,
                dtype=torch.float32,
            ).eval()
            if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
                model.cfg.device = device
            _global_hooked_models[model_name] = model
            add_log("HookedTransformer model loaded")

        if cache_key not in _global_transcoders_cache:
            _global_transcoders_cache[cache_key] = {}
        transcoders = _global_transcoders_cache[cache_key]

        add_log(f"Start loading Transcoders, {n_layers} layers...")
        for layer in range(n_layers):
            if cancel_flag is not None:
                if "check_fn" in cancel_flag and callable(cancel_flag["check_fn"]):
                    should_cancel = cancel_flag["check_fn"]()
                else:
                    should_cancel = cancel_flag.get("should_cancel", False)
                if should_cancel:
                    add_log(f"Loading interrupted (TC Layer {layer}/{n_layers - 1})")
                    raise InterruptedError("Loading interrupted by user")

            if layer in transcoders:
                add_log(f"  [TC Layer {layer}/{n_layers - 1}] Already cached, skip loading")
                continue

            tc_path = f"{tc_base_path}/L{layer}"
            add_log(f"  [TC Layer {layer}/{n_layers - 1}] Start loading: {tc_path}")
            start_time = time.time()
            transcoders[layer] = SparseAutoEncoder.from_pretrained(
                tc_path,
                dtype=torch.float32,
                device=device,
            )
            add_log(
                f"  [TC Layer {layer}/{n_layers - 1}] Loaded successfully, time: {time.time() - start_time:.2f} seconds"
            )

        add_log(f"All Transcoders loaded successfully, {len(transcoders)} layers")

        if cache_key not in _global_lorsas_cache:
            _global_lorsas_cache[cache_key] = []
        lorsas = _global_lorsas_cache[cache_key]

        add_log(f"Start loading Lorsas, {n_layers} layers...")
        for layer in range(n_layers):
            if cancel_flag is not None:
                if "check_fn" in cancel_flag and callable(cancel_flag["check_fn"]):
                    should_cancel = cancel_flag["check_fn"]()
                else:
                    should_cancel = cancel_flag.get("should_cancel", False)
                if should_cancel:
                    add_log(f"Loading interrupted (Lorsa Layer {layer}/{n_layers - 1})")
                    raise InterruptedError("Loading interrupted by user")

            if layer < len(lorsas):
                add_log(f"  [Lorsa Layer {layer}/{n_layers - 1}] Already cached, skip loading")
                continue

            lorsa_path = f"{lorsa_base_path}/L{layer}"
            add_log(f"  [Lorsa Layer {layer}/{n_layers - 1}] Start loading: {lorsa_path}")
            start_time = time.time()
            lorsas.append(LowRankSparseAttention.from_pretrained(lorsa_path, device=device))
            add_log(
                f"  [Lorsa Layer {layer}/{n_layers - 1}] Loaded successfully, time: {time.time() - start_time:.2f} seconds"
            )

        add_log(f"All Lorsas loaded successfully, {len(lorsas)} layers")

        add_log("Create ReplacementModel...")
        replacement_model = ReplacementModel.from_pretrained_model(model, transcoders, lorsas)
        if hasattr(replacement_model, "cfg") and hasattr(replacement_model.cfg, "device"):
            replacement_model.cfg.device = device
        replacement_model.to(device)
        add_log("ReplacementModel created successfully")

        set_cached_models(cache_key, model, transcoders, lorsas, replacement_model)
        add_log(f"Models, transcoders and lorsas cached: {cache_key}")
        return replacement_model, transcoders, lorsas
    except Exception as exc:
        add_log(f"Error loading {cache_key}: {exc}")
        try:
            clear_cached_sae_resources()
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    add_log("Called torch.cuda.empty_cache() after exception to release memory")
            except Exception:
                pass
        raise
    finally:
        with _loading_lock:
            _is_loading[cache_key] = False
            add_log(f"Release loading lock: {cache_key}")

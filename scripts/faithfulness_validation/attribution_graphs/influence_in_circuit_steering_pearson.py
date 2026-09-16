from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import torch
from transformer_lens import HookedTransformer

from lm_saes import LowRankSparseAttention, SparseAutoEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "server"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lm_saes.circuit.leela_board import LeelaBoard  # noqa: E402
from server.constants import BT4_MODEL_NAME, get_bt4_sae_combo  # noqa: E402

DEFAULT_DEVICE = "cuda"
DEFAULT_COMBO_ID = "k_30_e_16"
DEFAULT_STEERING_SCALE = 0.0
DEFAULT_MAX_GRAPH_FEATURES = None
DEFAULT_RANDOM_SEED = 0
DEFAULT_CIRCUITS_DIR = Path(__file__).resolve().parent / "circuits_raw_influence"
DEFAULT_INFLUENCE_FIELD = "signed_link_influence"
STEERABLE_FEATURE_TYPES = ("lorsa", "transcoder")
NODE_ID_RE = re.compile(r"^(-?\d+)_(-?\d+)_(-?\d+)$")


@dataclass(frozen=True)
class FeatureConfig:
    feature_type: str
    layer: int
    pos: int
    feature_id: int


@dataclass(frozen=True)
class FeatureCandidate:
    config: FeatureConfig
    influence: float
    abs_influence: float
    influence_source: str
    graph_activation: Optional[float]


@dataclass(frozen=True)
class FeatureEffect:
    group: str
    feature_type: str
    layer: int
    pos: int
    feature_id: int
    activation_value: float
    target_logit_before: float
    target_logit_after: float
    target_logit_drop: float
    target_prob_before: float
    target_prob_after: float
    target_prob_drop: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate attribution-graph influence by comparing node influence against "
            "target-move logit drop after removing each feature's current decoder contribution."
        )
    )
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        default=DEFAULT_CIRCUITS_DIR,
        help="Directory containing circuit JSON files.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary/detail outputs.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cuda or cpu")
    parser.add_argument("--combo-id", default=DEFAULT_COMBO_ID, help="BT4 SAE combo id")
    parser.add_argument(
        "--steering-scale",
        type=float,
        default=DEFAULT_STEERING_SCALE,
        help=(
            "Feature activation multiplier at the intervention site. "
            "0.0 ablates the feature contribution, 1.0 leaves it unchanged."
        ),
    )
    parser.add_argument(
        "--max-graph-features-per-circuit",
        type=int,
        default=DEFAULT_MAX_GRAPH_FEATURES,
        help=(
            "Optional top-K steerable graph features (by |influence|) to evaluate per circuit. "
            "By default, evaluate all steerable graph features. "
            "Only lorsa/transcoder graph nodes are steerable; error/embedding nodes are excluded."
        ),
    )
    parser.add_argument("--max-circuits", type=int, default=None, help="Optional circuit cap.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--influence-field",
        choices=("signed_link_influence", "raw_influence", "influence"),
        default=DEFAULT_INFLUENCE_FIELD,
        help=(
            "Graph score used as the attribution score. signed_link_influence recomputes a "
            "signed path score from signed graph edges; raw_influence/influence read JSON node fields."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_model_bundle(
    *,
    device: str,
    combo_id: str,
) -> Tuple[HookedTransformer, Dict[int, SparseAutoEncoder], List[LowRankSparseAttention]]:
    combo = get_bt4_sae_combo(combo_id)
    resolved_device = resolve_device(device)

    model = HookedTransformer.from_pretrained_no_processing(
        BT4_MODEL_NAME,
        dtype=torch.float32,
    ).eval()
    if resolved_device != "cpu":
        model = model.to(resolved_device)

    transcoders = {
        layer: SparseAutoEncoder.from_pretrained(
            str(Path(combo["tc_base_path"]) / f"L{layer}"),
            dtype=torch.float32,
            device=resolved_device,
        )
        for layer in range(15)
    }
    lorsas = [
        LowRankSparseAttention.from_pretrained(
            str(Path(combo["lorsa_base_path"]) / f"L{layer}"),
            dtype=torch.float32,
            device=resolved_device,
        )
        for layer in range(15)
    ]
    return model, transcoders, lorsas


def list_circuit_files(circuits_dir: Path) -> List[Path]:
    circuit_files: List[Path] = []
    for path in sorted(circuits_dir.rglob("*.json")):
        if path.name.startswith("manifest"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
            circuit_files.append(path)
    return circuit_files


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_graph_node_config(node: Dict[str, Any]) -> Optional[FeatureConfig]:
    node_id = str(node.get("node_id", ""))
    match = NODE_ID_RE.match(node_id)
    if not match:
        return None
    first = int(match.group(1))
    local_feature_id = int(match.group(2))
    if first < 0 or local_feature_id < 0:
        return None
    actual_feature_type = "lorsa" if first % 2 == 0 else "transcoder"
    actual_layer = first // 2
    if actual_layer < 0 or actual_layer >= 15:
        return None
    pos = int(node["ctx_idx"])
    if pos < 0:
        return None
    return FeatureConfig(
        feature_type=actual_feature_type,
        layer=actual_layer,
        pos=pos,
        feature_id=local_feature_id,
    )


def compute_signed_link_influence(
    payload: Dict[str, Any],
    *,
    max_iter: int = 1000,
    eps: float = 1e-30,
) -> Dict[str, float]:
    """Propagate target-logit influence backward through signed graph edges.

    Exported graph JSON stores links as source -> target, while the graph
    matrix uses rows as downstream targets and columns as upstream sources.
    To preserve signs, each target row is normalized by the sum of absolute
    incoming weights, then influence is propagated from target logit nodes
    backward to upstream feature nodes.
    """
    nodes = payload.get("nodes", [])
    node_ids = {str(node.get("node_id")) for node in nodes}
    incoming_by_target: Dict[str, List[Tuple[str, float]]] = {}
    abs_row_sum: Dict[str, float] = {}

    for link in payload.get("links", []):
        source = str(link.get("source"))
        target = str(link.get("target"))
        if source not in node_ids or target not in node_ids:
            continue
        weight = float(link.get("weight") or 0.0)
        if weight == 0.0:
            continue
        incoming_by_target.setdefault(target, []).append((source, weight))
        abs_row_sum[target] = abs_row_sum.get(target, 0.0) + abs(weight)

    current: Dict[str, float] = {}
    for node in nodes:
        if node.get("feature_type") == "logit" and node.get("is_target_logit"):
            node_id = str(node["node_id"])
            # Match graph_lc0.prune_graph: target logit rows are seeded by logit probability.
            current[node_id] = float(node.get("token_prob") or 1.0)
    if not current:
        for node in nodes:
            if node.get("feature_type") == "logit":
                node_id = str(node["node_id"])
                current[node_id] = float(node.get("token_prob") or 1.0)

    influence: Dict[str, float] = {node_id: 0.0 for node_id in node_ids}
    for _ in range(max_iter):
        next_current: Dict[str, float] = {}
        for target, target_value in current.items():
            if abs(target_value) <= eps:
                continue
            denom = abs_row_sum.get(target, 0.0)
            if denom <= 0.0:
                continue
            for source, weight in incoming_by_target.get(target, []):
                propagated = target_value * (weight / denom)
                next_current[source] = next_current.get(source, 0.0) + propagated
        if not next_current or sum(abs(value) for value in next_current.values()) <= eps:
            break
        for node_id, value in next_current.items():
            influence[node_id] = influence.get(node_id, 0.0) + value
        current = next_current

    return influence


def extract_graph_candidates(
    payload: Dict[str, Any],
    *,
    max_graph_features: Optional[int],
    influence_field: str = DEFAULT_INFLUENCE_FIELD,
) -> List[FeatureCandidate]:
    dedup: Dict[FeatureConfig, FeatureCandidate] = {}
    signed_link_influence = (
        compute_signed_link_influence(payload)
        if influence_field == "signed_link_influence"
        else {}
    )
    for node in payload.get("nodes", []):
        config = parse_graph_node_config(node)
        if config is None:
            continue
        if influence_field == "signed_link_influence":
            if str(node.get("node_id")) not in signed_link_influence:
                continue
            influence = float(signed_link_influence[str(node["node_id"])])
        else:
            if node.get(influence_field) is None:
                continue
            influence = float(node[influence_field])
        candidate = FeatureCandidate(
            config=config,
            influence=influence,
            abs_influence=abs(influence),
            influence_source=influence_field,
            graph_activation=(
                float(node["activation"])
                if node.get("activation") is not None
                else None
            ),
        )

        prev = dedup.get(config)
        if prev is None or candidate.abs_influence > prev.abs_influence:
            dedup[config] = candidate

    candidates = sorted(dedup.values(), key=lambda item: item.abs_influence, reverse=True)
    if max_graph_features is not None:
        return candidates[:max_graph_features]
    return candidates


def get_fen_and_target_move(payload: Dict[str, Any]) -> Tuple[str, str]:
    metadata = payload.get("metadata", {})
    fen = metadata.get("prompt") or (
        metadata.get("prompt_tokens", [None])[0]
        if isinstance(metadata.get("prompt_tokens"), list)
        else None
    )
    target_move = metadata.get("target_move")
    if not fen or not target_move:
        raise ValueError("Circuit JSON is missing metadata.prompt or metadata.target_move")
    return str(fen), str(target_move)


def get_activation_tensor(
    *,
    original_cache: Dict[str, torch.Tensor],
    feature_type: str,
    layer: int,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    activation_cache: Dict[Tuple[str, int], torch.Tensor],
) -> torch.Tensor:
    key = (feature_type, layer)
    if key in activation_cache:
        return activation_cache[key]

    if feature_type == "transcoder":
        hook_name = f"blocks.{layer}.resid_mid_after_ln"
        acts = transcoders[layer].encode(original_cache[hook_name])
    elif feature_type == "lorsa":
        hook_name = f"blocks.{layer}.hook_attn_in"
        acts = lorsas[layer].encode(original_cache[hook_name])
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    activation_cache[key] = acts
    return acts


def build_active_feature_pool(
    *,
    original_cache: Dict[str, torch.Tensor],
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    activation_cache: Dict[Tuple[str, int], torch.Tensor],
) -> Tuple[List[FeatureConfig], Dict[str, Dict[int, int]], int]:
    pool: List[FeatureConfig] = []
    feature_dims = {"transcoder": {}, "lorsa": {}}
    seq_len: Optional[int] = None

    for feature_type in STEERABLE_FEATURE_TYPES:
        for layer in range(15):
            acts = get_activation_tensor(
                original_cache=original_cache,
                feature_type=feature_type,
                layer=layer,
                transcoders=transcoders,
                lorsas=lorsas,
                activation_cache=activation_cache,
            )
            if seq_len is None:
                seq_len = int(acts.shape[1])
            feature_dims[feature_type][layer] = int(acts.shape[2])

            sparse = acts[0].to_sparse_coo().coalesce()
            indices = sparse.indices()
            if indices.numel() == 0:
                continue

            for idx in range(indices.shape[1]):
                pos = int(indices[0, idx].item())
                feature_id = int(indices[1, idx].item())
                pool.append(
                    FeatureConfig(
                        feature_type=feature_type,
                        layer=layer,
                        pos=pos,
                        feature_id=feature_id,
                    )
                )

    if seq_len is None:
        raise ValueError("Failed to infer sequence length from activations")
    return pool, feature_dims, seq_len


def is_valid_feature_config(
    config: FeatureConfig,
    *,
    feature_dims: Dict[str, Dict[int, int]],
    seq_len: int,
) -> bool:
    if config.feature_type not in feature_dims:
        return False
    if config.layer not in feature_dims[config.feature_type]:
        return False
    if config.pos < 0 or config.pos >= seq_len:
        return False
    if config.feature_id < 0 or config.feature_id >= feature_dims[config.feature_type][config.layer]:
        return False
    return True


def sample_random_active_features(
    *,
    pool: Sequence[FeatureConfig],
    excluded: set,
    sample_size: int,
    rng: random.Random,
) -> List[FeatureConfig]:
    candidates = [cfg for cfg in pool if cfg not in excluded]
    if not candidates:
        return []
    if len(candidates) >= sample_size:
        return rng.sample(candidates, sample_size)
    return [rng.choice(candidates) for _ in range(sample_size)]


def sample_random_any_features(
    *,
    feature_dims: Dict[str, Dict[int, int]],
    seq_len: int,
    excluded: set,
    sample_size: int,
    rng: random.Random,
) -> List[FeatureConfig]:
    sampled: List[FeatureConfig] = []
    seen = set()
    attempts = 0
    max_attempts = sample_size * 50

    while len(sampled) < sample_size and attempts < max_attempts:
        attempts += 1
        feature_type = rng.choice(list(STEERABLE_FEATURE_TYPES))
        layer = rng.randrange(15)
        n_features = feature_dims[feature_type][layer]
        cfg = FeatureConfig(
            feature_type=feature_type,
            layer=layer,
            pos=rng.randrange(seq_len),
            feature_id=rng.randrange(n_features),
        )
        if cfg in excluded or cfg in seen:
            continue
        seen.add(cfg)
        sampled.append(cfg)
    return sampled


def get_move_logit(policy_output: torch.Tensor, fen: str, move_uci: str) -> float:
    if policy_output.ndim > 1:
        policy_output = policy_output[0]

    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    try:
        idx = leela_board.uci2idx(move_uci)
    except (KeyError, IndexError):
        if len(move_uci) == 5 and move_uci[4] in ("q", "r", "b", "n"):
            idx = leela_board.uci2idx(move_uci[:4])
        else:
            raise
    return float(policy_output[idx].item())


def get_move_prob(policy_output: torch.Tensor, fen: str, move_uci: str) -> float:
    if policy_output.ndim > 1:
        policy_output = policy_output[0]

    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    board = chess.Board(fen)
    valid_moves: List[Tuple[str, float]] = []
    for legal_move in board.legal_moves:
        uci = legal_move.uci()
        try:
            idx = leela_board.uci2idx(uci)
        except (KeyError, IndexError):
            if len(uci) == 5 and uci[4] in ("q", "r", "b", "n"):
                idx = leela_board.uci2idx(uci[:4])
            else:
                continue
        valid_moves.append((uci, float(policy_output[idx].item())))

    valid_uci_list = [uci for uci, _ in valid_moves]
    logits = torch.tensor([logit for _, logit in valid_moves], dtype=torch.float32)
    probs = torch.softmax(logits - logits.max(), dim=0)
    move_index = valid_uci_list.index(move_uci)
    return float(probs[move_index].item())


def get_activation_value(
    *,
    config: FeatureConfig,
    original_cache: Dict[str, torch.Tensor],
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    activation_cache: Dict[Tuple[str, int], torch.Tensor],
) -> float:
    acts = get_activation_tensor(
        original_cache=original_cache,
        feature_type=config.feature_type,
        layer=config.layer,
        transcoders=transcoders,
        lorsas=lorsas,
        activation_cache=activation_cache,
    )
    return float(acts[0, config.pos, config.feature_id].item())


def get_decoder_vector(
    *,
    config: FeatureConfig,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
) -> torch.Tensor:
    if config.feature_type == "transcoder":
        return transcoders[config.layer].W_D[config.feature_id]
    if config.feature_type == "lorsa":
        return lorsas[config.layer].W_O[config.feature_id]
    raise ValueError(f"Unsupported feature_type: {config.feature_type}")


def get_hook_name(config: FeatureConfig) -> str:
    if config.feature_type == "transcoder":
        return f"blocks.{config.layer}.hook_mlp_out"
    if config.feature_type == "lorsa":
        return f"blocks.{config.layer}.hook_attn_out"
    raise ValueError(f"Unsupported feature_type: {config.feature_type}")


def evaluate_feature_effect(
    *,
    model: HookedTransformer,
    fen: str,
    target_move: str,
    config: FeatureConfig,
    steering_scale: float,
    original_output: Sequence[torch.Tensor],
    original_cache: Dict[str, torch.Tensor],
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    activation_cache: Dict[Tuple[str, int], torch.Tensor],
    group: str,
) -> FeatureEffect:
    original_policy = original_output[0]
    if original_policy.ndim == 2:
        original_policy = original_policy[0]

    activation_value = get_activation_value(
        config=config,
        original_cache=original_cache,
        transcoders=transcoders,
        lorsas=lorsas,
        activation_cache=activation_cache,
    )
    target_logit_before = get_move_logit(original_policy, fen, target_move)
    target_prob_before = get_move_prob(original_policy, fen, target_move)

    if math.isclose(activation_value, 0.0, abs_tol=1e-12) or math.isclose(steering_scale, 1.0, abs_tol=1e-12):
        return FeatureEffect(
            group=group,
            feature_type=config.feature_type,
            layer=config.layer,
            pos=config.pos,
            feature_id=config.feature_id,
            activation_value=activation_value,
            target_logit_before=target_logit_before,
            target_logit_after=target_logit_before,
            target_logit_drop=0.0,
            target_prob_before=target_prob_before,
            target_prob_after=target_prob_before,
            target_prob_drop=0.0,
        )

    decoder = get_decoder_vector(
        config=config,
        transcoders=transcoders,
        lorsas=lorsas,
    )
    intervention_val = (steering_scale - 1.0) * activation_value * decoder
    hook_name = get_hook_name(config)

    def _set_feature_contribution_scale(act: torch.Tensor, hook: Any) -> torch.Tensor:
        out = act.clone()
        delta = intervention_val.to(out.device)
        out[(slice(None), config.pos) if out.dim() == 3 else (config.pos,)] += delta
        return out

    model.reset_hooks()
    model.add_hook(hook_name, _set_feature_contribution_scale)
    with torch.no_grad():
        modified_output, _ = model.run_with_cache(fen, prepend_bos=False)
    model.reset_hooks()

    modified_policy = modified_output[0]
    if modified_policy.ndim == 2:
        modified_policy = modified_policy[0]

    target_logit_after = get_move_logit(modified_policy, fen, target_move)
    target_prob_after = get_move_prob(modified_policy, fen, target_move)

    return FeatureEffect(
        group=group,
        feature_type=config.feature_type,
        layer=config.layer,
        pos=config.pos,
        feature_id=config.feature_id,
        activation_value=activation_value,
        target_logit_before=target_logit_before,
        target_logit_after=target_logit_after,
        target_logit_drop=target_logit_before - target_logit_after,
        target_prob_before=target_prob_before,
        target_prob_after=target_prob_after,
        target_prob_drop=target_prob_before - target_prob_after,
    )


def summarize_effects(effects: Sequence[FeatureEffect]) -> Dict[str, float]:
    if not effects:
        return {
            "count": 0.0,
            "mean_activation": float("nan"),
            "mean_target_logit_drop": float("nan"),
            "mean_abs_target_logit_drop": float("nan"),
            "mean_target_prob_drop": float("nan"),
            "mean_abs_target_prob_drop": float("nan"),
        }

    count = float(len(effects))
    return {
        "count": count,
        "mean_activation": float(sum(x.activation_value for x in effects) / len(effects)),
        "mean_target_logit_drop": float(sum(x.target_logit_drop for x in effects) / len(effects)),
        "mean_abs_target_logit_drop": float(sum(abs(x.target_logit_drop) for x in effects) / len(effects)),
        "mean_target_prob_drop": float(sum(x.target_prob_drop for x in effects) / len(effects)),
        "mean_abs_target_prob_drop": float(sum(abs(x.target_prob_drop) for x in effects) / len(effects)),
    }


def pearsonr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")

    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def summarize_graph_influence_relationship(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    graph_rows = [row for row in rows if row.get("group") == "graph"]
    influences = [float(row["graph_influence"]) for row in graph_rows]
    logit_drops = [float(row["target_logit_drop"]) for row in graph_rows]

    return {
        "count": float(len(graph_rows)),
        "total_graph_influence": float(sum(influences)),
        "total_abs_graph_influence": float(sum(abs(value) for value in influences)),
        "total_target_logit_drop": float(sum(logit_drops)),
        "total_abs_target_logit_drop": float(sum(abs(value) for value in logit_drops)),
        "n_positive_graph_influence": float(sum(value > 0 for value in influences)),
        "n_negative_graph_influence": float(sum(value < 0 for value in influences)),
        "n_zero_graph_influence": float(sum(value == 0 for value in influences)),
        "pearson_influence_vs_target_logit_drop": pearsonr(influences, logit_drops),
        "pearson_abs_influence_vs_abs_target_logit_drop": pearsonr(
            [abs(value) for value in influences],
            [abs(value) for value in logit_drops],
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_circuit(
    *,
    circuit_path: Path,
    model: HookedTransformer,
    transcoders: Dict[int, SparseAutoEncoder],
    lorsas: List[LowRankSparseAttention],
    steering_scale: float,
    max_graph_features: Optional[int],
    influence_field: str,
    rng: random.Random,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = read_json(circuit_path)
    fen, target_move = get_fen_and_target_move(payload)
    graph_candidates_all = extract_graph_candidates(
        payload,
        max_graph_features=None,
        influence_field=influence_field,
    )

    with torch.no_grad():
        original_output, original_cache = model.run_with_cache(fen, prepend_bos=False)

    activation_cache: Dict[Tuple[str, int], torch.Tensor] = {}
    active_pool, feature_dims, seq_len = build_active_feature_pool(
        original_cache=original_cache,
        transcoders=transcoders,
        lorsas=lorsas,
        activation_cache=activation_cache,
    )

    graph_candidates_valid = [
        candidate
        for candidate in graph_candidates_all
        if is_valid_feature_config(
            candidate.config,
            feature_dims=feature_dims,
            seq_len=seq_len,
        )
    ]
    if max_graph_features is None:
        graph_candidates = graph_candidates_valid
    else:
        graph_candidates = graph_candidates_valid[:max_graph_features]
    if not graph_candidates:
        raise ValueError(
            f"No valid steerable graph features with {influence_field} found in circuit"
        )

    graph_candidate_by_config = {candidate.config: candidate for candidate in graph_candidates}
    graph_configs = [candidate.config for candidate in graph_candidates]
    graph_config_set = set(graph_configs)
    random_active_configs = sample_random_active_features(
        pool=active_pool,
        excluded=graph_config_set,
        sample_size=len(graph_configs),
        rng=rng,
    )
    random_any_configs = sample_random_any_features(
        feature_dims=feature_dims,
        seq_len=seq_len,
        excluded=graph_config_set,
        sample_size=len(graph_configs),
        rng=rng,
    )

    all_detail_rows: List[Dict[str, Any]] = []
    group_effects: Dict[str, List[FeatureEffect]] = {
        "graph": [],
        "random_active": [],
        "random_any": [],
    }
    groups = [
        ("graph", graph_configs),
        ("random_active", random_active_configs),
        ("random_any", random_any_configs),
    ]

    for group_name, configs in groups:
        for config in configs:
            effect = evaluate_feature_effect(
                model=model,
                fen=fen,
                target_move=target_move,
                config=config,
                steering_scale=steering_scale,
                original_output=original_output,
                original_cache=original_cache,
                transcoders=transcoders,
                lorsas=lorsas,
                activation_cache=activation_cache,
                group=group_name,
            )
            group_effects[group_name].append(effect)
            row = {
                "circuit_path": str(circuit_path),
                "fen": fen,
                "target_move": target_move,
                "graph_influence": (
                    graph_candidate_by_config[config].influence
                    if group_name == "graph"
                    else None
                ),
                "graph_abs_influence": (
                    graph_candidate_by_config[config].abs_influence
                    if group_name == "graph"
                    else None
                ),
                "graph_influence_source": (
                    graph_candidate_by_config[config].influence_source
                    if group_name == "graph"
                    else None
                ),
                "graph_activation": (
                    graph_candidate_by_config[config].graph_activation
                    if group_name == "graph"
                    else None
                ),
                **asdict(effect),
            }
            all_detail_rows.append(row)

    graph_relationship = summarize_graph_influence_relationship(all_detail_rows)
    summary = {
        "circuit_path": str(circuit_path),
        "target_move": target_move,
        "fen": fen,
        "n_graph_features_total": len(graph_candidates_valid),
        "n_graph_features_raw": len(graph_candidates_all),
        "n_graph_features_skipped_invalid": len(graph_candidates_all) - len(graph_candidates_valid),
        "n_graph_features_used": len(group_effects["graph"]),
        "graph_influence_field": influence_field,
        "n_random_active_used": len(group_effects["random_active"]),
        "n_random_any_used": len(group_effects["random_any"]),
        "graph": summarize_effects(group_effects["graph"]),
        "random_active": summarize_effects(group_effects["random_active"]),
        "random_any": summarize_effects(group_effects["random_any"]),
        "graph_influence_relationship": graph_relationship,
    }
    return summary, all_detail_rows


def flatten_summary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "circuit_path": summary["circuit_path"],
        "target_move": summary["target_move"],
        "fen": summary["fen"],
        "n_graph_features_total": summary["n_graph_features_total"],
        "n_graph_features_raw": summary["n_graph_features_raw"],
        "n_graph_features_skipped_invalid": summary["n_graph_features_skipped_invalid"],
        "n_graph_features_used": summary["n_graph_features_used"],
        "graph_influence_field": summary["graph_influence_field"],
        "n_random_active_used": summary["n_random_active_used"],
        "n_random_any_used": summary["n_random_any_used"],
    }
    for group_name in ("graph", "random_active", "random_any"):
        metrics = summary[group_name]
        for key, value in metrics.items():
            flat[f"{group_name}_{key}"] = value
    for key, value in summary["graph_influence_relationship"].items():
        flat[f"graph_relationship_{key}"] = value
    return flat


def aggregate_overall(detail_rows: Sequence[Dict[str, Any]], n_circuits: int) -> Dict[str, Any]:
    overall: Dict[str, Any] = {"n_circuits": n_circuits}
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {
        "graph": [],
        "random_active": [],
        "random_any": [],
    }
    for row in detail_rows:
        grouped_rows[row["group"]].append(row)

    for group_name in ("graph", "random_active", "random_any"):
        rows = grouped_rows[group_name]
        if not rows:
            overall[group_name] = {
                "count": 0.0,
                "mean_activation": float("nan"),
                "mean_target_logit_drop": float("nan"),
                "mean_abs_target_logit_drop": float("nan"),
                "mean_target_prob_drop": float("nan"),
                "mean_abs_target_prob_drop": float("nan"),
            }
            continue

        overall[group_name] = {
            "count": float(len(rows)),
            "mean_activation": float(sum(float(r["activation_value"]) for r in rows) / len(rows)),
            "mean_target_logit_drop": float(sum(float(r["target_logit_drop"]) for r in rows) / len(rows)),
            "mean_abs_target_logit_drop": float(
                sum(abs(float(r["target_logit_drop"])) for r in rows) / len(rows)
            ),
            "mean_target_prob_drop": float(sum(float(r["target_prob_drop"]) for r in rows) / len(rows)),
            "mean_abs_target_prob_drop": float(
                sum(abs(float(r["target_prob_drop"])) for r in rows) / len(rows)
            ),
        }

    graph_abs_logit = float(overall["graph"]["mean_abs_target_logit_drop"])
    graph_abs_prob = float(overall["graph"]["mean_abs_target_prob_drop"])
    comparisons: Dict[str, float] = {}
    for baseline in ("random_active", "random_any"):
        baseline_abs_logit = float(overall[baseline]["mean_abs_target_logit_drop"])
        baseline_abs_prob = float(overall[baseline]["mean_abs_target_prob_drop"])
        comparisons[f"graph_over_{baseline}_abs_logit_ratio"] = (
            graph_abs_logit / baseline_abs_logit
            if baseline_abs_logit > 0
            else float("nan")
        )
        comparisons[f"graph_over_{baseline}_abs_prob_ratio"] = (
            graph_abs_prob / baseline_abs_prob
            if baseline_abs_prob > 0
            else float("nan")
        )
    overall["comparisons"] = comparisons

    graph_rows = grouped_rows["graph"]
    graph_influences = [float(row["graph_influence"]) for row in graph_rows]
    graph_logit_drops = [float(row["target_logit_drop"]) for row in graph_rows]
    overall["graph_feature_level_relationship"] = {
        "count": float(len(graph_rows)),
        "n_positive_graph_influence": float(sum(value > 0 for value in graph_influences)),
        "n_negative_graph_influence": float(sum(value < 0 for value in graph_influences)),
        "n_zero_graph_influence": float(sum(value == 0 for value in graph_influences)),
        "pearson_influence_vs_target_logit_drop": pearsonr(graph_influences, graph_logit_drops),
        "pearson_abs_influence_vs_abs_target_logit_drop": pearsonr(
            [abs(value) for value in graph_influences],
            [abs(value) for value in graph_logit_drops],
        ),
    }

    circuit_totals: Dict[str, Dict[str, float]] = {}
    for row in graph_rows:
        circuit_path = str(row["circuit_path"])
        totals = circuit_totals.setdefault(
            circuit_path,
            {
                "graph_influence": 0.0,
                "graph_abs_influence": 0.0,
                "target_logit_drop": 0.0,
                "abs_target_logit_drop": 0.0,
            },
        )
        graph_influence = float(row["graph_influence"])
        target_logit_drop = float(row["target_logit_drop"])
        totals["graph_influence"] += graph_influence
        totals["graph_abs_influence"] += abs(graph_influence)
        totals["target_logit_drop"] += target_logit_drop
        totals["abs_target_logit_drop"] += abs(target_logit_drop)

    total_rows = list(circuit_totals.values())
    overall["graph_circuit_total_relationship"] = {
        "count": float(len(total_rows)),
        "pearson_total_influence_vs_total_target_logit_drop": pearsonr(
            [row["graph_influence"] for row in total_rows],
            [row["target_logit_drop"] for row in total_rows],
        ),
        "pearson_total_abs_influence_vs_total_abs_target_logit_drop": pearsonr(
            [row["graph_abs_influence"] for row in total_rows],
            [row["abs_target_logit_drop"] for row in total_rows],
        ),
    }
    return overall


def main() -> None:
    args = parse_args()
    circuits_dir = Path(args.circuits_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    rng = random.Random(args.seed)
    max_graph_features = args.max_graph_features_per_circuit

    circuit_files = list_circuit_files(circuits_dir)
    if args.max_circuits is not None:
        circuit_files = circuit_files[: args.max_circuits]
    if not circuit_files:
        raise ValueError(f"No circuit JSON files found in {circuits_dir}")

    model, transcoders, lorsas = load_model_bundle(
        device=args.device,
        combo_id=args.combo_id,
    )

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, circuit_path in enumerate(circuit_files, start=1):
        print(f"[{idx}/{len(circuit_files)}] {circuit_path.name}")
        try:
            summary, details = evaluate_circuit(
                circuit_path=circuit_path,
                model=model,
                transcoders=transcoders,
                lorsas=lorsas,
                steering_scale=args.steering_scale,
                max_graph_features=max_graph_features,
                influence_field=args.influence_field,
                rng=rng,
            )
            summary_rows.append(summary)
            detail_rows.extend(details)
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append({"circuit_path": str(circuit_path), "error": str(exc), "traceback": tb})
            print(f"  failed: {exc}", file=sys.stderr)
            print(tb, file=sys.stderr)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    overall = aggregate_overall(detail_rows, len(summary_rows))
    manifest = {
        "circuits_dir": str(circuits_dir),
        "output_dir": str(output_dir),
        "combo_id": args.combo_id,
        "device": resolve_device(args.device),
        "steering_scale": args.steering_scale,
        "graph_influence_field": args.influence_field,
        "max_graph_features_per_circuit": max_graph_features,
        "all_graph_features": max_graph_features is None,
        "n_circuits_total": len(circuit_files),
        "n_circuits_succeeded": len(summary_rows),
        "n_circuits_failed": len(failures),
        "overall": overall,
        "failures": failures,
    }

    write_json(output_dir / "summary.json", manifest)
    write_jsonl(output_dir / "per_circuit_summary.jsonl", summary_rows)
    write_csv(output_dir / "per_circuit_summary.csv", [flatten_summary_row(row) for row in summary_rows])
    write_jsonl(output_dir / "feature_effects.jsonl", detail_rows)


if __name__ == "__main__":
    main()

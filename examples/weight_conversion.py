"""
BT4 (LC0) ONNX -> PyTorch checkpoint conversion.

This script is a cleaned-up, runnable Python version of `examples/bt4_transfer.ipynb`.
It converts a LC0 BT4 ONNX model into a PyTorch `state_dict` checkpoint (`BT4.pt`)
that can be loaded by the local TransformerLens fork in this repository.

No CLI arguments are used. Edit the constants below if needed.
"""

# ruff: noqa: I001

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# User-editable default paths
# -----------------------------

# Prefer setting this file under a repo-local folder so the path is anonymous and reproducible.
# Put your ONNX file here:
#   models/lc0/BT4-1024x15x32h-swa-6147500.onnx
DEFAULT_ONNX_PATH = REPO_ROOT / "BT4-1024x15x32h-swa-6147500.onnx"

# Output checkpoint (used by TransformerLens/transformer_lens/loading_from_pretrained.py with Project_root=/path/to/models)
DEFAULT_OUTPUT_PT = REPO_ROOT / "/path/to/models/lc0/BT4.pt"


def _require_onnx_deps() -> tuple[object, object]:
    try:
        import onnx  # type: ignore
        import onnx2torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "This script requires `onnx` and `onnx2torch`.\n"
            "Install them (e.g. `uv sync` with the repo's dependencies) and retry."
        ) from exc
    return onnx, onnx2torch


class LayerNorm(nn.Module):
    """LayerNorm with parameter names matching the notebook (w/b instead of weight/bias)."""

    def __init__(self, d_model: int, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.w + self.b


class AttentionBody(nn.Module):
    """BT4 attention_body embedding stack (shape-focused, for weight loading)."""

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model

        # [B, 64, 12] -> [B, 768] -> Linear(768->32768) -> [B, 64, 512]
        self.embedding_preprocess = nn.Linear(768, 32768)

        # [B, 64, 112] concat [B, 64, 512] => [B, 64, 624] -> Linear(624->d_model)
        self.main_linear = nn.Linear(624, d_model)

        # MA gating parameters (registered as parameters, not modules)
        self.ma_gating_mul = nn.Parameter(torch.randn(64, d_model))
        self.ma_gating_add = nn.Parameter(torch.randn(64, d_model))

        # FFN in attention body: 1024 -> 1536 -> 1024, with alpha + residual
        self.ffn_dense1 = nn.Linear(d_model, 1536)
        self.ffn_dense2 = nn.Linear(1536, d_model)
        self.ffn_alpha = nn.Parameter(torch.ones(1))

        # NOTE: In the notebook, only ln.weight / ln2.weight were explicitly mapped.
        # We keep standard LayerNorm modules so the parameter names exist.
        self.ln = nn.LayerNorm(d_model, eps=1e-3)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward is not used for conversion; keep a minimal implementation for sanity.
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, 64, 112)
        pos_slice = x[:, :, :12].reshape(batch_size, -1)
        pos_processed = self.embedding_preprocess(pos_slice).reshape(batch_size, 64, 512)
        x = torch.cat([x, pos_processed], dim=-1).reshape(-1, 624)
        x = self.main_linear(x)
        x = F.mish(x)
        x = self.ln(x).reshape(batch_size, 64, self.d_model)
        x = x * self.ma_gating_mul.unsqueeze(0) + self.ma_gating_add.unsqueeze(0)
        x = x.reshape(-1, self.d_model)
        residual = x
        x = self.ffn_dense2(F.mish(self.ffn_dense1(x))) * self.ffn_alpha + residual
        x = self.ln2(x).reshape(batch_size, 64, self.d_model)
        return x


class SmolGen(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.compress = nn.Linear(d_model, 32, bias=False)
        self.dense1 = nn.Linear(2048, 256)
        self.ln1 = nn.LayerNorm(256, eps=1e-3)
        self.dense2 = nn.Linear(256, 256 * n_heads)
        self.ln2 = nn.LayerNorm(256 * n_heads, eps=1e-3)
        self.smol_weight_gen = nn.Linear(256, 4096, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        compressed = self.compress(x).view(batch_size, -1)  # [B, 64, 32] -> [B, 2048]
        x = self.ln1(F.silu(self.dense1(compressed)))
        x = self.ln2(F.silu(self.dense2(x)))
        x = x.view(batch_size, self.n_heads, 256)
        weights = self.smol_weight_gen(x).view(batch_size, self.n_heads, 64, 64)
        return weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.qk_scale = nn.Parameter(torch.tensor([1.0 / (self.d_k**0.5)]))
        self.smolgen = SmolGen(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale
        scores = scores + self.smolgen(x)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class LC0MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense2(F.mish(self.dense1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 1024, n_heads: int = 32, d_ff: int = 1536):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model, eps=1e-3)
        self.ln2 = LayerNorm(d_model, eps=1e-3)
        self.mlp = LC0MLP(d_model, d_ff)
        self.alpha_input = nn.Parameter(torch.ones(1))
        self.alpha_out1 = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(self.mha(x) * self.alpha_input + residual)
        residual2 = x
        x = self.ln2(self.mlp(x) * self.alpha_out1 + residual2)
        return x


class PolicyHead(nn.Module):
    def __init__(self, d_model: int = 1024, policy_dim: int = 1858):
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.ones(1))
        self.promotion = nn.Linear(d_model, 4, bias=False)
        self.indices = nn.Parameter(torch.randn(policy_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # Not needed for conversion; keep a minimal forward.
        batch_size = x.shape[0]
        x = F.mish(self.dense1(x))
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        promotion_slice = k[:, 56:64, :]
        promotion_out = self.promotion(promotion_slice)  # [B, 8, 4]
        promotion_out = promotion_out.transpose(1, 2)  # [B, 4, 8]
        a, b = torch.split(promotion_out, [3, 1], dim=1)
        promotion_out = (a + b).transpose(1, 2).reshape(batch_size, 1, 24)
        promotion_slice2 = scores[:, 48:56, 56:64].reshape(-1, 64, 1)
        promotion_out2 = torch.cat([promotion_slice2, promotion_slice2, promotion_slice2], dim=-1).reshape(-1, 8, 24)
        promotion = (promotion_out2 + promotion_out).reshape(-1, 3, 64)
        policy = torch.cat([scores, promotion], dim=1).reshape(-1, 4288)
        idx = self.indices.detach().long()
        return policy[:, idx]


class ValueHead(nn.Module):
    def __init__(self, d_model: int = 1024, d_value_head: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(d_model, d_value_head)
        self.dense1 = nn.Linear(d_value_head * 64, 128)
        self.dense2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        batch_size = x.shape[0]
        x = F.mish(self.embed(x.view(batch_size * 64, self.d_model)))
        x = F.mish(self.dense1(x.view(batch_size, -1)))
        return torch.softmax(self.dense2(x), dim=-1)


class MLHHead(nn.Module):
    def __init__(self, d_model: int = 1024, d_mlh_head: int = 32):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(d_model, d_mlh_head)
        self.dense1 = nn.Linear(d_mlh_head * 64, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        batch_size = x.shape[0]
        x = F.mish(self.embed(x.view(batch_size * 64, self.d_model)))
        x = F.mish(self.dense1(x.view(batch_size, -1)))
        return F.mish(self.dense2(x))


@dataclass(frozen=True)
class BT4Config:
    d_model: int = 1024
    n_heads: int = 32
    n_layers: int = 15
    d_ff: int = 1536


class CleanLC0Model(nn.Module):
    def __init__(self, cfg: BT4Config = BT4Config()):
        super().__init__()
        self.cfg = cfg
        self.attention_body = AttentionBody(cfg.d_model)
        self.encoders = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)])
        self.policy_head = PolicyHead(cfg.d_model)
        self.value_head = ValueHead(cfg.d_model)
        self.mlh_head = MLHHead(cfg.d_model)

    def create_weight_mapping_from_graph(self, onnx_model) -> Dict[str, str]:
        # Mapping is taken from the notebook (based on ONNX initializer indices).
        mapping: Dict[str, str] = {}

        mapping.update(
            {
                "attention_body.embedding_preprocess.weight": "initializers.onnx_initializer_4",
                "attention_body.embedding_preprocess.bias": "initializers.onnx_initializer_5",
                "attention_body.main_linear.weight": "initializers.onnx_initializer_8",
                "attention_body.main_linear.bias": "initializers.onnx_initializer_9",
                "attention_body.ma_gating_mul": "initializers.onnx_initializer_11",
                "attention_body.ma_gating_add": "initializers.onnx_initializer_12",
                "attention_body.ffn_dense1.weight": "initializers.onnx_initializer_14",
                "attention_body.ffn_dense1.bias": "initializers.onnx_initializer_15",
                "attention_body.ffn_dense2.weight": "initializers.onnx_initializer_16",
                "attention_body.ffn_dense2.bias": "initializers.onnx_initializer_17",
                "attention_body.ffn_alpha": "initializers.onnx_initializer_18",
                "attention_body.ln.weight": "attn_body/ln.weight",
                "attention_body.ln2.weight": "attn_body/ln2.weight",
            }
        )

        for i in range(self.cfg.n_layers):
            encoder_prefix = f"encoders.{i}"
            base_idx = 19 + i * 28

            mapping.update(
                {
                    f"{encoder_prefix}.mha.q_proj.weight": f"initializers.onnx_initializer_{base_idx}",
                    f"{encoder_prefix}.mha.q_proj.bias": f"initializers.onnx_initializer_{base_idx + 1}",
                    f"{encoder_prefix}.mha.k_proj.weight": f"initializers.onnx_initializer_{base_idx + 3}",
                    f"{encoder_prefix}.mha.k_proj.bias": f"initializers.onnx_initializer_{base_idx + 4}",
                    f"{encoder_prefix}.mha.v_proj.weight": f"initializers.onnx_initializer_{base_idx + 6}",
                    f"{encoder_prefix}.mha.v_proj.bias": f"initializers.onnx_initializer_{base_idx + 7}",
                    f"{encoder_prefix}.mha.out_proj.weight": f"initializers.onnx_initializer_{base_idx + 20}",
                    f"{encoder_prefix}.mha.out_proj.bias": f"initializers.onnx_initializer_{base_idx + 21}",
                    f"{encoder_prefix}.mha.smolgen.compress.weight": f"initializers.onnx_initializer_{base_idx + 10}",
                    f"{encoder_prefix}.mha.smolgen.dense1.weight": f"initializers.onnx_initializer_{base_idx + 12}",
                    f"{encoder_prefix}.mha.smolgen.dense1.bias": f"initializers.onnx_initializer_{base_idx + 13}",
                    f"{encoder_prefix}.mha.smolgen.dense2.weight": f"initializers.onnx_initializer_{base_idx + 14}",
                    f"{encoder_prefix}.mha.smolgen.dense2.bias": f"initializers.onnx_initializer_{base_idx + 15}",
                    f"{encoder_prefix}.mha.smolgen.smol_weight_gen.weight": f"initializers.onnx_initializer_{base_idx + 17}",
                    f"{encoder_prefix}.mlp.dense1.weight": f"initializers.onnx_initializer_{base_idx + 23}",
                    f"{encoder_prefix}.mlp.dense1.bias": f"initializers.onnx_initializer_{base_idx + 24}",
                    f"{encoder_prefix}.mlp.dense2.weight": f"initializers.onnx_initializer_{base_idx + 25}",
                    f"{encoder_prefix}.mlp.dense2.bias": f"initializers.onnx_initializer_{base_idx + 26}",
                }
            )

            alpha_input_idx = 41 + i * 28
            alpha_out1_idx = 46 + i * 28
            mapping.update(
                {
                    f"{encoder_prefix}.alpha_input": f"initializers.onnx_initializer_{alpha_input_idx}",
                    f"{encoder_prefix}.alpha_out1": f"initializers.onnx_initializer_{alpha_out1_idx}",
                }
            )

            qk_scale_idx = 28 + i * 28
            mapping[f"{encoder_prefix}.mha.qk_scale"] = f"initializers.onnx_initializer_{qk_scale_idx}"

            # LayerNorm weights/biases (custom LayerNorm uses w/b names)
            mapping.update(
                {
                    f"{encoder_prefix}.ln1.w": f"encoder{i}/ln1.weight",
                    f"{encoder_prefix}.ln1.b": f"encoder{i}/ln1.bias",
                    f"{encoder_prefix}.ln2.w": f"encoder{i}/ln2.weight",
                    f"{encoder_prefix}.ln2.b": f"encoder{i}/ln2.bias",
                    f"{encoder_prefix}.mha.smolgen.ln1.weight": f"encoder{i}/smolgen/ln1.weight",
                    f"{encoder_prefix}.mha.smolgen.ln1.bias": f"encoder{i}/smolgen/ln1.bias",
                    f"{encoder_prefix}.mha.smolgen.ln2.weight": f"encoder{i}/smolgen/ln2.weight",
                    f"{encoder_prefix}.mha.smolgen.ln2.bias": f"encoder{i}/smolgen/ln2.bias",
                }
            )

        mapping.update(
            {
                "policy_head.dense1.weight": "initializers.onnx_initializer_439",
                "policy_head.dense1.bias": "initializers.onnx_initializer_440",
                "policy_head.q_proj.weight": "initializers.onnx_initializer_441",
                "policy_head.q_proj.bias": "initializers.onnx_initializer_442",
                "policy_head.k_proj.weight": "initializers.onnx_initializer_444",
                "policy_head.k_proj.bias": "initializers.onnx_initializer_445",
                "policy_head.scale": "initializers.onnx_initializer_447",
                "policy_head.promotion.weight": "initializers.onnx_initializer_450",
                "policy_head.indices": "initializers.onnx_initializer_459",
                "value_head.embed.weight": "initializers.onnx_initializer_460",
                "value_head.embed.bias": "initializers.onnx_initializer_461",
                "value_head.dense1.weight": "initializers.onnx_initializer_463",
                "value_head.dense1.bias": "initializers.onnx_initializer_464",
                "value_head.dense2.weight": "initializers.onnx_initializer_465",
                "value_head.dense2.bias": "initializers.onnx_initializer_466",
                "mlh_head.embed.weight": "initializers.onnx_initializer_467",
                "mlh_head.embed.bias": "initializers.onnx_initializer_468",
                "mlh_head.dense1.weight": "initializers.onnx_initializer_470",
                "mlh_head.dense1.bias": "initializers.onnx_initializer_471",
                "mlh_head.dense2.weight": "initializers.onnx_initializer_472",
                "mlh_head.dense2.bias": "initializers.onnx_initializer_473",
            }
        )

        return mapping

    def load_from_onnx2torch_module(self, onnx2torch_module: nn.Module) -> None:
        onnx_state_dict = onnx2torch_module.state_dict()
        all_onnx_weights: Dict[str, torch.Tensor] = dict(onnx_state_dict)
        graph_mapping = self.create_weight_mapping_from_graph(onnx2torch_module)

        matched = 0
        transposed_preference = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
            "policy_head.dense1.weight",
            "smol_weight_gen.weight",
            "embedding_preprocess.weight",
        )

        for name, param in self.named_parameters():
            onnx_name = graph_mapping.get(name)
            if not onnx_name:
                continue
            onnx_weight = all_onnx_weights.get(onnx_name)
            if onnx_weight is None:
                continue

            if any(key in name for key in transposed_preference):
                if param.shape == tuple(reversed(onnx_weight.shape)):
                    param.data.copy_(onnx_weight.T)
                    matched += 1
                    continue

            if param.shape == onnx_weight.shape:
                param.data.copy_(onnx_weight)
                matched += 1
            elif param.shape == tuple(reversed(onnx_weight.shape)):
                param.data.copy_(onnx_weight.T)
                matched += 1

        print(f"[weight_conversion] Loaded {matched} parameters using the explicit mapping.")


def convert_bt4_onnx_to_pt(
    *,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    output_path: Path = DEFAULT_OUTPUT_PT,
    device: str | None = None,
) -> Path:
    onnx, onnx2torch = _require_onnx_deps()

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX file not found: {onnx_path}\n"
            "Place the BT4 ONNX file at the path above or edit DEFAULT_ONNX_PATH in this script."
        )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[weight_conversion] Using device: {resolved_device}")
    print(f"[weight_conversion] Loading ONNX: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    converted_model = onnx2torch.convert(onnx_model).to(resolved_device)

    clean_model = CleanLC0Model().to(resolved_device)
    clean_model.load_from_onnx2torch_module(converted_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_model.state_dict(), str(output_path))
    print(f"[weight_conversion] Saved checkpoint: {output_path}")
    return output_path


def main() -> None:
    convert_bt4_onnx_to_pt()


if __name__ == "__main__":
    main()


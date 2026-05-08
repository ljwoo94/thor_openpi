"""OPT-7: QKV and MLP projection fusion utilities.

Fuses separate Q/K/V linear projections into a single packed linear,
and gate_proj + up_proj into a single packed linear. This reduces the
number of kernel launches per layer from 5 to 3, improving GPU utilization.

Usage (post-load, before compile):
    from openpi.models_pytorch.fusion_utils import fuse_qkv_projections, fuse_gate_up_projections
    fuse_qkv_projections(model.paligemma_with_expert.paligemma.language_model)
    fuse_qkv_projections(model.paligemma_with_expert.gemma_expert.model)
    fuse_gate_up_projections(model.paligemma_with_expert.paligemma.language_model)
    fuse_gate_up_projections(model.paligemma_with_expert.gemma_expert.model)
"""

import torch
from torch import nn


def fuse_qkv_projections(model):
    """OPT-7: Pack separate Q/K/V nn.Linear into a single fused linear.

    Concatenates q_proj, k_proj, v_proj weights into a single qkv_proj.
    The fused weight has shape [q_out + k_out + v_out, hidden_size].
    During forward, the output is split back into Q, K, V by slicing.

    This reduces 3 GEMM kernel launches to 1 per attention layer.
    Weight values are exactly preserved — no approximation.

    Args:
        model: A GemmaModel (or language_model) with .layers[i].self_attn.{q,k,v}_proj
    """
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.self_attn

        # Skip if already fused
        if hasattr(attn, '_qkv_fused') and attn._qkv_fused:
            continue

        q_weight = attn.q_proj.weight.data  # [q_out, hidden]
        k_weight = attn.k_proj.weight.data  # [k_out, hidden]
        v_weight = attn.v_proj.weight.data  # [v_out, hidden]

        # OPT-7: Pack Q/K/V into single [q_out+k_out+v_out, hidden] weight
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        # Store the split sizes for unpacking during forward
        attn._qkv_split_sizes = [q_weight.shape[0], k_weight.shape[0], v_weight.shape[0]]

        # Create fused linear (no bias since Gemma attention uses bias=False)
        has_bias = attn.q_proj.bias is not None
        attn.qkv_proj = nn.Linear(
            qkv_weight.shape[1], qkv_weight.shape[0],
            bias=has_bias, device=qkv_weight.device, dtype=qkv_weight.dtype,
        )
        attn.qkv_proj.weight.data = qkv_weight

        if has_bias:
            qkv_bias = torch.cat([attn.q_proj.bias.data, attn.k_proj.bias.data, attn.v_proj.bias.data])
            attn.qkv_proj.bias.data = qkv_bias

        # Remove original projections to free memory
        del attn.q_proj, attn.k_proj, attn.v_proj
        attn._qkv_fused = True


def fuse_gate_up_projections(model):
    """OPT-7: Pack gate_proj + up_proj into a single fused linear.

    Concatenates gate_proj and up_proj weights into gate_up_proj.
    During forward, the output is split and the gate half is activated
    then multiplied with the up half, preserving the GLU structure.

    This reduces 2 GEMM kernel launches to 1 per MLP layer.
    Weight values are exactly preserved — no approximation.

    Args:
        model: A GemmaModel (or language_model) with .layers[i].mlp.{gate,up}_proj
    """
    for layer_idx, layer in enumerate(model.layers):
        mlp = layer.mlp

        # Skip if already fused
        if hasattr(mlp, '_gate_up_fused') and mlp._gate_up_fused:
            continue

        gate_weight = mlp.gate_proj.weight.data  # [intermediate, hidden]
        up_weight = mlp.up_proj.weight.data      # [intermediate, hidden]

        # OPT-7: Pack gate + up into single [2*intermediate, hidden] weight
        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

        # Store split size for unpacking
        mlp._gate_up_split_size = gate_weight.shape[0]

        # Create fused linear (no bias since Gemma MLP uses bias=False)
        mlp.gate_up_proj = nn.Linear(
            gate_up_weight.shape[1], gate_up_weight.shape[0],
            bias=False, device=gate_up_weight.device, dtype=gate_up_weight.dtype,
        )
        mlp.gate_up_proj.weight.data = gate_up_weight

        # Remove original projections to free memory
        del mlp.gate_proj, mlp.up_proj
        mlp._gate_up_fused = True


def unfuse_qkv_projections(model):
    """Reverse OPT-7 QKV fusion — restore separate Q/K/V projections.

    Useful for checkpoint saving or when you need to inspect individual projections.
    """
    for layer in model.layers:
        attn = layer.self_attn
        if not (hasattr(attn, '_qkv_fused') and attn._qkv_fused):
            continue

        q_out, k_out, v_out = attn._qkv_split_sizes
        q_weight, k_weight, v_weight = attn.qkv_proj.weight.data.split([q_out, k_out, v_out], dim=0)
        hidden = q_weight.shape[1]

        has_bias = attn.qkv_proj.bias is not None

        attn.q_proj = nn.Linear(hidden, q_out, bias=has_bias, device=q_weight.device, dtype=q_weight.dtype)
        attn.k_proj = nn.Linear(hidden, k_out, bias=has_bias, device=k_weight.device, dtype=k_weight.dtype)
        attn.v_proj = nn.Linear(hidden, v_out, bias=has_bias, device=v_weight.device, dtype=v_weight.dtype)

        attn.q_proj.weight.data = q_weight
        attn.k_proj.weight.data = k_weight
        attn.v_proj.weight.data = v_weight

        if has_bias:
            q_bias, k_bias, v_bias = attn.qkv_proj.bias.data.split([q_out, k_out, v_out])
            attn.q_proj.bias.data = q_bias
            attn.k_proj.bias.data = k_bias
            attn.v_proj.bias.data = v_bias

        del attn.qkv_proj, attn._qkv_split_sizes
        attn._qkv_fused = False


def unfuse_gate_up_projections(model):
    """Reverse OPT-7 gate+up fusion — restore separate gate_proj and up_proj."""
    for layer in model.layers:
        mlp = layer.mlp
        if not (hasattr(mlp, '_gate_up_fused') and mlp._gate_up_fused):
            continue

        split = mlp._gate_up_split_size
        gate_weight, up_weight = mlp.gate_up_proj.weight.data.split([split, split], dim=0)
        hidden = gate_weight.shape[1]

        mlp.gate_proj = nn.Linear(hidden, split, bias=False, device=gate_weight.device, dtype=gate_weight.dtype)
        mlp.up_proj = nn.Linear(hidden, split, bias=False, device=up_weight.device, dtype=up_weight.dtype)
        mlp.gate_proj.weight.data = gate_weight
        mlp.up_proj.weight.data = up_weight

        del mlp.gate_up_proj, mlp._gate_up_split_size
        mlp._gate_up_fused = False

'''
Helpers for applying Rotary Position Embeddings.
cos/sin are computed once by RoPE and then injected into Q/K.
'''

import torch


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to query and key tensors.

    Args:
        q, k: (batch, heads, seq, head_dim) by default
        cos, sin: (batch, seq, rotary_dim); rotary_dim <= head_dim
        unsqueeze_dim: where to insert the heads dimension so cos/sin broadcast.
                       1 for (batch, heads, seq, head_dim).
    Returns:
        rotated q, k with original dtypes preserved

    Partial rotary (Qwen3.5): when cos/sin carry fewer dims than head_dim,
    only the first rotary_dim coordinates are rotated, the rest pass through.
    With rotary_dim == head_dim the pass-through slices are empty and this
    reduces to the original full rotation.
    """
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed.to(q_type), k_embed.to(k_type)

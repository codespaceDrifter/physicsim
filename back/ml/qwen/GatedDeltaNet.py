'''
GATED DELTANET - Qwen3.5 linear-attention layer

Linear attention with a gated delta rule: instead of softmax over the full
sequence, each layer keeps a fixed-size recurrent state
    S: (num_v_heads, head_k_dim, head_v_dim)
updated per token by a delta rule with per-head decay:
    S_t = S_{t-1} * exp(g_t) + k_t (v_t - S_{t-1} k_t)^T beta_t
    o_t = S_t^T q_t
so decode is O(1) per token instead of O(seq_len).

Block layout (matching the HF Qwen3.5 checkpoint names):
- in_proj_qkv -> causal depthwise Conv1d (silu) -> split into q, k, v
- in_proj_b / in_proj_a produce beta (write strength) and the decay gate g
- q/k are l2-normalized; the gated delta rule runs chunked (prefill) or
  recurrent (single-token decode)
- output is gated RMSNorm against z (in_proj_z), then out_proj

Two implementations of the gated delta rule, picked per layer by `fast`:
- fast=False: the torch kernels below, ported from the HF reference (python loops over the chunk, fp32). runs anywhere
  and is traceable by torch.export (interpviz), but launches hundreds of tiny gpu ops per layer.
- fast=True: flash-linear-attention's fused triton kernels (chunk_gated_delta_rule / fused_recurrent_gated_delta_rule,
  what HF's Qwen3.5 uses when fla is installed): same math, gpu only, for training and bulk inference.

Cache protocol: cache_params must expose
    conv_states[layer_idx], recurrent_states[layer_idx],
    has_previous_state(layer_idx),
    update_conv_state(state, layer_idx), update_recurrent_state(state, layer_idx)
(see Qwen3_5Cache in transformer/qwen3_5.py).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from ml.qwen.RMSnorm import RMSNorm


def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Single-token causal Conv1d update used during decode.

    hidden_states: (batch, hidden_size, seq_len)  -- usually seq_len == 1
    conv_state:    (batch, hidden_size, state_len)
    weight:        (hidden_size, kernel_size)
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
    )
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x, dim=-1, eps=1e-6):
    # L2 normalization matching the FLA reference implementation
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Chunk-parallel Gated Delta Rule (torch fallback).

    Inputs are expected in (batch, seq_len, num_heads, head_dim) layout; they
    are transposed to (batch, num_heads, seq_len, head_dim) internally.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # (batch, heads, num_chunks, chunk_size, head_dim)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Intra-chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # Cross-chunk recurrent state
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    # back to (batch, seq_len, num_heads, head_v_dim)
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Sequential Gated Delta Rule used for single-token decode."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_mask_to_padding_states(hidden_states, attention_mask):
    # zero out hidden states for padding tokens (used by linear attention)
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention layer (Qwen3.5 style).

    Module names match the HF checkpoint layout under
    model.language_model.layers.{i}.linear_attn.*
    """

    def __init__(
        self,
        hidden_size,
        num_v_heads,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        layer_idx,
        rms_norm_eps=1e-6,
        fast=False,
    ):
        super().__init__()
        # fla's fused triton kernels instead of the torch reference (see the module docstring)
        self.fast = fast
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Time-step / decay parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNorm(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        # hidden_states: (batch, seq_len, hidden_size)
        # attention_mask: optional (batch, seq_len) padding mask, 1 = real token
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        )

        if use_precomputed_states:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch, conv_dim, seq_len)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # (batch, seq, num_v_heads, head_v_dim)

        b = self.in_proj_b(hidden_states)  # (batch, seq, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (batch, seq, num_v_heads)

        if use_precomputed_states and seq_len == 1:
            # Single-token decode path
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation="silu",
            )
        else:
            # Prefill or chunked-decode path
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)

            if cache_params is not None:
                # Keep the last kernel_size-1 inputs as the conv context for the next step.
                keep = self.conv_kernel_size - 1
                if mixed_qkv.shape[-1] >= keep:
                    new_conv_state = mixed_qkv[:, :, -keep:].contiguous()
                else:
                    new_conv_state = F.pad(
                        mixed_qkv, (keep - mixed_qkv.shape[-1], 0)
                    )
                cache_params.update_conv_state(new_conv_state, self.layer_idx)

            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])

            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch, seq_len, conv_dim)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # (batch, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # `g` is the negative decay rate used by the gated delta rule.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            recurrent_fn = fused_recurrent_gated_delta_rule if self.fast else torch_recurrent_gated_delta_rule
            core_attn_out, last_recurrent_state = recurrent_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            chunk_fn = chunk_gated_delta_rule if self.fast else torch_chunk_gated_delta_rule
            core_attn_out, last_recurrent_state = chunk_fn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # per-head RMSNorm gated by z (silu valve on the way out)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out) * F.silu(z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)  # (batch, seq_len, hidden_size)
        return output

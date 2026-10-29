'''
FLASH ATTENTION WITH GQA - COMPLETE EXAMPLE WALKTHROUGH

================================================================================
PART 1: GQA SETUP
================================================================================

Input: x of shape (seq_len, embed_dim) = (4, 8)
Config: num_q_heads=4, num_kv_heads=2, head_dim=2

Weight shapes:
    W_q: (8, 8)   # embed_dim -> num_q_heads * head_dim
    W_k: (8, 4)   # embed_dim -> num_kv_heads * head_dim
    W_v: (8, 4)   # embed_dim -> num_kv_heads * head_dim

After projection and reshape to (num_heads, seq, head_dim):
    Q: (4, 4, 2)   # num_q_heads, seq, head_dim
    K: (2, 4, 2)   # num_kv_heads, seq, head_dim
    V: (2, 4, 2)   # num_kv_heads, seq, head_dim

GQA expansion - repeat KV heads to match Q heads:
    K: (2, 4, 2) -> repeat_interleave(2, dim=0) -> (4, 4, 2)
    V: (2, 4, 2) -> repeat_interleave(2, dim=0) -> (4, 4, 2)

================================================================================
PART 2: FLASH ATTENTION ALGORITHM
================================================================================

For one head, chunk into seq_len into blocks of block_len (block_len=2):
    Q_blocks: [Q[0:2], Q[2:4]]   # two (2, 2) chunks
    K_blocks: [K[0:2], K[2:4]]
    V_blocks: [V[0:4]]

SRAM (persists through KV loop):
    Q_block: (2, 2)   # block_len, head_dim
    m:       (2,)     # running max per query, init -inf
    l:       (2,)     # running sum per query, init 0
    o:       (2, 2)   # running output, init 0

HBM -> SRAM each iteration (then discarded) we iterate over each block across the sequence:
    K_block: (2, 2)
    V_block: (2, 2)
    S:       (2, 2)   # attention scores

THE LOOP:

for kv_idx in range(num_kv_blocks):
    K_blk, V_blk = load from HBM                    # (2, 2) each
    S = Q_block @ K_blk.T / sqrt(head_dim)          # (2, 2)
    m_new = max(m, S.max(dim=-1))                   # (2,)
    rescale = exp(m - m_new)                        # (2,)
    l = l * rescale                                 # (2,)
    o = o * rescale[:, None]                        # (2, 2)
    weights = exp(S - m_new[:, None])               # (2, 2)
    l = l + weights.sum(dim=-1)                     # (2,)
    o = o + weights @ V_blk                         # (2, 2)
    m = m_new

output = o / l[:, None]   # (2, 2) - final normalization

================================================================================
PART 3: WHY RESCALING WORKS
================================================================================

At any point:
o = sum_j exp(s_j - m) * v_j
l = sum_j exp(s_j - m)

When max changes from m_old to m_new:
    exp(s_j - m_new) = exp(s_j - m_old) * exp(m_old - m_new)

So multiply o and l by exp(m_old - m_new) to convert to new max.
Division by l happens once at the end.

================================================================================
PART 4: OUTPUT
================================================================================

After all heads complete:
    out: (num_heads, seq, head_dim) -> transpose -> (seq, num_heads, head_dim)
    out: reshape -> (seq, embed_dim)
    out = out @ W_o   # final projection, mixes heads


HF-compatible defaults:
- Separate Q, K, V projections (no fused qkv_proj) so HF weight names map 1:1.
- Optional q_norm / k_norm after projection (controlled by qk_norm param).
- Optional flash/SDPA path vs manual eager attention (controlled by use_flash param).
- position_embeddings come from outside (cos, sin) to support KV-cache decoding.

Qwen3.5 tweaks (all off by default):
- query_gate: q_proj outputs 2x the query dim; per head the output is split
  into (query, gate) and the attention output is multiplied by sigmoid(gate)
  before o_proj.
- qk_norm_per_head: q_norm / k_norm are per-head RMSNorm over head_dim
  (applied after the heads reshape) instead of over the flattened q_out/kv_out.
- norm_centered: use the zero-centered Qwen3.5 RMSNorm style for the qk norms.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.qwen.rope_utils import apply_rotary_pos_emb


class GQA(nn.Module):
    _printed_backend = False  # class var to print only once

    def __init__(
        self,
        embed_dim,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        attention_bias=False,
        attention_dropout=0.0,
        qk_norm=False,
        rms_norm_eps=1e-6,
        use_flash=False,
        query_gate=False,
        qk_norm_per_head=False,
        norm_centered=False,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_q_heads // num_kv_heads
        self.attention_dropout = attention_dropout
        self.qk_norm = qk_norm
        self.use_flash = use_flash
        self.query_gate = query_gate
        self.qk_norm_per_head = qk_norm_per_head

        q_out = num_q_heads * head_dim
        kv_out = num_kv_heads * head_dim

        # query_gate doubles the q projection: per head it emits (query | gate)
        self.q_proj = nn.Linear(embed_dim, q_out * 2 if query_gate else q_out, bias=attention_bias)
        self.k_proj = nn.Linear(embed_dim, kv_out, bias=attention_bias)
        self.v_proj = nn.Linear(embed_dim, kv_out, bias=attention_bias)
        self.o_proj = nn.Linear(q_out, embed_dim, bias=attention_bias)

        if self.qk_norm:
            from ml.qwen.RMSnorm import RMSNorm
            q_norm_dim = head_dim if qk_norm_per_head else q_out
            k_norm_dim = head_dim if qk_norm_per_head else kv_out
            self.q_norm = RMSNorm(q_norm_dim, eps=rms_norm_eps, centered=norm_centered)
            self.k_norm = RMSNorm(k_norm_dim, eps=rms_norm_eps, centered=norm_centered)

    def forward(self, x, position_embeddings, attention_mask=None, past_key_value=None):
        """
        x: (batch, seq, embed_dim)
        position_embeddings: (cos, sin) each (batch, seq, head_dim)
        attention_mask: optional (batch, 1, seq, total_kv_len) additive mask
        past_key_value: optional tuple (past_k, past_v) for incremental decoding
        returns: (attn_output, (k, v)) where k,v are the *full* keys/values
        """
        batch, seq, _ = x.shape

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        # qwen3.5 query gate: split each head's 2*head_dim into (query | gate)
        gate = None
        if self.query_gate:
            query_states = query_states.view(batch, seq, self.num_q_heads, self.head_dim * 2)
            query_states, gate = query_states.chunk(2, dim=-1)  # each (batch, seq, num_q_heads, head_dim)
            gate = gate.reshape(batch, seq, -1)  # (batch, seq, num_q_heads * head_dim)

        if self.qk_norm and not self.qk_norm_per_head:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # reshape to (batch, seq, num_heads, head_dim)
        query_states = query_states.view(batch, seq, self.num_q_heads, self.head_dim)
        key_states = key_states.view(batch, seq, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch, seq, self.num_kv_heads, self.head_dim)

        # qwen3.5 per-head qk norm: RMSNorm over head_dim
        if self.qk_norm and self.qk_norm_per_head:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # (batch, num_heads, seq, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV-cache concat
        if past_key_value is not None:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        current_kv = (key_states, value_states)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
                enable_gqa=self.num_kv_groups > 1,
            )
            if not GQA._printed_backend:
                GQA._printed_backend = True
                print(
                    f"SDPA backends - Flash: {torch.backends.cuda.flash_sdp_enabled()}, "
                    f"MemEfficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
                    f"Math: {torch.backends.cuda.math_sdp_enabled()}"
                )
        else:
            # manual eager attention (your own normal attention code)
            key_states_eager = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states_eager = value_states.repeat_interleave(self.num_kv_groups, dim=1)

            attn_weights = torch.matmul(
                query_states, key_states_eager.transpose(2, 3)
            ) * (self.head_dim ** -0.5)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                # causal mask
                causal_mask = torch.triu(
                    torch.full((seq, key_states_eager.size(2)), float('-inf'), device=x.device),
                    diagonal=1,
                )
                attn_weights = attn_weights + causal_mask[None, None, :, :]

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
            out = torch.matmul(attn_weights, value_states_eager)

        # (batch, seq, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        # qwen3.5 query gate: scale the attention output by sigmoid(gate)
        if self.query_gate:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out), current_kv

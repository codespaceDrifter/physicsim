'''
Qwen3.5 decoder-only hybrid causal LM (text branch).

Alternates Gated DeltaNet linear-attention layers with gated full-attention
layers according to config.layer_types (4B layout: 3 linear + 1 full, x8).

Built from the generic primitives in synapse/algorithms:
- basic.RMSnorm.RMSNorm            (centered=True, the Qwen3.5 zero-init style)
- basic.gatedMLP.GatedMLP
- encoding.RoPE.RoPE               (partial_rotary_factor < 1)
- encoding.mask.create_causal_mask
- attention.GQA.GQA                (query_gate=True, per-head centered qk norms)
- attention.GatedDeltaNet.GatedDeltaNet

Weight key layout matches the official HF Qwen3.5 text checkpoint so loading
is a straight state_dict copy:
    model.language_model.embed_tokens.weight
    model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight
    model.language_model.layers.{i}.self_attn.q_proj.weight
    model.language_model.layers.{i}.mlp.gate_proj.weight
    model.language_model.norm.weight
    lm_head.weight (tied to embed_tokens)

Qwen3.5's MRoPE interleaving of the 3 spatial grids is a no-op for text, so
position_ids stay 2D here and the interleaving is intentionally not ported.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.qwen.RMSnorm import RMSNorm
from ml.qwen.gatedMLP import GatedMLP
from ml.qwen.RoPE import RoPE
from ml.qwen.mask import create_causal_mask
from ml.qwen.GQA import GQA
from ml.qwen.GatedDeltaNet import GatedDeltaNet


class Qwen3_5Cache:
    # minimal hybrid cache for generation:
    # KV tensors for full-attention layers, conv/recurrent states for linear layers
    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers
        self.key_cache = [None] * num_hidden_layers
        self.value_cache = [None] * num_hidden_layers
        self.conv_states = [None] * num_hidden_layers
        self.recurrent_states = [None] * num_hidden_layers
        self._seen_tokens = 0

    # --- full attention KV (adapts GQA's (k, v) tuple interface) ---
    def get_kv(self, layer_idx):
        # returns (k, v) each (batch, num_kv_heads, past_len, head_dim), or None
        if self.key_cache[layer_idx] is None:
            return None
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def set_kv(self, kv, layer_idx):
        self.key_cache[layer_idx], self.value_cache[layer_idx] = kv

    # --- linear attention states ---
    def update_conv_state(self, conv_state, layer_idx):
        # conv_state: (batch, conv_dim, conv_kernel_size - 1)
        self.conv_states[layer_idx] = conv_state

    def update_recurrent_state(self, recurrent_state, layer_idx):
        # recurrent_state: (batch, num_v_heads, head_k_dim, head_v_dim)
        self.recurrent_states[layer_idx] = recurrent_state

    # --- sequence length bookkeeping ---
    def get_seq_length(self, layer_idx=0):
        # linear-attention layers do not store per-token KV tensors,
        # so we maintain an explicit counter
        if self._seen_tokens:
            return self._seen_tokens
        if self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0

    def set_seq_length(self, seq_length):
        self._seen_tokens = seq_length

    def has_previous_state(self, layer_idx=None):
        if layer_idx is not None:
            return (
                self.conv_states[layer_idx] is not None
                or self.recurrent_states[layer_idx] is not None
                or self.key_cache[layer_idx] is not None
            )
        return any(s is not None for s in self.key_cache + self.conv_states + self.recurrent_states)


class Qwen3_5DecoderLayer(nn.Module):
    # switches between linear attention and full attention per layer_types
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_v_heads=config.linear_num_value_heads,
                num_k_heads=config.linear_num_key_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
                fast=config.use_fla,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = GQA(
                embed_dim=config.hidden_size,
                num_q_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_seq_len=config.max_position_embeddings,
                attention_bias=config.attention_bias,
                attention_dropout=config.attention_dropout,
                qk_norm=True,
                rms_norm_eps=config.rms_norm_eps,
                use_flash=config.use_flash_attention,
                query_gate=True,
                qk_norm_per_head=True,
                norm_centered=True,
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            separate_gate_up_proj=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, centered=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, centered=True)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        # hidden_states: (batch, seq, hidden_size)
        # attention_mask: 4D additive causal mask for full-attention layers,
        #                 2D padding mask (or None) for linear-attention layers
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            past_kv = past_key_values.get_kv(self.layer_idx) if past_key_values is not None else None
            hidden_states, present_kv = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_kv,
            )
            if past_key_values is not None:
                past_key_values.set_kv(present_kv, self.layer_idx)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, centered=True)
        self.rotary_emb = RoPE(
            config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
        )

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        # linear attention uses a simple left-padding mask, or None when safe:
        # cached decode re-reads no padded tokens, and an all-ones mask is a no-op
        if past_key_values is not None and past_key_values.has_previous_state():
            return None
        if attention_mask is not None and torch.all(attention_mask == 1):
            return None
        return attention_mask

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, use_cache=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5Cache(self.config.num_hidden_layers)

        batch, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        causal_mask = create_causal_mask(
            inputs_embeds,
            attention_mask=attention_mask,
            past_key_values_length=past_length,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            layer_mask = (
                linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)

        if use_cache and past_key_values is not None:
            past_key_values.set_seq_length(past_length + seq_len)

        return hidden_states, past_key_values


class Qwen3_5Model(nn.Module):
    # the model.language_model wrapper is kept so the state_dict key layout
    # matches the official HF Qwen3.5 text checkpoint
    def __init__(self, config):
        super().__init__()
        self.language_model = Qwen3_5TextModel(config)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
    ):
        hidden_states, past_key_values = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        # only compute logits for the tokens we actually need (decode: last token)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # (batch, seq, vocab_size)

        loss = None
        if labels is not None:
            # shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss, past_key_values

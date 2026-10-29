"""
Config for Qwen/Qwen3.5-4B (text branch only).

This is the concrete 4B checkpoint text config, pulled from the cached HF
weights' config.json. Keep model-specific hyperparameters here so the generic
modules in synapse/algorithms stay reusable.

Architecture: 8 blocks of [3 linear-attention (Gated DeltaNet) layers
+ 1 full-attention layer]. The vision tower is not implemented.
"""

from dataclasses import dataclass


@dataclass
class Qwen3_5Config:
    """Configuration for the Qwen3.5 4B text model."""

    # tokenization
    vocab_size: int = 248320
    eos_token_id: int = 248044

    # model size
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 9216

    # activation / norm
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # RoPE (partial rotary: only 64 of 256 head dims are rotated)
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    # mrope_section [11, 11, 10] from the checkpoint is vision-only;
    # for text the MRoPE interleaving is a no-op and is not implemented

    # attention
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # layout: every 4th layer is full attention; the rest are linear attention
    full_attention_interval: int = 4

    # inference
    use_cache: bool = True
    use_flash_attention: bool = False
    # gated deltanet layers on flash-linear-attention's fused triton kernels (gpu only, not traceable by torch.export)
    # instead of the pure-torch reference loops; load_qwen3_5_model(fast=True) turns this and use_flash_attention on
    use_fla: bool = False

    # lm head
    tie_word_embeddings: bool = True

    # checkpoint dtype stored on disk
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        # derived layer layout: 8 x [linear, linear, linear, full]
        self.layer_types = [
            "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads


# shortcut for the 4B checkpoint
QWEN3_5_4B_CONFIG = Qwen3_5Config()

# load Linears as int8 via bitsandbytes (in memory only, the safetensors on disk are untouched).
# off: everything runs the exact bf16 model on weighty (5090, 32GB).
QUANTIZE_INT8 = False

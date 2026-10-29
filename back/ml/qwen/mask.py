'''
Causal mask construction for decoder-only LMs.
Supports optional padding masks and KV-cache offsets so generation works.

Returns an additive mask: 0 means "attend", -inf means "ignore".
Shape: (batch, 1, query_len, key_len) so it broadcasts over heads.
'''

import torch


def create_causal_mask(
    inputs_embeds,
    attention_mask=None,
    past_key_values_length=0,
    position_ids=None,
):
    """
    Build a causal attention mask (with optional padding and kv-cache offset) for prefill and generation.

    inputs_embeds: (batch, seq, hidden_size) -- used for device/dtype
    attention_mask: optional (batch, seq) bool/int, 1 for real tokens, 0 for pad.
    past_key_values_length: int, length of already cached KV states.
    position_ids: optional (batch, seq) positions; if None, inferred from cache.

    returns: additive mask of shape (batch, 1, seq, seq + past_len)
    """
    batch, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    total_len = seq_len + past_key_values_length

    # base causal mask
    causal_mask = torch.full((seq_len, total_len), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)

    # expand to (batch, 1, seq, total_len)
    causal_mask = causal_mask[None, None, :, :].expand(batch, 1, -1, -1).to(dtype)

    if attention_mask is not None:
        # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
        # tokens with mask=0 make *their own* queries unable to attend anywhere
        mask_4d = attention_mask[:, None, None, :].to(dtype)
        # invert: 0 (pad) -> -inf, 1 (real) -> 0
        # use where/masked_fill so 1.0 * -inf does not become NaN
        padding_mask = torch.where(mask_4d == 1.0, 0.0, float('-inf'))
        causal_mask = causal_mask + padding_mask

    return causal_mask

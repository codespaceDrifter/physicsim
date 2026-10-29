'''
RMSnorm scales by sqrt(n) and then scales by weight per coordinate.
RMS is applied at the token level across the embed_dim. the total would be sqrt(n)*gamma rather than normal norm would be 1.
without gamma the mean(x_norm^2) = 1
RMS(x) = sqrt (1/n * sum(x^2)) * gamma

default: cast to float32 for numerical stability, multiply by weight,
then cast back to the input dtype.

Qwen3.5 tweak: centered=True switches to the zero-centered variant
    out = x_norm * (1 + weight),   weight initialized to zeros.
This is the Qwen3.5 RMSNorm style; the checkpoint weight maps 1:1.
'''

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, cast_to_float32=True, centered=False):
        super().__init__()
        self.eps = eps
        self.cast_to_float32 = cast_to_float32
        self.centered = centered
        init = torch.zeros(dim) if centered else torch.ones(dim)
        self.weight = nn.Parameter(init)

    def forward(self, x):
        # x: (batch, seq, dim)
        input_dtype = x.dtype
        if self.cast_to_float32 and x.dtype != torch.float32:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True) # not technically variance
        x_norm = x * torch.rsqrt(variance + self.eps) #reciprocal square root

        # default: multiply weight before dtype conversion
        # Qwen3.5 centers the weight around 1 instead
        if self.centered:
            out = x_norm * (1.0 + self.weight)
        else:
            out = self.weight * x_norm
        return out.to(input_dtype)


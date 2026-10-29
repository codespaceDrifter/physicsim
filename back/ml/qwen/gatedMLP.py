'''
gated mlp has double the parameters in the first half. one gate and one up. chunked together for parallelism.
first half function:
hidden = sigmoid(gate) * gate * up
think of it like the gate part deciding IF and the up part deciding WHAT. they can target DIFFERENT PARTS OF THE ACTIVATION.
for example, say a two dimensional vector, i want to pass coordinate_2 through IF coordinate_1 is positive:
in a traditional single no gate MLP this would be impossible:
[0,1] : misses coordinate_1 conditional. [1,0]: misses coordinate_2 value. [1,1]: mixes values of coordinate_1 and coordinate_2.
but in swiglu:
gate: [1 , 0]. up : [0,1]. this passes through value of coordinate_2 IF coordinate_1 is positive.

it is sigmoid(gate) * gate * up rather than sigmoid (gate) * up due to the potential for gate to amplify the up.

HF-compatible defaults:
- separate gate_proj / up_proj by default so HF weight names map directly.
- optional fused gate_up_proj if you want fewer params (set separate_gate_up_proj=False).
- bias is configurable.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size=3072,
        intermediate_size=8192,
        bias=False,
        separate_gate_up_proj=True,
    ):
        super().__init__()
        self.separate_gate_up_proj = separate_gate_up_proj

        if separate_gate_up_proj:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        # x: (batch, seq, hidden_size)
        if self.separate_gate_up_proj:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        else:
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)

        # SiLU(gate) * up - this is the gating
        hidden = F.silu(gate) * up
        out = self.down_proj(hidden)
        return out

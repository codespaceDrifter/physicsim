# Teacher: the course from privileged simulator state (CourseEnv.state(), 62 numbers per robot), trained by PPO
# (ml/train_teacher.py). an actor (gaussian policy) and a separate critic behind a running input normalizer.
#
# the blocks are a set, not slots: each block is a token (its 5 state numbers, "in my home zone" / "in the other home
# zone" flags, its offset from the gripper), encoded together with the rest of the state, then pooled by attention
# (query from the robot's state) and max over blocks. a fixed sorted-slot MLP had to work out from raw coordinates
# which slot was still worth fetching, and with blocks already home it mostly didn't.
#   state -> robot part (everything but the blocks and the time left, 31) -> context (WIDTH)
#   per block: [block (5), in own home, in other home, block - grip (3), context] -> token (WIDTH)
#   [context, attention pool, max pool] (3 WIDTH) -> HIDDEN x LAYERS -> out
# the actor is residual: its mean action = the action that holds every joint where it is now (read off proprio, which
# is normalized like the actions) + the network's output. arm commands are small offsets from the current pose; as
# absolute targets the network could not reproduce the current pose precisely enough and the arm stalled
import torch
import torch.nn as nn
from robot.course import STATE_DIM, NUM_BLOCKS, ZONE_SIZE
from robot.env import ACTION_DIM, HOLD

HIDDEN = 512
LAYERS = 3
WIDTH = 128
HEADS = 4
# where the grip point (forward, right, height), the blocks (NUM_BLOCKS x (forward, right, height, sin 4θ, cos 4θ))
# and the homes (own forward, right, other forward, right) sit in the state vector
GRIP_AT = 9 + 3
BLOCKS_AT = GRIP_AT + 3
HOMES_AT = BLOCKS_AT + NUM_BLOCKS * 5
# per-block token input: 5 block numbers + 2 home flags + 3 offset from the grip point
TOKEN_IN = 5 + 2 + 3
# robot part fed to the context: everything but the blocks and the last entry (time left in the episode). the
# network does not see time: the expert ignores it and always finished long before the end, so its clone had never
# seen a late clock and fell apart once one showed up
ROBOT_DIM = STATE_DIM - NUM_BLOCKS * 5 - 1


class Net(nn.Module):
    # normalized state (..., STATE_DIM) + raw state (for the home flags) -> (..., out_dim)
    def __init__(self, out_dim):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(ROBOT_DIM, WIDTH), nn.ELU(), nn.Linear(WIDTH, WIDTH), nn.ELU())
        self.token = nn.Sequential(nn.Linear(TOKEN_IN + WIDTH, WIDTH), nn.ELU(), nn.Linear(WIDTH, WIDTH), nn.ELU())
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        dims = [3 * WIDTH] + [HIDDEN] * LAYERS
        layers = [m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.ELU())]  # [nn.Module]
        self.trunk = nn.Sequential(*layers, nn.Linear(HIDDEN, out_dim))

    def forward(self, x, raw):
        lead = x.shape[:-1]
        x, raw = x.reshape(-1, STATE_DIM), raw.reshape(-1, STATE_DIM)
        # (N, ROBOT_DIM) -> (N, WIDTH)
        ctx = self.context(torch.cat([x[:, :BLOCKS_AT], x[:, HOMES_AT:-1]], -1))
        # (N, B, 5) normalized blocks; raw positions for the flags and the grip offset
        blocks = x[:, BLOCKS_AT:HOMES_AT].unflatten(-1, (NUM_BLOCKS, 5))
        raw_blocks = raw[:, BLOCKS_AT:HOMES_AT].unflatten(-1, (NUM_BLOCKS, 5))
        own, other = raw[:, HOMES_AT:HOMES_AT + 2], raw[:, HOMES_AT + 2:HOMES_AT + 4]
        # (N, B, 1) each
        in_own = (torch.linalg.norm(raw_blocks[..., :2] - own[:, None], dim=-1, keepdim=True) < ZONE_SIZE / 2).float()
        in_other = (torch.linalg.norm(raw_blocks[..., :2] - other[:, None], dim=-1, keepdim=True) < ZONE_SIZE / 2).float()
        # (N, B, 3)
        to_grip = raw_blocks[..., :3] - raw[:, None, GRIP_AT:GRIP_AT + 3]
        # (N, B, TOKEN_IN + WIDTH) -> (N, B, WIDTH)
        tokens = self.token(torch.cat([blocks, in_own, in_other, to_grip, ctx[:, None].expand(-1, NUM_BLOCKS, -1)], -1))
        # (N, 1, WIDTH) attention pool with the context as the query, (N, WIDTH) max pool
        pooled, _ = self.attn(ctx[:, None], tokens, tokens, need_weights=False)
        out = self.trunk(torch.cat([ctx, pooled[:, 0], tokens.max(1).values], -1))
        return out.reshape(*lead, -1)


class Teacher(nn.Module):
    def __init__(self, init_log_std=-0.7):
        super().__init__()
        self.actor = Net(ACTION_DIM)
        self.critic = Net(1)
        # state-independent log std of the gaussian, one per action dim
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), init_log_std))
        # running mean / variance of the state (Welford over every state seen), count = samples so far
        self.register_buffer('mean', torch.zeros(STATE_DIM))
        self.register_buffer('var', torch.ones(STATE_DIM))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update_norm(self, states):
        # fold a batch of states (N, STATE_DIM) into the running mean / variance (parallel Welford)
        n = states.shape[0]
        b_mean, b_var = states.mean(0), states.var(0, unbiased=False)
        total = self.count + n
        delta = b_mean - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + b_var * n + delta ** 2 * self.count * n / total) / total)
        self.count.fill_(total)

    def forward(self, state):
        # state (..., STATE_DIM) -> mean action (..., ACTION_DIM) (the env clips actions to [-1, 1]), value (...)
        # every block normalized alike (pooled mean / variance over the block slots), so block order doesn't matter
        mean, var = self.mean.clone(), self.var.clone()
        # (B, 5) per slot -> (5,) pooled
        m_b, v_b = self.mean[BLOCKS_AT:HOMES_AT].view(NUM_BLOCKS, 5), self.var[BLOCKS_AT:HOMES_AT].view(NUM_BLOCKS, 5)
        pooled_mean = m_b.mean(0)
        mean[BLOCKS_AT:HOMES_AT] = pooled_mean.repeat(NUM_BLOCKS)
        var[BLOCKS_AT:HOMES_AT] = (v_b + (m_b - pooled_mean) ** 2).mean(0).repeat(NUM_BLOCKS)
        x = ((state - mean) / (var.sqrt() + 1e-8)).clamp(-5, 5)
        # (..., ACTION_DIM) action that holds the current joint positions and wheel speeds (proprio leads the state)
        hold = state[..., HOLD]
        return hold + self.actor(x, state), self.critic(x, state).squeeze(-1)

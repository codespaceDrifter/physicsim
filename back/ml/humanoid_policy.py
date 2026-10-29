# The humanoid model: ONE network that is both what gets deployed and what RL trains. single-stage asymmetric
# actor-critic with a state estimator inside the actor (Ji et al. 2022 "Concurrent training of a control policy and a
# state estimator"; Gu et al. 2024 "Denoising World Model Learning" for humanoids):
#   Policy  sees only what the real robot has: the last HISTORY steps of proprio (robot/humanoid.py observe()), the
#           command (words, or for the pure motor baseline one task vector: robot/mj_warehouse.py) and, with vision,
#           its head camera's latest frame. a transformer reads them as tokens: [act] + HISTORY /
#           GROUP proprio tokens (GROUP consecutive readings each) + command tokens + IMAGE_TOKENS image tokens. image tokens are always there: a learned
#           position embedding per patch, plus the frame's patch features from a conv stem in stages with a camera
#           (blind stages skip rendering). the stem's last layer starts at zero, so a model trained blind acts exactly
#           the same the moment the camera turns on, and learns to use it from there
#           its [act] output z feeds two heads:
#             action head   -> mean action (A), plus a learned per-joint log std (gaussian policy for PPO)
#             state decoder -> the world state it can't sense directly (base velocity, height, foot contacts /
#                              heights, the boxes, pedestals and platform; "aux"), supervised by the simulator's true
#                              values. z has to carry them, so the action head acts on an estimate of the true state
#   Critic  training only: an MLP on the privileged state (exact, noise-free), so its value estimates are sharp even
#           while the policy still has to infer that state from noisy sensor history
# every input goes through a running mean / std normalizer (updated by the trainer from rollout data)
import torch
import torch.nn as nn
from robot.g1 import CAMERA_RES

HISTORY = 16
# proprio readings per history token
GROUP = 4
# ~12M parameters: a 26M model (512 x 8) updates 3.5x slower than this on the A100 (28 s vs 8 s per iteration at
# 2.4M), this one 2.5x (20 s)
WIDTH = 384
LAYERS = 6
CRITIC_HIDDEN = [512, 256, 128]
# camera frames are CAMERA_RES x CAMERA_RES; the conv stem downsamples by 16 to a 4 x 4 grid of tokens
IMAGE_TOKENS = (CAMERA_RES // 16) ** 2


class Normalizer(nn.Module):
    # running mean / variance of a feature vector (Chan et al. parallel update), applied as (x - mean) / std
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        # x (..., dim) new samples
        x = x.reshape(-1, x.shape[-1])
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        total = self.count + n
        delta = m - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)

    @torch.no_grad()
    def forget_constant(self):
        # dimensions that never varied so far (an input no stage used yet) start their statistics over, so the first
        # real values aren't scaled by a near-zero variance
        constant = self.var < 1e-8
        self.mean[constant] = 0.0
        self.var[constant] = 1.0


class Policy(nn.Module):
    # proprio_dim, action_dim, aux_dim: the course's sizes. command: how the task comes in, either words (vocab > 0:
    # command_dim word ids, embedded, with a position each) or one task vector (vocab = 0: command_dim floats,
    # normalized and projected to one token). vision: the head camera's IMAGE_TOKENS (off: no image tokens at all).
    # stages: with a task vector, its first `stages` entries are the stage one-hot, and each stage gets its own action
    # noise (a new stage explores with large noise while a learned one keeps its small noise; 0: one noise for all).
    # width, layers, group: the size (defaults above; a checkpoint's trainer args record them). heads of 64 dims
    def __init__(self, proprio_dim, action_dim, aux_dim, command_dim, vocab, vision, stages=0, width=WIDTH, layers=LAYERS, group=GROUP):
        super().__init__()
        self.group, self.vocab, self.vision, self.stages = group, vocab, vision, stages
        self.norm = Normalizer(proprio_dim)
        self.proprio_in = nn.Linear(group * proprio_dim, width)
        # learned position embeddings of the [act] token, the history steps (oldest first) and the command words
        self.act_token = nn.Parameter(torch.zeros(1, 1, width))
        self.history_pos = nn.Parameter(torch.zeros(1, HISTORY // group, width))
        if vocab:
            self.word = nn.Embedding(vocab, width)
            self.word_pos = nn.Parameter(torch.zeros(1, command_dim, width))
            nn.init.normal_(self.word_pos, std=0.02)
        else:
            self.task_norm = Normalizer(command_dim)
            self.task_in = nn.Linear(command_dim, width)
        for p in (self.act_token, self.history_pos):
            nn.init.normal_(p, std=0.02)
        if vision:
            self.image_pos = nn.Parameter(torch.zeros(1, IMAGE_TOKENS, width))
            nn.init.normal_(self.image_pos, std=0.02)
            # (B, 3, 64, 64) -> (B, width, 4, 4): four stride-2 convs, the last one zero (see the header)
            self.image_stem = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                                            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Conv2d(128, width, 4, 2, 1))
            nn.init.zeros_(self.image_stem[-1].weight)
            nn.init.zeros_(self.image_stem[-1].bias)
        layer = nn.TransformerEncoderLayer(width, width // 64, 4 * width, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(width)
        self.action_head = nn.Linear(width, action_dim)
        # (stages or 1, A) log std of the gaussian policy per stage
        self.log_std = nn.Parameter(torch.zeros(max(stages, 1), action_dim))
        # the state decoder predicts the privileged quantities normalized by their own running statistics
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, aux_dim))
        self.aux_norm = Normalizer(aux_dim)
        # start near "stand still": small initial actions
        nn.init.normal_(self.action_head.weight, std=0.01)
        nn.init.zeros_(self.action_head.bias)

    def std(self, command):
        # command (B, command_dim) -> (B, A) action noise std (its stage's, or the shared one)
        if self.stages:
            return self.log_std[command[:, :self.stages].argmax(-1)].exp()
        return self.log_std[0].exp().expand(command.shape[0], -1)

    def forward(self, history, command, image=None):
        # history (B, HISTORY, proprio_dim) oldest first, command (B, command_dim) word ids or task vector, image (B, 3,
        # CAMERA_RES, CAMERA_RES) uint8 or None (blind stages) -> mean action (B, A), state estimate (B, aux_dim) in
        # aux_norm units
        B = history.shape[0]
        # (B, HISTORY, P) -> (B, HISTORY / group, group * P)
        grouped = self.norm(history).reshape(B, HISTORY // self.group, -1)
        # (B, command tokens, width)
        words = self.word(command) + self.word_pos if self.vocab else self.task_in(self.task_norm(command))[:, None]
        tokens = [self.act_token.expand(B, -1, -1), self.proprio_in(grouped) + self.history_pos, words]
        if self.vision:
            # (B, IMAGE_TOKENS, width): (B, 3, 64, 64) -> (B, width, 4, 4) patch features, or none without a camera
            patches = self.image_pos.expand(B, -1, -1)
            if image is not None:
                patches = patches + self.image_stem(image.float() / 255 - 0.5).flatten(2).transpose(1, 2)
            tokens.append(patches)
        # (B, tokens, width)
        x = torch.cat(tokens, 1)
        # (B, width) the [act] token's output. the transformer runs in bfloat16 (half the activation memory), heads in float32
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            h = self.encoder(x)[:, 0]
        z = self.out_norm(h.float())
        return self.action_head(z), self.decoder(z)


class Critic(nn.Module):
    # privileged state (B, priv_dim) -> value (B,)
    def __init__(self, priv_dim):
        super().__init__()
        self.norm = Normalizer(priv_dim)
        layers = []  # [nn.Module]
        for a, b in zip([priv_dim] + CRITIC_HIDDEN, CRITIC_HIDDEN):
            layers += [nn.Linear(a, b), nn.ELU()]
        self.net = nn.Sequential(*layers, nn.Linear(CRITIC_HIDDEN[-1], 1))

    def forward(self, priv):
        return self.net(self.norm(priv)).squeeze(-1)


def load_matching(net, saved):
    # load a checkpoint's state dict into net where it still fits: tensors of equal shape copy; the word embedding,
    # whose vocabulary only grows (append-only), copies the saved words' rows, and the word positions, whose commands
    # only grow at the end, the saved positions. returns [names of tensors left as
    # initialized]
    own = net.state_dict()
    fresh = []  # [str]
    for k, v in own.items():
        if k in saved and saved[k].shape == v.shape:
            v.copy_(saved[k])
        elif k == 'word.weight' and k in saved and saved[k].shape[1:] == v.shape[1:] and saved[k].shape[0] < v.shape[0]:
            v[:saved[k].shape[0]].copy_(saved[k])
        elif k == 'word_pos' and k in saved and saved[k].shape[2] == v.shape[2] and saved[k].shape[1] < v.shape[1]:
            v[:, :saved[k].shape[1]].copy_(saved[k])
        else:
            fresh.append(k)
    return fresh

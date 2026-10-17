# VLA v1: our own small vision-language-action transformer, trained from scratch (no pretrained weights).
# one sequence per robot per decision, a plain pre-norm transformer encoder over it:
#   front camera patches (64) | wrist camera patches (64) | command words (≤ MAX_WORDS, padded) | proprio (1) | action queries (CHUNK)
# each action query's output -> 9 mean actions as offsets from the action that holds the current pose (proprio
# [HOLD], residual like the teacher); the robot executes the CHUNK actions, then the model runs again.
# distillation trains the means with L1 on the teacher's actions (ml/distill.py); GRPO (ml/train_grpo.py) samples
# around them with the learned log_std.
import re
import torch
import torch.nn as nn
from robot.env import PROPRIO_DIM, ACTION_DIM, CAMERAS, HOLD
from robot.lekiwi import CAMERA_RES
from robot.course import GOAL_STEP, GOAL_RANGE, TURN_STEP, COMMANDS

# camera images are IMAGE x IMAGE, cut into PATCH x PATCH patches: (96 / 12)² = 64 per camera. at 64 px a block 30 cm
# away was ~5 px wide and the student could not place or release it precisely; 12 px patches keep the token count
IMAGE = CAMERA_RES
PATCH = 12
# patches per camera
PATCHES = (IMAGE // PATCH) ** 2
MAX_WORDS = 16
# actions per decision: 0.5 s at 10 Hz
CHUNK = 5
WIDTH = 384
LAYERS = 6
HEADS = 6

# word vocabulary: every word the command texts use (robot/course.py command()). index 0 = padding
NUMBERS = [f'{GOAL_STEP * i:.1f}' for i in range(round(GOAL_RANGE / GOAL_STEP) + 1)]  # ["0.0", "0.1", ..., "1.0"]
DEGREES = [str(d) for d in range(0, 181, TURN_STEP)]  # ["0", "15", ..., "180"]
WORDS = sorted({w for text in COMMANDS.values() for w in text.replace(',', ' ,').split()}
               | {'move', 'm', 'forward', 'back', 'and', 'right', 'left', ',', 'then', 'turn', '°', 'stay', 'parked'})  # [str]
VOCAB = ['<pad>'] + WORDS + NUMBERS + DEGREES  # [str]
WORD_ID = {w: i for i, w in enumerate(VOCAB)}  # {str: int}


def tokenize(texts, device):
    # [str] -> (B, MAX_WORDS) word ids, 0-padded. numbers, "°" and "," are their own words
    rows = [[WORD_ID[w] for w in re.findall(r'\d+\.\d+|\d+|[a-z]+|°|,', t)] for t in texts]  # [[int]]
    assert all(len(r) <= MAX_WORDS for r in rows), f'command longer than {MAX_WORDS} words'
    return torch.tensor([r + [0] * (MAX_WORDS - len(r)) for r in rows], device=device)


class VLA(nn.Module):
    def __init__(self):
        super().__init__()
        # one patch projection per camera: (3 * 12 * 12 = 432) -> WIDTH
        self.patch = nn.ModuleList([nn.Linear(3 * PATCH * PATCH, WIDTH) for _ in CAMERAS])
        self.words = nn.Embedding(len(VOCAB), WIDTH)
        self.proprio = nn.Linear(PROPRIO_DIM, WIDTH)
        self.queries = nn.Parameter(torch.randn(CHUNK, WIDTH) * 0.02)
        # learned position embedding for every slot of the sequence (also tells the model which part a token is)
        self.seq_len = len(CAMERAS) * PATCHES + MAX_WORDS + 1 + CHUNK
        self.pos = nn.Parameter(torch.randn(self.seq_len, WIDTH) * 0.02)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, 4 * WIDTH, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, ACTION_DIM)
        # state-independent log std of the gaussian around the mean actions (GRPO only)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -2.3))

    def forward(self, rgb, words, proprio):
        # rgb (B, 3 * len(CAMERAS), 96, 96) uint8, words (B, MAX_WORDS) ids, proprio (B, PROPRIO_DIM)
        # -> mean actions (B, CHUNK, ACTION_DIM) (the env clips actions to [-1, 1])
        B = rgb.shape[0]
        # (B, 6, 96, 96) -> (B, cams, 3, 8 rows, 12, 8 cols, 12) -> (B, cams, 64 patches, 432)
        x = (rgb.float() / 255.0).view(B, len(CAMERAS), 3, IMAGE // PATCH, PATCH, IMAGE // PATCH, PATCH)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, len(CAMERAS), PATCHES, 3 * PATCH * PATCH)
        # (B, cams * 64, WIDTH)
        patches = torch.cat([proj(x[:, c]) for c, proj in enumerate(self.patch)], 1)
        # (B, seq_len, WIDTH)
        tokens = torch.cat([patches, self.words(words), self.proprio(proprio)[:, None], self.queries.expand(B, CHUNK, WIDTH)], 1) + self.pos
        # (B, seq_len) True = padding word, ignored by attention
        pad = torch.zeros(B, self.seq_len, dtype=torch.bool, device=rgb.device)
        pad[:, len(CAMERAS) * PATCHES:len(CAMERAS) * PATCHES + MAX_WORDS] = words == 0
        out = self.encoder(tokens, src_key_padding_mask=pad)
        # (B, CHUNK, WIDTH) -> (B, CHUNK, ACTION_DIM) offsets, + (B, 1, ACTION_DIM) the action holding the current pose
        return proprio[:, None, HOLD] + self.head(self.norm(out[:, -CHUNK:]))

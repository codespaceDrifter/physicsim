# ToFuBot policy: camera frames + joint readings -> actions. shared by behavior cloning, PPO and the live server.
import torch
import torch.nn as nn
from robot.env import PROPRIO_DIM, ACTION_DIM, CAMERAS

# consecutive camera frames stacked on the channel axis (gives the policy motion / short memory)
FRAME_STACK = 3
# channels of one frame: RGB of every camera (front, wrist)
FRAME_CHANNELS = 3 * len(CAMERAS)
# random shift augmentation (DrQ, Kostrikov et al. 2020): pad by this many pixels and crop back at a random offset
SHIFT_PAD = 4


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        # nature-DQN style encoder for 64x64 inputs
        self.encoder = nn.Sequential(
            nn.Conv2d(FRAME_CHANNELS * FRAME_STACK, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # (64, 4, 4) feature map after the convs for 64x64 input
        conv_out = 64 * 4 * 4
        self.proprio = nn.Sequential(nn.Linear(PROPRIO_DIM, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(conv_out + 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mean = nn.Linear(256, ACTION_DIM)
        # state-independent log std of the gaussian (used by PPO, ignored by behavior cloning)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))
        self.value = nn.Sequential(nn.Linear(conv_out + 64, 256), nn.ReLU(), nn.Linear(256, 1))

    def features(self, frames, proprio):
        # frames: (B, FRAME_CHANNELS*FRAME_STACK, 64, 64) uint8, proprio: (B, PROPRIO_DIM)
        # (B, FRAME_CHANNELS*FRAME_STACK, 64, 64) -> (B, 1024)
        img = self.encoder(frames.float() / 255.0)
        # (B, 1024 + 64)
        return torch.cat([img, self.proprio(proprio)], dim=-1)

    def forward(self, frames, proprio):
        # (B, ACTION_DIM) mean action in [-1, 1], (B,) value estimate
        f = self.features(frames, proprio)
        return torch.tanh(self.mean(self.trunk(f))), self.value(f).squeeze(-1)


def random_shift(frames):
    # DrQ augmentation: replicate-pad and crop back at a random integer offset per sample. (B, C, H, W) uint8
    B, C, H, W = frames.shape
    P = SHIFT_PAD
    # (B, C, H + 2P, W + 2P)
    padded = nn.functional.pad(frames.float(), (P, P, P, P), mode='replicate')
    # (B, H) / (B, W) source row / column of every output pixel
    rows = torch.randint(0, 2 * P + 1, (B, 1), device=frames.device) + torch.arange(H, device=frames.device)
    cols = torch.randint(0, 2 * P + 1, (B, 1), device=frames.device) + torch.arange(W, device=frames.device)
    # (B, C, H, W + 2P) -> (B, C, H, W)
    out = padded.gather(2, rows[:, None, :, None].expand(B, C, H, W + 2 * P))
    out = out.gather(3, cols[:, None, None, :].expand(B, C, H, W))
    return out.to(torch.uint8)


class FrameStack:
    # rolling (E, FRAME_CHANNELS*FRAME_STACK, H, W) stack of the latest frames per env, refilled with the first frame on reset
    def __init__(self, first_rgb):
        self.frames = first_rgb.repeat(1, FRAME_STACK, 1, 1)

    def push(self, rgb, done):
        # rgb (E, FRAME_CHANNELS, H, W) uint8, done (E,) True where the env was just reset (rgb is its first frame)
        shifted = torch.cat([self.frames[:, FRAME_CHANNELS:], rgb], dim=1)
        self.frames = torch.where(done[:, None, None, None], rgb.repeat(1, FRAME_STACK, 1, 1), shifted)
        return self.frames

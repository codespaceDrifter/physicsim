# Behavior cloning: train the pixel policy to imitate the scripted expert's actions (data from ml.collect).
# the whole dataset is moved to the GPU as uint8 (200k frames ≈ 2.5 GB) so batches are sampled without copies.
#
# run from back/:  python -m ml.train_bc --steps 30000
# writes checkpoints/bc.pt  {"policy": state_dict, "step": int, "success": float}
import argparse
import json
import os
import time
import numpy as np
import torch
from ml.policy import Policy, random_shift, FRAME_STACK
from ml.rollout import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/bc')
parser.add_argument('--steps', type=int, default=30_000)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--eval-every', type=int, default=5000)
parser.add_argument('--eval-envs', type=int, default=64)
parser.add_argument('--out', default='checkpoints/bc.pt')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(os.path.join(args.data, 'meta.json')) as f:
    T = json.load(f)['frames']
assert T > 0, 'empty dataset'
# (T, 3, 64, 64) uint8 frames, (T, P) proprio, (T, A) actions on the device
frames = torch.from_numpy(np.load(os.path.join(args.data, 'frames.npy'), mmap_mode='r')[:T].copy()).permute(0, 3, 1, 2).contiguous().to(device)
proprio = torch.from_numpy(np.load(os.path.join(args.data, 'proprio.npy'), mmap_mode='r')[:T].copy()).to(device)
actions = torch.from_numpy(np.load(os.path.join(args.data, 'actions.npy'), mmap_mode='r')[:T].copy()).to(device)
episode = np.load(os.path.join(args.data, 'episode.npy'), mmap_mode='r')[:T]
# (T,) index of the first row of each row's episode, so stacks repeat the first frame instead of crossing episodes
first = np.r_[0, np.flatnonzero(np.diff(episode)) + 1]
ep_start = torch.from_numpy(np.repeat(first, np.diff(np.r_[first, T]))).to(device)
print(f'{T} frames, {len(first)} episodes')

policy = Policy().to(device)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
best = -1.0
start = time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(0, T, (args.batch,), device=device)
    # (B, FRAME_STACK) rows oldest -> newest, clamped to the episode start
    rows = torch.maximum(idx[:, None] - torch.arange(FRAME_STACK - 1, -1, -1, device=device), ep_start[idx][:, None])
    # (B, FRAME_STACK, 3, 64, 64) -> (B, 3*FRAME_STACK, 64, 64)
    stacked = random_shift(frames[rows].flatten(1, 2))
    pred, _ = policy(stacked, proprio[idx])
    loss = ((pred - actions[idx]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f'step {step} loss {loss.item():.4f} ({time.time() - start:.0f}s)')
    if step % args.eval_every == 0 or step == args.steps:
        success = evaluate(policy, args.eval_envs, device)
        print(f'step {step} eval success {success:.0%}')
        if success >= best:
            best = success
            torch.save({'policy': policy.state_dict(), 'step': step, 'success': success}, args.out)
            print(f'saved {args.out}')

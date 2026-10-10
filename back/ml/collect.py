# Collect behavior cloning data: run the scripted expert in parallel envs and keep only successful episodes.
#
# writes to --out (default data/bc/), one row per control step:
#   frames.npy   (T, 64, 64, 6) uint8    camera frames the robot saw before acting: front RGB, then wrist RGB
#   proprio.npy  (T, PROPRIO_DIM) float32 joint readings at that moment
#   actions.npy  (T, ACTION_DIM) float32  expert action in [-1, 1]
#   episode.npy  (T,) int32               episode id, so frame stacking never crosses episodes
#   meta.json    {"frames": T, "episodes": n, "attempts": m}  arrays are preallocated, only the first T rows are valid
#
# run from back/:  python -m ml.collect --envs 128 --episodes 500
import argparse
import json
import os
import time
import numpy as np
import torch
from robot.pick_place import PickPlaceEnv, PROPRIO_DIM, ACTION_DIM
from ml.expert import Expert
from ml.policy import FRAME_CHANNELS

parser = argparse.ArgumentParser()
parser.add_argument('--envs', type=int, default=128)
parser.add_argument('--episodes', type=int, default=500, help='successful episodes to keep')
parser.add_argument('--blocks', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max-frames', type=int, default=200_000)
parser.add_argument('--out', default='data/bc')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.out, exist_ok=True)
env = PickPlaceEnv(args.envs, device, num_blocks=args.blocks, seed=args.seed)
if device == 'cuda':
    env.enable_cuda_graph()
expert = Expert(args.envs, args.blocks)

frames = np.lib.format.open_memmap(os.path.join(args.out, 'frames.npy'), mode='w+', dtype=np.uint8, shape=(args.max_frames, 64, 64, FRAME_CHANNELS))
proprio = np.lib.format.open_memmap(os.path.join(args.out, 'proprio.npy'), mode='w+', dtype=np.float32, shape=(args.max_frames, PROPRIO_DIM))
actions = np.lib.format.open_memmap(os.path.join(args.out, 'actions.npy'), mode='w+', dtype=np.float32, shape=(args.max_frames, ACTION_DIM))
episode = np.lib.format.open_memmap(os.path.join(args.out, 'episode.npy'), mode='w+', dtype=np.int32, shape=(args.max_frames,))

# per env lists of the running episode: [(frames (64, 64, 6), proprio, action)]
running = [[] for _ in range(args.envs)]
T = 0  # rows written
kept = 0  # successful episodes written
attempts = 0  # finished episodes
obs = env.observe()
start = time.time()
while kept < args.episodes:
    targets = torch.tensor(expert.act(env), device=device)
    action = env.targets_to_action(targets)
    rgb = obs['rgb'].permute(0, 2, 3, 1).cpu().numpy()
    prop = obs['proprio'].cpu().numpy()
    act = action.cpu().numpy()
    for e in range(args.envs):
        running[e].append((rgb[e], prop[e], act[e]))
    obs, reward, done, info = env.step(action)
    for e in done.nonzero().view(-1).tolist():
        attempts += 1
        steps = running[e]
        if info['success'][e] and kept < args.episodes:
            n = min(len(steps), args.max_frames - T)
            assert n == len(steps), f'--max-frames {args.max_frames} too small'
            frames[T:T + n] = np.stack([s[0] for s in steps])
            proprio[T:T + n] = np.stack([s[1] for s in steps])
            actions[T:T + n] = np.stack([s[2] for s in steps])
            episode[T:T + n] = kept
            T += n
            kept += 1
        running[e] = []
        expert.reset(e)
    if done.any():
        print(f'kept {kept}/{args.episodes} episodes ({T} frames), expert success {kept / max(attempts, 1):.0%}, {time.time() - start:.0f}s')

for arr in (frames, proprio, actions, episode):
    arr.flush()
with open(os.path.join(args.out, 'meta.json'), 'w') as f:
    json.dump({'frames': T, 'episodes': kept, 'attempts': attempts}, f)
print(f'done: {kept} episodes, {T} frames -> {args.out}')

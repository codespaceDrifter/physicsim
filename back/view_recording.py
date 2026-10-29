# Look at a recording without the browser: render chosen moments from a camera chasing the robot into one contact
# sheet (PNG), and print how the robot moved (speed, bounce, sway, how jerky). For checking a policy by eye.
#
# run from back/:  python -m view_recording ../logs/warehouse_walk.simrec --seconds 0 1 2 3 4 5 --out sheet.png
# the robot followed is the body named --follow (default the G1's pelvis)
import argparse
import math
import struct
import zlib
import numpy as np
import torch
from recording import load
from physics.world import World
from physics.render import Renderer

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--seconds', type=float, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument('--follow', default='g1/pelvis')
parser.add_argument('--distance', type=float, default=2.5, help='camera distance behind and beside the robot (m)')
parser.add_argument('--size', type=int, default=320)
parser.add_argument('--out', default='sheet.png')
args = parser.parse_args()

scene, info, spf, frames = load(args.path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
world = World(scene, 1, device)
body = world.body_index[args.follow]
# (F,) times, (F, N, 7) poses
times = frames[:, 0]
poses = frames[:, 1:].reshape(len(frames), -1, 7)


def look_at(eye, target):
    # [w, x, y, z] camera rotation looking from eye to target (three.js camera: looks down local -z, up = local +y)
    f = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(f, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    R = np.stack([right, up, -f], 1)
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]


tiles = []  # [(H, W, 3) uint8]
for s in args.seconds:
    i = min(int(round(s / spf)), len(frames) - 1)
    world.pos[0] = torch.tensor(poses[i, :, :3], device=device)
    world.quat[0] = torch.tensor(poses[i, :, 3:], device=device)
    target = poses[i, body, :3].astype(np.float64)
    # behind-left of the robot's heading, above it
    w_, x_, y_, z_ = poses[i, body, 3:]
    heading = np.array([1 - 2 * (y_ * y_ + z_ * z_), 0.0, 2 * (x_ * z_ - w_ * y_)])
    heading /= max(np.linalg.norm(heading), 1e-6)
    side = np.array([heading[2], 0.0, -heading[0]])
    eye = target - args.distance * (0.8 * heading - 0.6 * side) + np.array([0.0, 1.0, 0.0])
    cam = {'id': 'view', 'body': 'ground', 'position': eye.tolist(), 'rotation': look_at(eye, target), 'fov': 60,
           'width': args.size, 'height': args.size}
    tiles.append(Renderer(world, cam).render()[0].cpu().numpy())
# PNG: the sheet as 8-bit RGB rows (filter byte 0), deflated, in IHDR / IDAT / IEND chunks
sheet = np.concatenate(tiles, 1)
chunk = lambda kind, body: struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff)
rows = b''.join(b'\x00' + row.tobytes() for row in sheet)
with open(args.out, 'wb') as f:
    f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', sheet.shape[1], sheet.shape[0], 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))

# motion summary of the followed body
p = poses[:, body, :3]
v = np.diff(p, axis=0) / spf
speed = np.linalg.norm(v[:, [0, 2]], axis=1)
# bounce: vertical velocity spread; jerk: change of horizontal velocity per frame
bounce = v[:, 1].std()
jerk = np.linalg.norm(np.diff(v[:, [0, 2]], axis=0), axis=1).mean() / spf
print(f'{args.path}: {times[-1]:.1f} s, info {info}')
print(f'  {args.follow}: mean speed {speed.mean():.2f} m/s (max {speed.max():.2f}), travelled {np.linalg.norm(p[-1, [0, 2]] - p[0, [0, 2]]):.2f} m net, '
      f'height {p[:, 1].mean():.3f} ± {p[:, 1].std():.3f} m, bounce {bounce:.3f} m/s, horizontal jerk {jerk:.1f} m/s²')
print(f'  wrote {args.out} ({len(tiles)} frames at {args.seconds} s)')

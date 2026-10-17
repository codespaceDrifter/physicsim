# Pre-recorded games for the web demo: the frontend's simple view plays them back to back, so the web server needs no
# GPU (it only serves these files). each game: one race (60 s), a random layout like in training, orange and teal
# driven by --orange / --teal, knocked-over robots respawn (live.py).
#
# run from back/ on a GPU:  python -m record_games --games 6
# writes logs/game_{i}/  scene.json, {chunk}.bin   body poses at FPS (recording.py format)
#                        game.json    {"stage": "race", "seconds": 60, "drivers": [orange, teal],
#                                      "cameras": [camera ids, mosaic order], "score": [[time, orange, teal], ...]}
#                                      score: at t = 0 and at every change
#                        cameras.mp4  the 4 robot cameras at 10 Hz as one 2 x 2 mosaic (row-major in "cameras" order:
#                                     orange front, orange wrist / teal front, teal wrist), H.264
import argparse
import json
import subprocess
import torch
from live import LiveSession, CAMERA_EVERY, make_driver
from recording import Recorder, LOGS_DIR
from robot.course import STAGE_STEPS
from robot.env import ACTION_REPEAT

parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, default=6)
parser.add_argument('--orange', default='teacher', help='expert | teacher | student')
parser.add_argument('--teal', default='teacher', help='expert | teacher | student')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = 'cuda'
# body poses saved per second
FPS = 30
scene = json.load(open('scenes/course_4_race.json'))
# 60 s: the race's control steps at 10 Hz
seconds = STAGE_STEPS['race'] * ACTION_REPEAT * scene['settings']['dt']
cams = scene['cameras']
W, H = cams[0]['width'], cams[0]['height']
assert len(cams) == 4 and all((c['width'], c['height']) == (W, H) for c in cams)
for g in range(args.games):
    s = LiveSession(scene, device)
    env, w = s.env, s.world
    # a random layout (robot starts, block spots) like in training instead of the scene's fixed one
    env.randomize = True
    env.gen.manual_seed(args.seed + g)
    env.reset(torch.ones(1, dtype=torch.bool, device=device))
    s.start_xz = env.base_frame()[0][0].clone()
    s.autopilot = f'{args.orange} vs {args.teal}'
    s.drivers = [(make_driver(args.orange, device), [0]), (make_driver(args.teal, device), [1])]
    name = f'game_{g}'
    total = int(round(seconds * FPS)) + 1
    rec = Recorder(name, w.scene, 1 / FPS, total)
    # [[time, orange, teal]] score at t = 0 and at every change
    score = [[0.0, *env.score()[0].tolist()]]
    video = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{2 * W}x{2 * H}',
                              '-r', str(round(1 / (CAMERA_EVERY * w.dt))), '-i', '-', '-c:v', 'libx264', '-crf', '16',
                              '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(LOGS_DIR / name / 'cameras.mp4')],
                             stdin=subprocess.PIPE)
    for i in range(total):
        while w.time < i / FPS - 1e-9:
            if s.steps % CAMERA_EVERY == 0:
                # (4, H, W, 3) -> (2 H, 2 W, 3) mosaic, row-major
                rgb = torch.stack([c.render()[0] for c in s.cameras])
                video.stdin.write(rgb.view(2, 2, H, W, 3).permute(0, 2, 1, 3, 4).reshape(2 * H, 2 * W, 3).cpu().numpy().tobytes())
            s.step()
        rec.add(i / FPS, w.poses()[0].cpu().numpy())
        now = env.score()[0].tolist()
        if now != score[-1][1:]:
            score.append([round(i / FPS, 3), *now])
    video.stdin.close()
    assert video.wait() == 0, 'ffmpeg failed'
    json.dump({'stage': 'race', 'seconds': seconds, 'drivers': [args.orange, args.teal], 'cameras': [c['id'] for c in cams],
               'score': score}, open(LOGS_DIR / name / 'game.json', 'w'))
    print(f'{name}: final score orange {score[-1][1]} : {score[-1][2]} teal', flush=True)

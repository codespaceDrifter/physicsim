# Record the MuJoCo course's policy (ml/train_humanoid.py --backend mujoco checkpoint) for the frontend's playback
# mode: one env, mean actions, no pushes, --seconds of one stage, episode after episode; prints passes / tries.
# the recording uses the physicsim scene format (robot/g1.py's G1 with its meshes, plus the box, pad and ground): every
# body's pose is MuJoCo's center-of-mass frame (xipos / ximat, the frame our MJCF loader builds bodies in) turned from
# MuJoCo's z-up axes into ours (robot/mjcf.py CONVERT). walk follows robot/humanoid.py's command SCRIPT.
#
# also renders logs/{name}_render.mp4 (640x480, 25 fps, a camera following the pelvis) from the scene with the G1's meshes
# (mjcf(visual=True)); headless rendering needs MUJOCO_GL=osmesa (or egl).
#
# run from back/:  MUJOCO_GL=osmesa python -m ml.record_mj --ckpt checkpoints/mj_v1.pt --stage walk --name mj_walk
# writes logs/{name}.simrec (recording.py format), info {"stage", "checkpoint", "summary"}, and logs/{name}_render.mp4 (not {name}.mp4: the API takes a recording with that video for a pre-recorded game)
import argparse
import subprocess
import mujoco
import numpy as np
import torch
from recording import Recorder, LOGS_DIR
from robot.g1 import g1, PREFIX
from robot.mjcf import CONVERT, matrix_to_quat
from robot.mj_warehouse import mjcf, MjWarehouseCourse, STAGES, PROPRIO_DIM, ACTION_DIM, AUX_DIM, TASK_DIM, BOX_SIZE, PAD_SIZE
from robot.humanoid import DT, SCRIPT
from ml.humanoid_policy import Policy, HISTORY, load_matching

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/mj_v1.pt')
parser.add_argument('--stage', default='walk', choices=STAGES)
parser.add_argument('--seconds', type=float, default=30.0, help='length of a box stage recording')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', default='mj_walk')
args = parser.parse_args()

device = 'cuda'
ck = torch.load(args.ckpt, map_location=device)
assert ck['args']['backend'] == 'mujoco'
policy = Policy(PROPRIO_DIM, ACTION_DIM, AUX_DIM, TASK_DIM, 0, False, len(STAGES), ck['args']['width'], ck['args']['layers'], ck['args']['group']).to(device)
with torch.no_grad():
    assert not load_matching(policy, ck['policy'])
policy.eval()
env = MjWarehouseCourse(1, device, [float(s == args.stage) for s in STAGES], seed=args.seed)
env.pushes = False
env.test_prob = 1.0
env.reset(torch.ones(1, dtype=torch.bool, device=device))
# the recording's commands come from SCRIPT (or the stage), not the env's heading hold
env.heading_mode[:] = False
walk = args.stage == 'walk'
total = int((sum(s for s, c in SCRIPT) if walk else args.seconds) / DT)
if walk:
    env.ep_steps[:] = total + 1
    env.stage_steps[:] = total + 1

# the scene: ground, box, pad, then the G1's bodies (physicsim format, y-up)
identity = [1.0, 0.0, 0.0, 0.0]
still = {'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'density': 1000.0, 'static': True, 'collide': True,
         'group': 0, 'friction': 1.0, 'restitution': 0.0}
box_shape = lambda size: [{'type': 'box', 'size': size, 'position': [0.0, 0.0, 0.0], 'rotation': identity}]
bot = g1()
scene = {'name': 'mj_warehouse', 'settings': {'dt': DT, 'substeps': 5, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0, 'max_contacts': 128},
         'bodies': [{'id': 'ground', 'shapes': [{'type': 'plane', 'position': [0.0, 0.0, 0.0], 'rotation': identity}],
                     'position': [0.0, 0.0, 0.0], 'rotation': identity, 'color': '#9a9a9a', **still},
                    {'id': 'box', 'shapes': box_shape(BOX_SIZE), 'position': [0.0, 0.0, 0.0], 'rotation': identity, 'color': '#d1495b', **still},
                    {'id': 'pad', 'shapes': box_shape([PAD_SIZE, 0.005, PAD_SIZE]), 'position': [0.0, 0.0, 0.0], 'rotation': identity,
                     'color': '#3a6ea5', **{**still, 'collide': False}}] + bot['bodies'],
         'joints': bot['joints'], 'cameras': []}
# MuJoCo body id of each recorded body after ground / box / pad (the G1's, by name)
mj_ids = [mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_BODY, b['id'].removeprefix(PREFIX)) for b in bot['bodies']]
assert min(mj_ids) >= 0
xipos, ximat = env.d.xipos, env.d.ximat


def poses():
    # (N, 7) every scene body's [px, py, pz, qw, qx, qy, qz] in our axes
    p = xipos.numpy()[0]
    R = ximat.numpy()[0]
    out = [[0.0, 0.0, 0.0] + identity]
    # the box's own axes change too (MuJoCo: x depth, y width, z height; ours: x depth, y height, z width)
    out.append(list(CONVERT @ p[env.box]) + matrix_to_quat(CONVERT @ R[env.box] @ CONVERT.T))
    pad = env.mocap_pos[0, 0].cpu().numpy()
    out.append(list(CONVERT @ pad) + identity)
    out += [list(CONVERT @ p[i]) + matrix_to_quat(CONVERT @ R[i]) for i in mj_ids]
    return np.array(out, dtype=np.float32)


# the video: the scene with meshes (same qpos layout as the env's), a camera tracking the pelvis, frames piped to ffmpeg
view = mujoco.MjModel.from_xml_string(mjcf(visual=True))
assert view.nq == env.mjm.nq
view_data = mujoco.MjData(view)
renderer = mujoco.Renderer(view, 480, 640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
camera.trackbodyid = mujoco.mj_name2id(view, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
camera.distance, camera.elevation, camera.azimuth = 3.0, -15.0, 135.0
# control steps per video frame (50 Hz control -> 25 fps)
FRAME_EVERY = 2
video = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480',
                          '-r', str(round(1 / (DT * FRAME_EVERY))), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                          str(LOGS_DIR / f'{args.name}_render.mp4')], stdin=subprocess.PIPE)


def frame():
    # render the env's current state into the video
    view_data.qpos[:] = env.qpos[0].cpu().numpy()
    view_data.mocap_pos[:] = env.mocap_pos[0].cpu().numpy()
    mujoco.mj_forward(view, view_data)
    renderer.update_scene(view_data, camera)
    video.stdin.write(renderer.render().tobytes())


rec = Recorder(args.name, scene, DT)
rec.add(0.0, poses())
frame()
obs = env.observe(noise=False)
history = obs['proprio'][:, 0][:, None].repeat(1, HISTORY, 1)
plan = [(int(s / DT), c) for s, c in SCRIPT] if walk else [(total, None)]
counts = {}  # {metric: sum over the recording}
summary = []  # [str]
step = 0
with torch.no_grad():
    for steps, command in plan:
        err = torch.zeros(3, device=device)
        for _ in range(steps):
            if command is not None:
                env.command[:] = torch.tensor(command, device=device)
            mean, _ = policy(history, env.observe(noise=False)['command'][:, 0], None)
            _, done, _, info = env.step(mean.view(1, 1, -1))
            env.heading_mode[:] = False
            assert not (walk and done.any()), f'fell at {step * DT:.2f} s'
            step += 1
            for k, v in info['metrics'].items():
                counts[k] = counts.get(k, 0.0) + float(v)
            if walk:
                err += (env.base_velocity()[0] - env.command[0, 0]).abs() / steps
            rec.add(step * DT, poses())
            if step % FRAME_EVERY == 0:
                frame()
            obs = env.observe(noise=False)
            history = torch.where(done[:, None, None], obs['proprio'][:, 0][:, None], torch.cat([history[:, 1:], obs['proprio'][:, 0][:, None]], 1))
        if walk:
            summary.append(f'command {command}: mean |error| forward {err[0]:.2f} m/s, left {err[1]:.2f} m/s, turn {err[2]:.2f} rad/s')
if not walk:
    summary.append(f'{args.seconds:.0f} s of {args.stage}: {counts[f"passed_{args.stage}"]:.0f} passed of {counts[f"tried_{args.stage}"]:.0f} tries')
print('\n'.join(summary))
rec.save({'stage': args.stage, 'checkpoint': args.ckpt, 'summary': summary})
video.stdin.close()
assert video.wait() == 0
print(f'wrote {LOGS_DIR / args.name}_render.mp4')

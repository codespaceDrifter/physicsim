# Live simulation for the editor: one world stepped in real time, driven by websocket messages (see api.py).
#
# a session holds one world (1 env). messages in: play / pause / resume / step / stop / drive / autopilot.
# frames out: {"type": "frame", "time", "poses": [[px,py,pz,qw,qx,qy,qz], ...], "joints": {id: value}, "cameras": [...],
#              "info": {"score": [orange, teal]} in the block race}
#
# the scene picks the task: two LeKiwis + home zones = block race (you drive lekiwi/, the autopilot drives the
# other robot, or both with the "_both" modes), one LeKiwi + block/i + zone/i = pick-and-place.
import base64
import os
import torch
from physics.world import World
from physics.render import Renderer
from robot.lekiwi import PREFIX
from robot.env import ACTION_REPEAT
from robot.pick_place import PickPlaceEnv
from robot.block_race import BlockRaceEnv
from ml.expert import Expert
from ml.policy import Policy, FrameStack

CHECKPOINTS = [os.path.join(os.path.dirname(__file__), 'checkpoints', n) for n in ('ppo.pt', 'bc.pt')]
# physics steps between camera renders sent to the client (60 Hz physics -> 10 Hz camera)
CAMERA_EVERY = 6
AUTOPILOT_MODES = ('off', 'expert', 'policy', 'expert_both', 'policy_both')


class LiveSession:
    def __init__(self, scene, device):
        self.device = device
        ids = {b['id'] for b in scene['bodies']}
        # number of block/i bodies with a matching zone/i: the pick-and-place autopilot needs them and the robot
        num_blocks = 0
        while f'block/{num_blocks}' in ids and f'zone/{num_blocks}' in ids:
            num_blocks += 1
        self.race = PREFIX + 'base' in ids and 'lekiwi2/base' in ids and 'home/0' in ids and 'home/1' in ids
        if self.race:
            self.env = BlockRaceEnv(1, device, scene_json=scene)
        elif PREFIX + 'base' in ids and num_blocks > 0:
            self.env = PickPlaceEnv(1, device, num_blocks=num_blocks, scene_json=scene)
        else:
            self.env = None
        self.world = self.env.world if self.env else World(scene, 1, device)
        self.cameras = [Renderer(self.world, c) for c in scene['cameras']]
        if str(device).startswith('cuda'):
            if self.env:
                self.env.enable_cuda_graph()
            else:
                self.world.enable_cuda_graph()
                # capture warm-up stepped the world, start from the scene again
                self.world.reset(torch.ones(1, dtype=torch.bool, device=device))
            for cam in self.cameras:
                cam.enable_cuda_graph()
            self.world.time = 0.0
        self.world.joints.prepare()
        self.steps = 0
        self.autopilot = 'off'
        # [int] robots the autopilot drives
        self.auto_robots = []
        self.expert = None
        self.policy = None
        self.stack = None

    def set_autopilot(self, mode):
        assert mode in AUTOPILOT_MODES, f'unknown autopilot mode {mode}'
        if mode != 'off' and self.env is None:
            raise ValueError('autopilot needs a lekiwi task scene (lekiwi_pick or block_race)')
        if mode.startswith('policy'):
            path = next((p for p in CHECKPOINTS if os.path.exists(p)), None)
            if path is None:
                raise ValueError('no trained policy: run python -m ml.train_bc (checkpoints/bc.pt) first')
            self.policy = Policy().to(self.device)
            self.policy.load_state_dict(torch.load(path, map_location=self.device)['policy'])
            self.policy.eval()
            self.stack = None
        if mode.startswith('expert'):
            self.expert = Expert(1, self.env.R)
        self.autopilot = mode
        # race: you keep robot 0 unless "_both"; pick: the only robot
        both = mode.endswith('_both') or not self.race
        self.auto_robots = [] if mode == 'off' else (list(range(self.env.R)) if both else [1])

    def drive(self, targets):
        # {joint_id: target} from teleop (robot 0's joints). ignored while the autopilot drives robot 0
        if 0 not in self.auto_robots:
            self.world.joints.set_targets(targets)

    @torch.no_grad()
    def step(self):
        # one physics step, with the autopilot choosing new targets at the 10 Hz control rate
        if self.auto_robots and self.steps % ACTION_REPEAT == 0:
            env = self.env
            if self.autopilot.startswith('expert'):
                # (1, R, A)
                targets = torch.tensor(self.expert.act(env, robots=self.auto_robots), device=self.device)
            else:
                obs = env.observe()
                # one frame stack per robot: (R, C, H, W)
                rgb = obs['rgb'][0]
                if self.stack is None:
                    self.stack = FrameStack(rgb)
                frames = self.stack.push(rgb, torch.zeros(env.R, dtype=torch.bool, device=self.device))
                action, _ = self.policy(frames, obs['proprio'][0])
                targets = env.action_to_targets(action).unsqueeze(0)
            for r in self.auto_robots:
                self.world.joints.target[:, env.act_joints[r]] = targets[:, r]
        self.world.step()
        self.steps += 1

    def frame(self, with_cameras):
        # websocket frame message for env 0
        w = self.world
        msg = {'type': 'frame', 'time': w.time, 'poses': w.poses()[0].cpu().tolist(),
               'joints': dict(zip(w.joints.ids, w.joints.value[0].cpu().tolist())), 'cameras': []}
        if self.race:
            msg['info'] = {'score': self.env.score()[0].tolist()}
        if with_cameras:
            for cam in self.cameras:
                rgb = cam.render()[0].cpu().numpy()
                msg['cameras'].append({'id': cam.id, 'width': cam.W, 'height': cam.H, 'rgb': base64.b64encode(rgb.tobytes()).decode()})
        return msg

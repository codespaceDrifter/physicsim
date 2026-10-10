# Live simulation for the editor: one world stepped in real time, driven by websocket messages (see api.py).
#
# a session holds one world (1 env). messages in: play / pause / resume / step / stop / drive / autopilot.
# frames out: {"type": "frame", "time", "poses": [[px,py,pz,qw,qx,qy,qz], ...], "joints": {id: value}, "cameras": [...]}
import base64
import os
import torch
from physics.world import World
from physics.render import Renderer
from robot.lekiwi import PREFIX
from robot.pick_place import PickPlaceEnv, ACTION_REPEAT
from ml.expert import Expert
from ml.policy import Policy, FrameStack

CHECKPOINTS = [os.path.join(os.path.dirname(__file__), 'checkpoints', n) for n in ('ppo.pt', 'bc.pt')]
# physics steps between camera renders sent to the client (60 Hz physics -> 10 Hz camera)
CAMERA_EVERY = 6


class LiveSession:
    def __init__(self, scene, device):
        self.device = device
        ids = {b['id'] for b in scene['bodies']}
        # number of block/i bodies with a matching zone/i: the pick-and-place autopilot needs them and the robot
        num_blocks = 0
        while f'block/{num_blocks}' in ids and f'zone/{num_blocks}' in ids:
            num_blocks += 1
        self.has_robot = PREFIX + 'base' in ids and num_blocks > 0
        if self.has_robot:
            self.env = PickPlaceEnv(1, device, num_blocks=num_blocks, scene_json=scene)
            self.world = self.env.world
        else:
            self.env = None
            self.world = World(scene, 1, device)
        self.cameras = [Renderer(self.world, c) for c in scene['cameras']]
        if str(device).startswith('cuda'):
            if self.has_robot:
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
        self.expert = None
        self.policy = None
        self.stack = None

    def set_autopilot(self, mode):
        assert mode in ('off', 'expert', 'policy'), f'unknown autopilot mode {mode}'
        if mode != 'off' and not self.has_robot:
            raise ValueError('autopilot needs the lekiwi and block/i + zone/i bodies in the scene')
        if mode == 'policy':
            path = next((p for p in CHECKPOINTS if os.path.exists(p)), None)
            if path is None:
                raise ValueError('no trained policy: run python -m ml.train_bc (checkpoints/bc.pt) first')
            self.policy = Policy().to(self.device)
            self.policy.load_state_dict(torch.load(path, map_location=self.device)['policy'])
            self.policy.eval()
            self.stack = None
        if mode == 'expert':
            self.expert = Expert(1, self.env.num_blocks)
        self.autopilot = mode

    def drive(self, targets):
        # {joint_id: target} from teleop. ignored while an autopilot drives
        if self.autopilot == 'off':
            self.world.joints.set_targets(targets)

    @torch.no_grad()
    def step(self):
        # one physics step, with the autopilot choosing new targets at the 10 Hz control rate
        if self.autopilot != 'off' and self.steps % ACTION_REPEAT == 0:
            if self.autopilot == 'expert':
                targets = torch.tensor(self.expert.act(self.env), device=self.device)
            else:
                obs = self.env.observe()
                if self.stack is None:
                    self.stack = FrameStack(obs['rgb'])
                frames = self.stack.push(obs['rgb'], torch.zeros(1, dtype=torch.bool, device=self.device))
                action, _ = self.policy(frames, obs['proprio'])
                targets = self.env.action_to_targets(action)
            self.world.joints.target[:, self.env.act_joints] = targets
        self.world.step()
        self.steps += 1

    def frame(self, with_cameras):
        # websocket frame message for env 0
        w = self.world
        msg = {'type': 'frame', 'time': w.time, 'poses': w.poses()[0].cpu().tolist(),
               'joints': dict(zip(w.joints.ids, w.joints.value[0].cpu().tolist())), 'cameras': []}
        if with_cameras:
            for cam in self.cameras:
                rgb = cam.render()[0].cpu().numpy()
                msg['cameras'].append({'id': cam.id, 'width': cam.W, 'height': cam.H, 'rgb': base64.b64encode(rgb.tobytes()).decode()})
        return msg

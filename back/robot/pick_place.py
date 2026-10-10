# LeKiwi pick-and-place: batched environment. each env has the robot, colored blocks and matching colored target
# zones on the floor. the robot must carry block i into zone i. observations are camera pixels + joint readings.
import math
import torch
from physics.world import World
from physics.render import Renderer
from physics.math3d import quat_mul, quat_rotate, quat_from_rotvec
from robot.lekiwi import lekiwi, ACTUATORS, STS3215_SPEED, ARM_LIMITS, PREFIX

# sized for the SO-101 gripper
BLOCK_SIZE = 0.035
ZONE_SIZE = 0.14
# block i and zone i share a hue
BLOCK_COLORS = ['#d93b3b', '#3b7fd9', '#3bb34a']
ZONE_COLORS = ['#f2a3a3', '#a3c4f2', '#a8e0ae']
# blocks and zones spawn at this distance range from the robot (m), at least MIN_GAP apart
SPAWN_MIN, SPAWN_MAX = 0.45, 0.95
MIN_GAP = 0.25
DT = 1 / 60
SUBSTEPS = 4
# physics steps per control step: 6 * (1/60) s = 10 Hz control
ACTION_REPEAT = 6
# control steps allowed per block (30 s at 10 Hz)
EPISODE_STEPS_PER_BLOCK = 300
# reward shaping weights: potential = -(GRIP_W * |gripper - block| + PLACE_W * |block - zone|_xz)
GRIP_W = 1.0
PLACE_W = 2.0
SUCCESS_REWARD = 10.0
# joint readings fed to the policy: arm joint angles + wheel speeds
PROPRIO_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
WHEELS = ['wheel_back', 'wheel_left', 'wheel_right']
PROPRIO_DIM = len(PROPRIO_JOINTS) + len(WHEELS)
# cameras stacked on the channel axis of the "rgb" observation: front RGB, then wrist RGB
CAMERAS = ['front_cam', 'wrist_cam']
ACTION_DIM = len(ACTUATORS)


def scene(num_blocks):
    # scene JSON with the robot at the origin facing +x, blocks and zones in a default (non-random) layout
    def static_box(id, size, pos, color, collide):
        return {'id': id, 'shape': {'type': 'box', 'size': size}, 'position': pos, 'rotation': [1.0, 0.0, 0.0, 0.0],
                'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'density': 1000.0, 'static': True,
                'collide': collide, 'group': 0, 'friction': 0.8, 'restitution': 0.0, 'color': color}

    ground = static_box('ground', None, [0.0, 0.0, 0.0], '#9a9a9a', True)
    ground['shape'] = {'type': 'plane'}
    ground['friction'] = 1.0
    bot = lekiwi()
    blocks = []  # [body dict]
    zones = []  # [body dict]
    for i in range(num_blocks):
        angle = (i - (num_blocks - 1) / 2) * 0.7
        block = static_box(f'block/{i}', [BLOCK_SIZE] * 3, [0.6 * math.cos(angle), BLOCK_SIZE / 2, 0.6 * math.sin(angle)], BLOCK_COLORS[i], True)
        block.update(static=False, density=800.0)
        blocks.append(block)
        # zones are thin static markers that nothing collides with
        zones.append(static_box(f'zone/{i}', [ZONE_SIZE, 0.002, ZONE_SIZE], [-0.6 * math.cos(angle), 0.001, 0.6 * math.sin(angle)], ZONE_COLORS[i], False))
    return {'name': 'lekiwi_pick', 'settings': {'dt': DT, 'substeps': SUBSTEPS, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0},
            'bodies': [ground] + bot['bodies'] + blocks + zones, 'joints': bot['joints'], 'cameras': bot['cameras']}


class PickPlaceEnv:
    # scene_json: use this scene as is (e.g. edited in the editor) instead of a randomized layout. it must contain
    # the lekiwi and bodies "block/0".."block/{num_blocks-1}", "zone/0"..
    def __init__(self, num_envs, device, num_blocks=1, seed=0, scene_json=None):
        self.E = num_envs
        self.device = device
        self.num_blocks = num_blocks
        self.max_steps = EPISODE_STEPS_PER_BLOCK * num_blocks
        self.world = World(scene_json if scene_json is not None else scene(num_blocks), num_envs, device)
        # [Renderer] one per camera, in CAMERAS order
        cams = {c['id']: c for c in self.world.scene['cameras']}
        self.cameras = [Renderer(self.world, cams[PREFIX + n]) for n in CAMERAS]
        self.gen = torch.Generator(device=device).manual_seed(seed)
        w = self.world
        # (R,) body indices of robot parts, block i, zone i
        self.robot = torch.tensor([i for i, bid in enumerate(w.body_ids) if bid.startswith(PREFIX)], device=device)
        self.blocks = torch.tensor([w.body_index[f'block/{i}'] for i in range(num_blocks)], device=device)
        self.zones = torch.tensor([w.body_index[f'zone/{i}'] for i in range(num_blocks)], device=device)
        self.base = w.body_index[PREFIX + 'base']
        # (2,) the two gripper jaws; their mean is the grip point
        self.jaws = torch.tensor([w.body_index[PREFIX + 'gripper'], w.body_index[PREFIX + 'jaw']], device=device)
        # (A,) joint index of each actuator in action order
        self.act_joints = torch.tensor([w.joints.index[PREFIX + name] for name, kind in ACTUATORS], device=device)
        self.proprio_joints = torch.tensor([w.joints.index[PREFIX + n] for n in PROPRIO_JOINTS], device=device)
        self.wheel_joints = torch.tensor([w.joints.index[PREFIX + n] for n in WHEELS], device=device)
        # (A,) affine map from action in [-1, 1] to joint target: target = mid + half * action
        lo, hi = [], []  # [float], [float]
        for name, kind in ACTUATORS:
            if kind == 'wheel':
                lo.append(-STS3215_SPEED), hi.append(STS3215_SPEED)
            else:
                lo.append(ARM_LIMITS[name][0]), hi.append(ARM_LIMITS[name][1])
        self.act_lo = torch.tensor(lo, device=device)
        self.act_hi = torch.tensor(hi, device=device)
        # (P,) position of each proprio joint inside the action vector, to normalize readings with the same ranges
        proprio_act = [[name for name, kind in ACTUATORS].index(n) for n in PROPRIO_JOINTS]
        self.proprio_lo = self.act_lo[proprio_act]
        self.proprio_hi = self.act_hi[proprio_act]
        # (E,) control steps taken this episode, (E, B) block already delivered
        self.steps = torch.zeros(self.E, dtype=torch.long, device=device)
        self.placed = torch.zeros(self.E, num_blocks, dtype=torch.bool, device=device)
        # randomize layouts on reset (training), or keep the given scene as is (the live editor)
        self.randomize = scene_json is None
        self.reset(torch.ones(self.E, dtype=torch.bool, device=device))

    def enable_cuda_graph(self):
        # physics step and every camera as CUDA graphs, then a fresh reset (capture warm-up stepped the world)
        self.world.enable_cuda_graph()
        for cam in self.cameras:
            cam.enable_cuda_graph()
        self.reset(torch.ones(self.E, dtype=torch.bool, device=self.device))

    def action_to_targets(self, action):
        # (E, A) in [-1, 1] -> (E, A) joint targets
        return self.act_lo + (action.clamp(-1, 1) + 1) / 2 * (self.act_hi - self.act_lo)

    def targets_to_action(self, targets):
        # inverse of action_to_targets, used to record the expert's targets as actions
        return ((targets - self.act_lo) / (self.act_hi - self.act_lo) * 2 - 1).clamp(-1, 1)

    def reset(self, mask):
        # re-randomize envs where mask (E,) is True: robot yaw, block and zone positions
        w = self.world
        w.reset(mask)
        n = int(mask.sum())
        if n == 0:
            return
        if not self.randomize:
            w.joints.prepare()
            self.steps[mask] = 0
            self.placed[mask] = False
            self.potential = self.compute_potential()
            return
        envs = mask.nonzero().view(-1)
        # rotate the whole robot about y around the origin
        yaw = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * math.pi
        q_yaw = quat_from_rotvec(torch.stack([torch.zeros_like(yaw), yaw, torch.zeros_like(yaw)], -1))
        R = len(self.robot)
        w.pos[envs[:, None], self.robot] = quat_rotate(q_yaw[:, None].expand(n, R, 4), w.init_pos[self.robot].expand(n, R, 3))
        w.quat[envs[:, None], self.robot] = quat_mul(q_yaw[:, None].expand(n, R, 4), w.init_quat[self.robot].expand(n, R, 4))
        # blocks and zones: rejection-sample positions around the robot that keep MIN_GAP from each other
        k = 2 * self.num_blocks
        # (n, k, 2) xz of [blocks..., zones...]
        spots = torch.zeros(n, k, 2, device=self.device)
        for i in range(k):
            todo = torch.ones(n, dtype=torch.bool, device=self.device)
            while todo.any():
                r = SPAWN_MIN + (SPAWN_MAX - SPAWN_MIN) * torch.rand(n, generator=self.gen, device=self.device)
                a = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * math.pi
                cand = torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
                ok = (torch.linalg.norm(spots[:, :i] - cand[:, None], dim=-1) > MIN_GAP).all(-1) if i > 0 else torch.ones(n, dtype=torch.bool, device=self.device)
                spots[:, i] = torch.where((todo & ok)[:, None], cand, spots[:, i])
                todo = todo & ~ok
        block_yaw = (torch.rand(n, self.num_blocks, generator=self.gen, device=self.device) * 2 - 1) * math.pi
        q_block = quat_from_rotvec(torch.stack([torch.zeros_like(block_yaw), block_yaw, torch.zeros_like(block_yaw)], -1))
        y = torch.full((n, self.num_blocks, 1), BLOCK_SIZE / 2, device=self.device)
        w.pos[envs[:, None], self.blocks] = torch.cat([spots[:, :self.num_blocks, :1], y, spots[:, :self.num_blocks, 1:]], -1)
        w.quat[envs[:, None], self.blocks] = q_block
        w.pos[envs[:, None], self.zones] = torch.cat([spots[:, self.num_blocks:, :1], torch.full_like(y, 0.001), spots[:, self.num_blocks:, 1:]], -1)
        w.inv_inertia_world = w.compute_inv_inertia_world()
        w.joints.prepare()
        self.steps[envs] = 0
        self.placed[envs] = False
        self.potential = self.compute_potential()

    def compute_potential(self):
        w = self.world
        # (E,) first block not delivered yet (the last one once all are done; its potential is zeroed below)
        cur = (~self.placed).float().argmax(-1)
        e = torch.arange(self.E, device=self.device)
        block = w.pos[e, self.blocks[cur]]
        zone = w.pos[e, self.zones[cur]]
        grip = w.pos[:, self.jaws].mean(1)
        d_grip = torch.linalg.norm(grip - block, dim=-1)
        d_zone = torch.linalg.norm((block - zone)[:, [0, 2]], dim=-1)
        pot = -(GRIP_W * d_grip + PLACE_W * d_zone)
        # finished episodes stop earning shaping reward
        return torch.where(self.placed.all(-1), 0.0, pot)

    def in_zone(self):
        # (E, B) block resting inside its zone
        w = self.world
        d = (w.pos[:, self.blocks] - w.pos[:, self.zones])[..., [0, 2]].abs()
        resting = (w.pos[:, self.blocks, 1] < BLOCK_SIZE * 0.75) & (torch.linalg.norm(w.velocity[:, self.blocks, :3], dim=-1) < 0.05)
        return (d < ZONE_SIZE / 2).all(-1) & resting

    def observe(self):
        # {"rgb": (E, 3 * len(CAMERAS), H, W) uint8 (front, wrist), "proprio": (E, PROPRIO_DIM) float}
        w = self.world
        q = (w.joints.value[:, self.proprio_joints] - self.proprio_lo) / (self.proprio_hi - self.proprio_lo) * 2 - 1
        wheels = w.joints.speeds()[:, self.wheel_joints] / STS3215_SPEED
        # (E, 3 * len(CAMERAS), H, W)
        rgb = torch.cat([cam.render().permute(0, 3, 1, 2) for cam in self.cameras], dim=1)
        return {'rgb': rgb, 'proprio': torch.cat([q, wheels], -1)}

    def step(self, action):
        # action (E, A) in [-1, 1]. returns obs, reward (E,), done (E,), info {"success": (E,) bool}
        w = self.world
        w.joints.target[:, self.act_joints] = self.action_to_targets(action)
        for _ in range(ACTION_REPEAT):
            w.step()
        self.steps += 1
        newly = self.in_zone() & ~self.placed
        self.placed |= newly
        pot = self.compute_potential()
        reward = pot - self.potential + SUCCESS_REWARD * newly.sum(-1)
        self.potential = pot
        success = self.placed.all(-1)
        # base up axis tipped past 60°: flipped
        up = quat_rotate(w.quat[:, self.base], torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.E, 3))
        flipped = up[:, 1] < 0.5
        done = success | flipped | (self.steps >= self.max_steps)
        info = {'success': success.clone()}
        if done.any():
            self.reset(done)
        return self.observe(), reward, done, info

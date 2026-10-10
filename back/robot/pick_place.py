# LeKiwi pick-and-place: one robot, colored blocks and matching colored target zones on the floor. the robot must
# carry block i into zone i. observations are camera pixels + joint readings (robot axis R = 1).
import math
import torch
from robot.env import RobotEnv, static_box, BLOCK_SIZE, DT, SUBSTEPS
from robot.lekiwi import lekiwi, PREFIX

ZONE_SIZE = 0.14
# block i and zone i share a hue
BLOCK_COLORS = ['#d93b3b', '#3b7fd9', '#3bb34a']
ZONE_COLORS = ['#f2a3a3', '#a3c4f2', '#a8e0ae']
# blocks and zones spawn at this distance range from the robot (m), at least MIN_GAP apart
SPAWN_MIN, SPAWN_MAX = 0.45, 0.95
MIN_GAP = 0.25
# control steps allowed per block (30 s at 10 Hz)
EPISODE_STEPS_PER_BLOCK = 300
# reward shaping weights: potential = -(GRIP_W * |gripper - block| + PLACE_W * |block - zone|_xz)
GRIP_W = 1.0
PLACE_W = 2.0
SUCCESS_REWARD = 10.0


def scene(num_blocks):
    # scene JSON with the robot at the origin facing +x, blocks and zones in a default (non-random) layout
    ground = static_box('ground', None, [0.0, 0.0, 0.0], '#9a9a9a')
    ground['shape'] = {'type': 'plane'}
    ground['friction'] = 1.0
    bot = lekiwi()
    blocks = []  # [body dict]
    zones = []  # [body dict]
    for i in range(num_blocks):
        angle = (i - (num_blocks - 1) / 2) * 0.7
        block = static_box(f'block/{i}', [BLOCK_SIZE] * 3, [0.6 * math.cos(angle), BLOCK_SIZE / 2, 0.6 * math.sin(angle)], BLOCK_COLORS[i])
        block.update(static=False, density=800.0)
        blocks.append(block)
        # zones are thin static markers that nothing collides with
        zones.append(static_box(f'zone/{i}', [ZONE_SIZE, 0.002, ZONE_SIZE], [-0.6 * math.cos(angle), 0.001, 0.6 * math.sin(angle)], ZONE_COLORS[i], False))
    return {'name': 'lekiwi_pick', 'settings': {'dt': DT, 'substeps': SUBSTEPS, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0},
            'bodies': [ground] + bot['bodies'] + blocks + zones, 'joints': bot['joints'], 'cameras': bot['cameras']}


class PickPlaceEnv(RobotEnv):
    # scene_json: use this scene as is (e.g. edited in the editor) instead of a randomized layout. it must contain
    # the lekiwi and bodies "block/0".."block/{num_blocks-1}", "zone/0"..
    def __init__(self, num_envs, device, num_blocks=1, seed=0, scene_json=None):
        super().__init__(scene_json if scene_json is not None else scene(num_blocks), num_envs, device, [PREFIX], seed)
        w = self.world
        self.num_blocks = num_blocks
        self.max_steps = EPISODE_STEPS_PER_BLOCK * num_blocks
        # (B,) body index of block i, zone i
        self.blocks = torch.tensor([w.body_index[f'block/{i}'] for i in range(num_blocks)], device=device)
        self.zones = torch.tensor([w.body_index[f'zone/{i}'] for i in range(num_blocks)], device=device)
        # (E,) control steps taken this episode, (E, B) block already delivered
        self.steps = torch.zeros(self.E, dtype=torch.long, device=device)
        self.placed = torch.zeros(self.E, num_blocks, dtype=torch.bool, device=device)
        # randomize layouts on reset (training), or keep the given scene as is (the live editor)
        self.randomize = scene_json is None
        self.reset(torch.ones(self.E, dtype=torch.bool, device=device))

    def reset(self, mask):
        # re-randomize envs where mask (E,) is True: robot yaw, block and zone positions
        w = self.world
        w.reset(mask)
        n = int(mask.sum())
        if n == 0:
            return
        envs = mask.nonzero().view(-1)
        if self.randomize:
            yaw = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * math.pi
            self.place_robot(envs, 0, torch.zeros(n, 2, device=self.device), yaw)
            # blocks then zones around the robot, all MIN_GAP apart
            k = 2 * self.num_blocks
            spots = torch.zeros(n, 0, 2, device=self.device)
            for _ in range(k):
                r = SPAWN_MIN + (SPAWN_MAX - SPAWN_MIN) * torch.rand(n, generator=self.gen, device=self.device)
                a = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * math.pi
                cand = torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
                ok = (torch.linalg.norm(spots - cand[:, None], dim=-1) > MIN_GAP).all(-1)
                # retry the envs that collided with an earlier spot (rare), until all fit
                while not ok.all():
                    r2 = SPAWN_MIN + (SPAWN_MAX - SPAWN_MIN) * torch.rand(n, generator=self.gen, device=self.device)
                    a2 = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * math.pi
                    cand = torch.where(ok[:, None], cand, torch.stack([r2 * torch.cos(a2), r2 * torch.sin(a2)], -1))
                    ok = (torch.linalg.norm(spots - cand[:, None], dim=-1) > MIN_GAP).all(-1)
                spots = torch.cat([spots, cand[:, None]], 1)
            block_yaw = (torch.rand(n, self.num_blocks, generator=self.gen, device=self.device) * 2 - 1) * math.pi
            q_block = torch.stack([torch.cos(block_yaw / 2), torch.zeros_like(block_yaw), torch.sin(block_yaw / 2), torch.zeros_like(block_yaw)], -1)
            y = torch.full((n, self.num_blocks, 1), BLOCK_SIZE / 2, device=self.device)
            w.pos[envs[:, None], self.blocks] = torch.cat([spots[:, :self.num_blocks, :1], y, spots[:, :self.num_blocks, 1:]], -1)
            w.quat[envs[:, None], self.blocks] = q_block
            w.pos[envs[:, None], self.zones] = torch.cat([spots[:, self.num_blocks:, :1], torch.full_like(y, 0.001), spots[:, self.num_blocks:, 1:]], -1)
            w.inv_inertia_world.copy_(w.compute_inv_inertia_world())
        w.joints.prepare()
        self.steps[envs] = 0
        self.placed[envs] = False
        self.potential = self.compute_potential()

    def compute_potential(self):
        # (E, 1) shaping potential of the block being worked on
        w = self.world
        # (E,) first block not delivered yet (the last one once all are done; its potential is zeroed below)
        cur = (~self.placed).float().argmax(-1)
        e = torch.arange(self.E, device=self.device)
        block = w.pos[e, self.blocks[cur]]
        zone = w.pos[e, self.zones[cur]]
        d_grip = torch.linalg.norm(self.grip()[:, 0] - block, dim=-1)
        d_zone = torch.linalg.norm((block - zone)[:, [0, 2]], dim=-1)
        pot = -(GRIP_W * d_grip + PLACE_W * d_zone)
        # finished episodes stop earning shaping reward
        return torch.where(self.placed.all(-1), 0.0, pot).unsqueeze(-1)

    def in_zone(self):
        # (E, B) block resting inside its zone
        w = self.world
        d = (w.pos[:, self.blocks] - w.pos[:, self.zones])[..., [0, 2]].abs()
        resting = (w.pos[:, self.blocks, 1] < BLOCK_SIZE * 0.75) & (torch.linalg.norm(w.velocity[:, self.blocks, :3], dim=-1) < 0.05)
        return (d < ZONE_SIZE / 2).all(-1) & resting

    def expert_goal(self, e, r, blocks):
        # for the scripted expert: (block index, drop point xz) for robot r in env e, or None when done.
        # blocks: (E, B, 3) numpy block positions
        b = next((i for i in range(self.num_blocks) if not bool(self.placed[e, i])), None)
        if b is None:
            return None
        return b, self.world.pos[e, self.zones[b]][[0, 2]].cpu().numpy()

    def step(self, action):
        # action (E, 1, A) in [-1, 1]. returns obs, reward (E, 1), done (E,), info {"success": (E,) bool}
        self.apply(action)
        self.steps += 1
        newly = self.in_zone() & ~self.placed
        self.placed |= newly
        pot = self.compute_potential()
        reward = pot - self.potential + SUCCESS_REWARD * newly.sum(-1, keepdim=True)
        self.potential = pot
        success = self.placed.all(-1)
        done = success | self.flipped().any(-1) | (self.steps >= self.max_steps)
        info = {'success': success.clone()}
        if done.any():
            self.reset(done)
        return self.observe(), reward, done, info

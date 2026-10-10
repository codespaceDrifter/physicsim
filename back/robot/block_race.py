# Block race: two LeKiwis in a walled arena, yellow blocks scattered in the middle, a home zone at each end.
# score = blocks resting in your home zone; stealing from the other zone is allowed. robot axis R = 2.
import math
import torch
from robot.env import RobotEnv, static_box, BLOCK_SIZE, DT, SUBSTEPS
from robot.lekiwi import lekiwi

# arena floor (x: length, z: width), wall height and thickness
ARENA = (2.4, 1.6)
WALL_H, WALL_T = 0.06, 0.04
NUM_BLOCKS = 6
BLOCK_COLOR = '#e8c53a'
ZONE_SIZE = 0.4
# robot r: id prefix, accent color, home zone color, home zone center x (zones on the arena's short ends)
ROBOTS = [('lekiwi/', '#e8833a', '#f5c49a', -0.95), ('lekiwi2/', '#2bb5a8', '#9adfd8', 0.95)]
# blocks spawn in this box (|x|, |z|), at least BLOCK_GAP apart
SPAWN = (0.5, 0.6)
BLOCK_GAP = 0.12
# round length in control steps (60 s at 10 Hz)
ROUND_STEPS = 600
# reward: SCORE_REWARD per block of (my score - their score) change, plus shaping toward the best block
SCORE_REWARD = 10.0
GRIP_W = 1.0
HOME_W = 2.0


def scene():
    # scene JSON: arena, both robots in front of their zones facing each other, blocks in a default layout
    L, W = ARENA
    ground = static_box('ground', None, [0.0, 0.0, 0.0], '#9a9a9a')
    ground['shape'] = {'type': 'plane'}
    ground['friction'] = 1.0
    walls = [static_box('wall/north', [L + 2 * WALL_T, WALL_H, WALL_T], [0.0, WALL_H / 2, -(W + WALL_T) / 2], '#5a5f66'),
             static_box('wall/south', [L + 2 * WALL_T, WALL_H, WALL_T], [0.0, WALL_H / 2, (W + WALL_T) / 2], '#5a5f66'),
             static_box('wall/west', [WALL_T, WALL_H, W], [-(L + WALL_T) / 2, WALL_H / 2, 0.0], '#5a5f66'),
             static_box('wall/east', [WALL_T, WALL_H, W], [(L + WALL_T) / 2, WALL_H / 2, 0.0], '#5a5f66')]
    bodies = [ground] + walls
    joints = []  # [joint dict]
    cameras = []  # [camera dict]
    for r, (prefix, accent, zone_color, zone_x) in enumerate(ROBOTS):
        bot = lekiwi(prefix, group=r + 1, accent=accent)
        # robot 0 starts at x = -0.7 facing +x, robot 1 at x = +0.7 facing -x (turned 180° about y)
        x0 = math.copysign(0.7, zone_x)
        for b in bot['bodies']:
            px, py, pz = b['position']
            b['position'] = [x0 - px, py, -pz] if zone_x > 0 else [x0 + px, py, pz]
            if zone_x > 0:
                w_, x_, y_, z_ = b['rotation']
                # 180° about y: [0, 0, 1, 0] ⊗ q
                b['rotation'] = [-y_, z_, w_, -x_]
        for j in bot['joints']:
            ax, ay, az = j['anchor']
            j['anchor'] = [x0 - ax, ay, -az] if zone_x > 0 else [x0 + ax, ay, az]
            dx, dy, dz = j['axis']
            j['axis'] = [-dx, dy, -dz] if zone_x > 0 else [dx, dy, dz]
        bodies += bot['bodies']
        joints += bot['joints']
        cameras += bot['cameras']
        bodies.append(static_box(f'home/{r}', [ZONE_SIZE, 0.002, ZONE_SIZE], [zone_x, 0.001, 0.0], zone_color, False))
    for i in range(NUM_BLOCKS):
        block = static_box(f'block/{i}', [BLOCK_SIZE] * 3, [(i % 3 - 1) * 0.3, BLOCK_SIZE / 2, (i // 3 - 0.5) * 0.5], BLOCK_COLOR)
        block.update(static=False, density=800.0)
        bodies.append(block)
    return {'name': 'block_race', 'settings': {'dt': DT, 'substeps': SUBSTEPS, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0,
                                               'max_contacts': 128},
            'bodies': bodies, 'joints': joints, 'cameras': cameras}


class BlockRaceEnv(RobotEnv):
    # scene_json: use this scene as is (the live editor) instead of a randomized layout
    def __init__(self, num_envs, device, seed=0, scene_json=None):
        super().__init__(scene_json if scene_json is not None else scene(), num_envs, device, [p for p, a, z, x in ROBOTS], seed)
        w = self.world
        # (B,) block bodies, (R,) home zone bodies
        self.blocks = torch.tensor([w.body_index[f'block/{i}'] for i in range(NUM_BLOCKS)], device=device)
        self.homes = torch.tensor([w.body_index[f'home/{r}'] for r in range(self.R)], device=device)
        self.max_steps = ROUND_STEPS
        # (E,) control steps taken this round
        self.steps = torch.zeros(self.E, dtype=torch.long, device=device)
        self.randomize = scene_json is None
        self.reset(torch.ones(self.E, dtype=torch.bool, device=device))

    def reset(self, mask):
        # re-randomize envs where mask (E,) is True: robot spots in front of their zones, block positions and yaw
        w = self.world
        w.reset(mask)
        n = int(mask.sum())
        if n == 0:
            return
        envs = mask.nonzero().view(-1)
        if self.randomize:
            for r, (prefix, accent, zone_color, zone_x) in enumerate(ROBOTS):
                z = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * 0.4
                xz = torch.stack([torch.full_like(z, math.copysign(0.7, zone_x)), z], -1)
                yaw = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * 0.4
                self.place_robot(envs, r, xz, yaw)
            lo = torch.tensor([-SPAWN[0], -SPAWN[1]], device=self.device)
            spots = self.sample_spots(n, NUM_BLOCKS, lo, -lo, BLOCK_GAP, torch.zeros(n, 0, 2, device=self.device))
            yaw = (torch.rand(n, NUM_BLOCKS, generator=self.gen, device=self.device) * 2 - 1) * math.pi
            y = torch.full((n, NUM_BLOCKS, 1), BLOCK_SIZE / 2, device=self.device)
            w.pos[envs[:, None], self.blocks] = torch.cat([spots[..., :1], y, spots[..., 1:]], -1)
            w.quat[envs[:, None], self.blocks] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.zeros_like(yaw)], -1)
            w.inv_inertia_world.copy_(w.compute_inv_inertia_world())
        w.joints.prepare()
        self.steps[envs] = 0
        self.score_now = self.score()
        self.potential = self.compute_potential()

    def in_home(self):
        # (E, R, B) block resting inside robot r's home zone
        w = self.world
        d = (w.pos[:, None, self.blocks] - w.pos[:, self.homes, None])[..., [0, 2]].abs()
        resting = (w.pos[:, self.blocks, 1] < BLOCK_SIZE * 0.75) & (torch.linalg.norm(w.velocity[:, self.blocks, :3], dim=-1) < 0.05)
        return (d < ZONE_SIZE / 2).all(-1) & resting[:, None]

    def score(self):
        # (E, R) blocks in each robot's home zone
        return self.in_home().sum(-1)

    def compute_potential(self):
        # (E, R) shaping: -(grip -> block + 2 * block -> home) for the best block not already home
        w = self.world
        blocks = w.pos[:, self.blocks]
        grip = self.grip()
        home = w.pos[:, self.homes]
        # (E, R, B)
        cost = GRIP_W * torch.linalg.norm(grip[:, :, None] - blocks[:, None], dim=-1) \
            + HOME_W * torch.linalg.norm((blocks[:, None] - home[:, :, None])[..., [0, 2]], dim=-1)
        cost = torch.where(self.in_home(), float('inf'), cost)
        best = cost.min(-1).values
        return torch.where(torch.isinf(best), 0.0, -best)

    def expert_goal(self, e, r, blocks):
        # for the scripted expert: nearest block not already home and not held by the other robot, dropped at its
        # home zone center (+ a little jitter so blocks don't pile up). blocks: (E, B, 3) numpy
        w = self.world
        home = w.pos[e, self.homes[r]].cpu().numpy()
        base = w.pos[e, self.base[r]].cpu().numpy()
        grips = self.grip()[e].cpu().numpy()
        best, best_d = None, float('inf')
        for b in range(NUM_BLOCKS):
            p = blocks[e, b]
            if abs(p[0] - home[0]) < ZONE_SIZE / 2 and abs(p[2] - home[2]) < ZONE_SIZE / 2 and p[1] < BLOCK_SIZE:
                continue
            held = any(o != r and p[1] > BLOCK_SIZE and ((p - grips[o]) ** 2).sum() < 0.06 ** 2 for o in range(self.R))
            if held:
                continue
            d = ((p[[0, 2]] - base[[0, 2]]) ** 2).sum()
            if d < best_d:
                best, best_d = b, d
        if best is None:
            return None
        jitter = (torch.rand(2, generator=self.gen, device=self.device).cpu().numpy() * 2 - 1) * 0.08
        return best, home[[0, 2]] + jitter

    def step(self, action):
        # action (E, R, A) in [-1, 1]. returns obs, reward (E, R), done (E,), info {"score": (E, R)} (final score on done)
        self.apply(action)
        self.steps += 1
        score = self.score()
        gain = score - self.score_now
        # zero-sum score reward: my gain minus theirs (R = 2)
        reward = SCORE_REWARD * (gain - gain.flip(-1)).float()
        pot = self.compute_potential()
        reward = reward + pot - self.potential
        self.potential = pot
        self.score_now = score
        done = self.flipped().any(-1) | (self.steps >= self.max_steps)
        info = {'score': score.clone()}
        if done.any():
            self.reset(done)
        return self.observe(), reward, done, info

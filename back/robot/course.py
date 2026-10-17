# The course: one arena (2.4 x 1.6 m, walls, a home zone at each end, six pink blocks, two LeKiwis) and five stages.
# every env runs one stage per episode, sampled at reset:
#   drive     each robot drives its base onto its goal pose (goal/r marker on the floor: pad + dot on its front edge)
#   pick      each robot lifts a block 5 cm off the floor
#   bring     each robot brings a block into its home zone
#   race      more blocks in your home zone than the other robot after 60 s wins; stealing allowed (zero-sum)
#   speedrun  robot 0 alone (robot 1 parked outside the arena) brings all blocks home, as fast as possible.
#             subcourses (reverse curriculum): an episode may start with k blocks already home (home_probs), so the
#             end of a speedrun can be practiced before the whole of it
# both robots exist in every stage (same world structure in every env, so all stages batch together): robot axis R = 2.
import math
import torch
from robot.env import RobotEnv, static_box, BLOCK_SIZE, DT, SUBSTEPS
from robot.lekiwi import lekiwi
from physics.math3d import wrap_angle, quat_to_matrix

STAGES = ['drive', 'pick', 'bring', 'race', 'speedrun']
# episode length per stage in control steps (10 Hz)
STAGE_STEPS = {'drive': 150, 'pick': 300, 'bring': 450, 'race': 600, 'speedrun': 1200}
# arena floor (x: length, z: width), wall height and thickness
ARENA = (2.4, 1.6)
WALL_H, WALL_T = 0.06, 0.04
NUM_BLOCKS = 6
BLOCK_COLOR = '#ef6fb0'
ZONE_SIZE = 0.4
# robot r: id prefix, accent color, home zone / goal pad color, home zone center x (zones on the arena's short ends)
ROBOTS = [('lekiwi/', '#e8833a', '#f5c49a', -0.95), ('lekiwi2/', '#2bb5a8', '#9adfd8', 0.95)]
# race start: robot r's base at x = ±START_X facing the arena center, z within ±START_Z, yaw within ±START_YAW
START_X, START_Z, START_YAW = 0.7, 0.4, 0.4
# blocks spawn in this box (|x|, |z|), at least BLOCK_GAP apart
SPAWN = (0.5, 0.6)
BLOCK_GAP = 0.12
# drive stage: robot starts and goals anywhere in this box (|x|, |z|) (keeps the base clear of the walls), at least
# DRIVE_GAP from each other and from blocks
DRIVE_BOX = (0.95, 0.55)
DRIVE_GAP = 0.3
GOAL_PAD = 0.14
# speedrun subcourses: blocks already home sit in this part of the zone (offset from its center: toward the wall,
# sideways), clear of the robot starting in front of it, at least HOME_GAP apart
HOME_SPOTS = ((-0.06, -0.14), (0.14, 0.14))
HOME_GAP = 0.07
# where unused things go: goal markers under the floor, robot 1 outside the south wall in the speedrun
HIDDEN_Y = -0.5
PARK = (0.0, 1.4)
# success: drive within GOAL_DIST and GOAL_ANGLE of the goal pose; pick a block LIFT above its resting height,
# within HOLD of the grip point
GOAL_DIST, GOAL_ANGLE = 0.05, math.radians(10)
LIFT, HOLD = 0.05, 0.04
# reward: potential shaping (cost weights below) + SUCCESS_REWARD when a robot first succeeds; race / speedrun pay
# SCORE_REWARD per block of score change instead, the speedrun also costs TIME_PENALTY per step
SUCCESS_REWARD = 10.0
SCORE_REWARD = 10.0
TIME_PENALTY = 0.02
HEADING_W = 0.2
GRIP_W = 1.0
LIFT_W = 2.0
HOME_W = 2.0
# teacher state vector per robot (see state()): proprio 9 + base velocity 3 + grip 3 + blocks 6 * 5 + homes 4
# + other robot 4 + command 8 + time left 1
STATE_DIM = 9 + 3 + 3 + NUM_BLOCKS * 5 + 4 + 4 + len(STAGES) + 3 + 1
# drive goals lie on the grid the command text can say exactly: offsets in multiples of GOAL_STEP m up to GOAL_RANGE,
# turns in multiples of TURN_STEP degrees
GOAL_STEP, GOAL_RANGE, TURN_STEP = 0.1, 1.0, 15
# fixed command text per stage (drive's is built from its goal, see command())
COMMANDS = {'pick': 'pick up a block', 'bring': 'bring a block home', 'race': 'win the race',
            'speedrun': 'bring all blocks home, fast'}


def scene(stage):
    # scene JSON for the editor: the stage's default layout. training envs use the same scene and re-randomize it
    assert stage in STAGES, f'unknown stage {stage}'
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
    for r, (prefix, accent, pad_color, zone_x) in enumerate(ROBOTS):
        bot = lekiwi(prefix, group=r + 1, accent=accent)
        # robot 0 faces +x, robot 1 faces -x (turned 180° about y); robot 1 parked south of the arena in the speedrun
        x0, z0 = (PARK[0], PARK[1]) if stage == 'speedrun' and r == 1 else (math.copysign(START_X, zone_x), 0.0)
        for b in bot['bodies']:
            px, py, pz = b['position']
            b['position'] = [x0 - px, py, z0 - pz] if zone_x > 0 else [x0 + px, py, z0 + pz]
            if zone_x > 0:
                w_, x_, y_, z_ = b['rotation']
                # 180° about y: [0, 0, 1, 0] ⊗ q
                b['rotation'] = [-y_, z_, w_, -x_]
        for j in bot['joints']:
            ax, ay, az = j['anchor']
            j['anchor'] = [x0 - ax, ay, z0 - az] if zone_x > 0 else [x0 + ax, ay, z0 + az]
            dx, dy, dz = j['axis']
            j['axis'] = [-dx, dy, -dz] if zone_x > 0 else [dx, dy, dz]
        bodies += bot['bodies']
        joints += bot['joints']
        cameras += bot['cameras']
        bodies.append(static_box(f'home/{r}', [ZONE_SIZE, 0.002, ZONE_SIZE], [zone_x, 0.001, 0.0], pad_color, False))
        # goal pose marker: a pad, and a dot on its front edge showing which way the robot must face. default drive
        # goals: robot 0 ends up 0.5 m ahead and 0.3 m to the right turned 90° right, robot 1 the mirror image
        gx, gy = (x0 - math.copysign(0.5, zone_x), 0.002) if stage == 'drive' else (0.0, HIDDEN_Y)
        gz = 0.3 if zone_x < 0 else -0.3
        # pad yaw: robot 0 faces +z (right of +x), robot 1 faces -z
        yaw = -math.pi / 2 if zone_x < 0 else math.pi / 2
        q = [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0]
        pad = static_box(f'goal/{r}', [GOAL_PAD, 0.002, GOAL_PAD], [gx, gy, gz], pad_color, False)
        pad['rotation'] = q
        dot = static_box(f'goal/{r}/front', [0.03, 0.003, 0.03], [gx + GOAL_PAD / 2 * math.cos(yaw), gy + 0.001, gz - GOAL_PAD / 2 * math.sin(yaw)], accent, False)
        dot['rotation'] = q
        bodies += [pad, dot]
    for i in range(NUM_BLOCKS):
        block = static_box(f'block/{i}', [BLOCK_SIZE] * 3, [(i % 3 - 1) * 0.3, BLOCK_SIZE / 2, (i // 3 - 0.5) * 0.5], BLOCK_COLOR)
        block.update(static=False, density=800.0)
        bodies.append(block)
    return {'name': f'course_{STAGES.index(stage) + 1}_{stage}', 'settings': {'dt': DT, 'substeps': SUBSTEPS, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0,
                                                   'max_contacts': 128, 'stage': stage},
            'bodies': bodies, 'joints': joints, 'cameras': cameras}


class CourseEnv(RobotEnv):
    # stage_probs: (S,) probability of each stage at reset, in STAGES order (settable later, e.g. by a curriculum).
    # scene_json: use this scene as is (the live editor) instead of randomized layouts; its settings "stage" is the stage
    def __init__(self, num_envs, device, stage_probs=None, seed=0, scene_json=None):
        assert (stage_probs is None) != (scene_json is None), 'give stage_probs (training) or scene_json (editor)'
        sc = scene_json if scene_json is not None else scene('race')
        super().__init__(sc, num_envs, device, [p for p, a, c, x in ROBOTS], seed)
        w = self.world
        if scene_json is not None:
            stage_probs = [float(s == sc['settings']['stage']) for s in STAGES]
        assert len(stage_probs) == len(STAGES) and sum(stage_probs) > 0
        self.stage_probs = torch.tensor(stage_probs, dtype=torch.float32, device=device)
        # (NUM_BLOCKS,) speedrun subcourses: probability of starting with k = 0 .. NUM_BLOCKS - 1 blocks already home
        # (settable later, e.g. by a reverse curriculum). default: the full speedrun
        self.home_probs = torch.zeros(NUM_BLOCKS, device=device)
        self.home_probs[0] = 1.0
        self.randomize = scene_json is None
        # (B,) block bodies, (R,) home zone bodies, (R,) goal pads, (R,) goal front dots
        self.blocks = torch.tensor([w.body_index[f'block/{i}'] for i in range(NUM_BLOCKS)], device=device)
        self.homes = torch.tensor([w.body_index[f'home/{r}'] for r in range(self.R)], device=device)
        self.goals = torch.tensor([w.body_index[f'goal/{r}'] for r in range(self.R)], device=device)
        self.goal_dots = torch.tensor([w.body_index[f'goal/{r}/front'] for r in range(self.R)], device=device)
        # (S,) episode length of each stage
        self.stage_steps = torch.tensor([STAGE_STEPS[s] for s in STAGES], device=device)
        # (R, A) action that holds a robot still in its scene pose (wheels stopped, arm at its initial targets)
        self.rest_action = self.targets_to_action(w.joints.init_target[self.act_joints])
        # per env: (E,) stage index, (E,) control steps taken, (E, R) robot has succeeded this episode,
        # (E, R) drive goal (forward, right, turn) relative to the robot's start pose, for the command text
        self.stage = torch.zeros(self.E, dtype=torch.long, device=device)
        self.steps = torch.zeros(self.E, dtype=torch.long, device=device)
        self.achieved = torch.zeros(self.E, self.R, dtype=torch.bool, device=device)
        self.goal_start = torch.zeros(self.E, self.R, 3, device=device)
        # (E,) blocks the episode started with already home (speedrun subcourses, else 0)
        self.start_home = torch.zeros(self.E, dtype=torch.long, device=device)
        self.reset(torch.ones(self.E, dtype=torch.bool, device=device))

    def is_stage(self, name):
        # (E,) env runs this stage
        return self.stage == STAGES.index(name)

    def active(self):
        # (E, R) robot takes part: all but robot 1 in the speedrun
        a = torch.ones(self.E, self.R, dtype=torch.bool, device=self.device)
        a[:, 1] = ~self.is_stage('speedrun')
        return a

    def reset(self, mask):
        # new episode in envs where mask (E,) is True: sample a stage, then (training) a random layout for it
        w = self.world
        w.reset(mask)
        n = int(mask.sum())
        if n == 0:
            return
        envs = mask.nonzero().view(-1)
        if self.randomize:
            self.stage[envs] = torch.multinomial(self.stage_probs, n, replacement=True, generator=self.gen)
            lo = torch.tensor([-SPAWN[0], -SPAWN[1]], device=self.device)
            blocks = self.sample_spots(n, NUM_BLOCKS, lo, -lo, BLOCK_GAP, torch.zeros(n, 0, 2, device=self.device))
            yaw = (torch.rand(n, NUM_BLOCKS, generator=self.gen, device=self.device) * 2 - 1) * math.pi
            w.pos[envs[:, None], self.blocks] = torch.cat([blocks[..., :1], torch.full((n, NUM_BLOCKS, 1), BLOCK_SIZE / 2, device=self.device), blocks[..., 1:]], -1)
            w.quat[envs[:, None], self.blocks] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.zeros_like(yaw)], -1)
            stage = self.stage[envs]
            drive = stage == STAGES.index('drive')
            # race start for every robot, then drive envs get random starts + grid goals, speedrun envs park robot 1
            for r, (prefix, accent, pad_color, zone_x) in enumerate(ROBOTS):
                z = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * START_Z
                xz = torch.stack([torch.full_like(z, math.copysign(START_X, zone_x)), z], -1)
                yaw = (torch.rand(n, generator=self.gen, device=self.device) * 2 - 1) * START_YAW
                self.place_robot(envs, r, xz, yaw)
            if drive.any():
                d = envs[drive]
                m = len(d)
                lo = torch.tensor([-DRIVE_BOX[0], -DRIVE_BOX[1]], device=self.device)
                # (m, R, 2) robot starts, clear of this env's blocks and of each other
                starts = self.sample_spots(m, self.R, lo, -lo, DRIVE_GAP, blocks[drive])
                yaw = (torch.rand(m, self.R, generator=self.gen, device=self.device) * 2 - 1) * math.pi
                for r in range(self.R):
                    self.place_robot(d, r, starts[:, r], yaw[:, r])
                # (m, R, 2) start xz, (m, R) start yaw as the robot's frame reads it (robot 1's scene pose is turned 180°)
                sxz, syaw = (x[d] for x in self.base_frame())
                # (m, B + R, 2) points every goal keeps DRIVE_GAP from: blocks, robot starts, then goals as they are placed
                taken = torch.cat([blocks[drive], starts], 1)
                # (K, 2) every (forward, right) offset on the grid
                steps = torch.arange(-round(GOAL_RANGE / GOAL_STEP), round(GOAL_RANGE / GOAL_STEP) + 1, device=self.device) * GOAL_STEP
                offs = torch.cartesian_prod(steps, steps)
                for r in range(self.R):
                    # goal = start + a grid offset, picked uniformly among those inside DRIVE_BOX and clear of everything
                    # taken (envs too crowded for any: inside DRIVE_BOX and clear of the robot's own start), turned by a
                    # grid turn. forward = (cos yaw, -sin yaw), right = (sin yaw, cos yaw) in world xz
                    c, s_ = torch.cos(syaw[:, r, None]), torch.sin(syaw[:, r, None])
                    # (m, K, 2)
                    cand = sxz[:, r, None] + torch.stack([offs[:, 0] * c + offs[:, 1] * s_, -offs[:, 0] * s_ + offs[:, 1] * c], -1)
                    inside = (cand.abs() <= -lo).all(-1)
                    ok = inside & (torch.linalg.norm(taken[:, None] - cand[:, :, None], dim=-1) > DRIVE_GAP).all(-1)
                    fallback = inside & (torch.linalg.norm(cand - sxz[:, r, None], dim=-1) > DRIVE_GAP)
                    ok = torch.where(ok.any(-1, keepdim=True), ok, fallback)
                    assert ok.any(-1).all(), 'no drive goal fits in the arena'
                    pick = torch.multinomial(ok.float(), 1, generator=self.gen)[:, 0]
                    goal = cand[torch.arange(m, device=self.device), pick]
                    turn = torch.randint(-180 // TURN_STEP, 180 // TURN_STEP, (m,), generator=self.gen, device=self.device) * math.radians(TURN_STEP)
                    gyaw = syaw[:, r] + turn
                    taken = torch.cat([taken, goal[:, None]], 1)
                    q = torch.stack([torch.cos(gyaw / 2), torch.zeros_like(gyaw), torch.sin(gyaw / 2), torch.zeros_like(gyaw)], -1)
                    gx, gz = goal[:, 0], goal[:, 1]
                    w.pos[d, self.goals[r]] = torch.stack([gx, torch.full_like(gx, 0.002), gz], -1)
                    w.pos[d, self.goal_dots[r]] = torch.stack([gx + GOAL_PAD / 2 * torch.cos(gyaw), torch.full_like(gx, 0.003), gz - GOAL_PAD / 2 * torch.sin(gyaw)], -1)
                    w.quat[d, self.goals[r]] = q
                    w.quat[d, self.goal_dots[r]] = q
            self.start_home[envs] = 0
            speedrun = envs[stage == STAGES.index('speedrun')]
            if len(speedrun):
                m = len(speedrun)
                park = torch.tensor(PARK, device=self.device).expand(m, 2)
                self.place_robot(speedrun, 1, park, torch.zeros(m, device=self.device))
                # subcourse: the first k blocks start resting in robot 0's home zone
                k = torch.multinomial(self.home_probs, m, replacement=True, generator=self.gen)
                self.start_home[speedrun] = k
                # (m, B - 1, 2) offsets (toward the wall, sideways) -> world xz
                off = self.sample_spots(m, NUM_BLOCKS - 1, torch.tensor(HOME_SPOTS[0], device=self.device),
                                        torch.tensor(HOME_SPOTS[1], device=self.device), HOME_GAP, torch.zeros(m, 0, 2, device=self.device))
                away = math.copysign(1.0, ROBOTS[0][3])
                home = w.pos[speedrun, self.homes[0]][:, [0, 2]]
                xz = home[:, None] + torch.stack([away * off[..., 0], off[..., 1]], -1)
                for j in range(NUM_BLOCKS - 1):
                    moved = (k > j)[:, None]
                    p = w.pos[speedrun, self.blocks[j]]
                    w.pos[speedrun, self.blocks[j]] = torch.where(moved, torch.stack([xz[:, j, 0], p[:, 1], xz[:, j, 1]], -1), p)
            w.inv_inertia_world.copy_(w.compute_inv_inertia_world())
        else:
            self.stage[envs] = int(self.stage_probs.argmax())
        w.joints.prepare()
        self.steps[envs] = 0
        self.achieved[envs] = False
        # drive command: goal relative to the start pose (other stages: zeros, unused)
        self.goal_start[envs] = torch.where(self.is_stage('drive')[envs, None, None], self.goal_relative()[envs], 0.0)
        self.score_now = self.score()
        self.potential = self.compute_potential()

    def copy_envs(self, src, dst):
        # make envs dst (n,) exact copies of envs src (n,), right after a reset of both (GRPO: one start, a group of
        # rollouts). a fresh reset zeroed the solver's warm-start buffers, so the bodies, joint targets and episode
        # bookkeeping are the whole state
        w = self.world
        for x in (w.pos, w.quat, w.velocity, w.inv_inertia_world, w.joints.target, self.stage, self.steps, self.achieved, self.goal_start, self.start_home):
            x[dst] = x[src]
        w.joints.prepare()
        self.score_now = self.score()
        self.potential = self.compute_potential()

    def base_frame(self):
        # (E, R, 2) base xz, (E, R) base yaw about +y (positive = turned left), robot forward = local +x
        w = self.world
        q = w.quat[:, self.base]
        qw, qx, qy, qz = q.unbind(-1)
        # forward (+x local) in world: (1 - 2(y² + z²), 2(xy + wz), 2(xz - wy)); yaw from its xz components
        yaw = torch.atan2(-2 * (qx * qz - qw * qy), 1 - 2 * (qy * qy + qz * qz))
        return w.pos[:, self.base][..., [0, 2]], yaw

    def to_local(self, p, xz, yaw):
        # world xz points p (E, R, ..., 2) -> (forward, right) offsets from the robot at xz (E, R, 2) with yaw (E, R)
        extra = p.dim() - 3
        d = p - xz.view(*xz.shape[:2], *([1] * extra), 2)
        c, s = torch.cos(yaw).view(*yaw.shape, *([1] * extra)), torch.sin(yaw).view(*yaw.shape, *([1] * extra))
        # forward = (cos yaw, -sin yaw), right = (sin yaw, cos yaw) in world xz
        return torch.stack([d[..., 0] * c - d[..., 1] * s, d[..., 0] * s + d[..., 1] * c], -1)

    def goal_yaw(self):
        # (E, R) goal pad yaw
        q = self.world.quat[:, self.goals]
        return 2 * torch.atan2(q[..., 2], q[..., 0])

    def goal_relative(self):
        # (E, R, 3) drive goal seen from the robot: (forward, right, turn needed, positive = left)
        xz, yaw = self.base_frame()
        goal = self.to_local(self.world.pos[:, self.goals][..., [0, 2]], xz, yaw)
        return torch.cat([goal, wrap_angle(self.goal_yaw() - yaw)[..., None]], -1)

    def in_home(self):
        # (E, R, B) block resting inside robot r's home zone
        w = self.world
        d = (w.pos[:, None, self.blocks] - w.pos[:, self.homes, None])[..., [0, 2]].abs()
        resting = (w.pos[:, self.blocks, 1] < BLOCK_SIZE * 0.75) & (torch.linalg.norm(w.velocity[:, self.blocks, :3], dim=-1) < 0.05)
        return (d < ZONE_SIZE / 2).all(-1) & resting[:, None]

    def score(self):
        # (E, R) blocks in each robot's home zone
        return self.in_home().sum(-1)

    def lifted(self):
        # (E, R, B) block b lifted LIFT off the floor and within HOLD of robot r's grip point
        w = self.world
        blocks = w.pos[:, self.blocks]
        up = blocks[..., 1] > BLOCK_SIZE / 2 + LIFT
        near = torch.linalg.norm(self.grip()[:, :, None] - blocks[:, None], dim=-1) < HOLD
        return up[:, None] & near

    def success_now(self):
        # (E, R) the robot's stage goal holds right now (race: leading; inactive robots: never)
        g = self.goal_relative()
        drive = (torch.linalg.norm(g[..., :2], dim=-1) < GOAL_DIST) & (g[..., 2].abs() < GOAL_ANGLE)
        score = self.score()
        ok = torch.stack([drive, self.lifted().any(-1), score > 0, score > score.flip(-1), score == NUM_BLOCKS], 1)
        # (E, S, R) -> (E, R) for each env's stage
        return ok[torch.arange(self.E, device=self.device), self.stage] & self.active()

    def compute_potential(self):
        # (E, R) shaping potential (minus a cost) of each env's stage; 0 once the robot succeeded or sits out
        w = self.world
        blocks = w.pos[:, self.blocks]
        grip = self.grip()
        g = self.goal_relative()
        drive = torch.linalg.norm(g[..., :2], dim=-1) + HEADING_W * g[..., 2].abs()
        # (E, R, B) grip -> block distance
        reach = torch.linalg.norm(grip[:, :, None] - blocks[:, None], dim=-1)
        pick = (GRIP_W * reach + LIFT_W * (BLOCK_SIZE / 2 + LIFT - blocks[:, None, :, 1]).clamp(min=0)).min(-1).values
        # grip -> block + block -> home for the best block not already home
        home = w.pos[:, self.homes]
        deliver = GRIP_W * reach + HOME_W * torch.linalg.norm((blocks[:, None] - home[:, :, None])[..., [0, 2]], dim=-1)
        deliver = torch.where(self.in_home(), float('inf'), deliver).min(-1).values
        deliver = torch.where(torch.isinf(deliver), 0.0, deliver)
        cost = torch.stack([drive, pick, deliver, deliver, deliver], 1)[torch.arange(self.E, device=self.device), self.stage]
        return torch.where(self.achieved | ~self.active(), 0.0, -cost)

    def block_yaw(self):
        # (E, B) yaw of each block about +y, mod 90° (cubes): the yaw of whichever body axis lies most horizontal, so a
        # block that tumbled onto a side face reads right too (the yaw of the quaternion alone assumes it is upright)
        # (E, B, 3, 3) columns = body axes in world
        axes = quat_to_matrix(self.world.quat[:, self.blocks])
        # (E, B, 3) horizontal length of each body axis -> the most horizontal one (E, B, 3)
        flat = torch.linalg.norm(axes[..., [0, 2], :], dim=-2)
        a = axes.gather(-1, flat.argmax(-1)[..., None, None].expand(*flat.shape[:2], 3, 1))[..., 0]
        # an axis turned by yaw ψ about +y points along (cos ψ, 0, -sin ψ)
        return torch.atan2(-a[..., 2], a[..., 0])

    def state(self):
        # (E, R, STATE_DIM) privileged state for the teacher, everything in the robot's own frame (forward, right):
        #   proprio (9) | base velocity: forward, right, yaw rate (3) | grip point: forward, right, height (3)
        #   | per block: forward, right, height, sin 4θ, cos 4θ (30; θ = block yaw relative to the robot, cubes
        #   repeat every 90°) | own home, other home: forward, right (4) | other robot: forward, right, sin, cos of
        #   its relative heading (4) | stage one-hot (5) | drive goal: forward, right, turn (3, zeros in other stages)
        #   | time left in the episode, 1 -> 0 (1). blocks are sorted nearest first (fixed slots would make the network
        #   learn every skill once per slot). no previous action: cloning the expert with it learns to copy its last
        #   action (the "copycat" problem, de Haan et al. 2019) and grasps far worse
        w = self.world
        E, R = self.E, self.R
        xz, yaw = self.base_frame()
        proprio = self.proprio()
        vel = w.velocity[:, self.base]
        # rotate the velocity into the robot frame (add xz so to_local's subtraction cancels)
        v = self.to_local(vel[..., [0, 2]] + xz, xz, yaw)
        grip = self.grip()
        grip_l = torch.cat([self.to_local(grip[..., [0, 2]], xz, yaw), grip[..., 1:2]], -1)
        blocks = w.pos[:, self.blocks]
        # (E, R, B, 2) blocks in each robot's frame
        blocks_l = self.to_local(blocks[:, None, :, [0, 2]].expand(E, R, NUM_BLOCKS, 2), xz, yaw)
        theta = self.block_yaw()[:, None] - yaw[..., None]
        blocks_f = torch.cat([blocks_l, blocks[:, None, :, 1:2].expand(E, R, NUM_BLOCKS, 1),
                              torch.sin(4 * theta)[..., None], torch.cos(4 * theta)[..., None]], -1)
        # (E, R, B) block order by distance from the robot
        order = torch.linalg.norm(blocks_l, dim=-1).argsort(-1)
        blocks_f = blocks_f.gather(2, order[..., None].expand(E, R, NUM_BLOCKS, 5))
        # homes ordered (own, other): robot r's own home is home r
        home = w.pos[:, self.homes][..., [0, 2]]
        homes = torch.stack([home, home.flip(1)], 2)
        homes_l = self.to_local(homes, xz, yaw)
        other_l = self.to_local(xz.flip(1), xz, yaw)
        other_yaw = yaw.flip(1) - yaw
        stage = torch.nn.functional.one_hot(self.stage, len(STAGES)).float()[:, None].expand(E, R, len(STAGES))
        goal = torch.where(self.is_stage('drive')[:, None, None], self.goal_relative(), 0.0)
        # (E,) -> (E, R, 1)
        time_left = (1 - self.steps / self.stage_steps[self.stage])[:, None, None].expand(E, R, 1)
        # (E, R, STATE_DIM)
        out = torch.cat([proprio, v, vel[..., 4:5], grip_l, blocks_f.flatten(2), homes_l.flatten(2), other_l,
                         torch.sin(other_yaw)[..., None], torch.cos(other_yaw)[..., None], stage, goal, time_left], -1)
        assert out.shape[-1] == STATE_DIM
        return out

    def command(self, e, r):
        # robot r's command in env e as text (what the vision-language student is told)
        stage = STAGES[int(self.stage[e])]
        if not bool(self.active()[e, r]):
            return 'stay parked'
        if stage != 'drive':
            return COMMANDS[stage]
        f, right, turn = self.goal_start[e, r].tolist()
        return (f'move {abs(f):.1f} m {"forward" if f >= 0 else "back"} and {abs(right):.1f} m {"right" if right >= 0 else "left"}, '
                f'then turn {abs(math.degrees(turn)):.0f}° {"left" if turn >= 0 else "right"}')

    def step(self, action):
        # action (E, R, A) in [-1, 1] (inactive robots are held still). returns reward (E, R), done (E,),
        # info {"success": (E, R) bool, "score": (E, R), "stage": (E,), "start_home": (E,)} (as of the episode's end where done).
        # done envs are already reset; read state() / observe() after the step (rendering is the caller's choice)
        active = self.active()
        self.apply(torch.where(active[..., None], action, self.rest_action))
        self.steps += 1
        newly = self.success_now() & ~self.achieved & ~self.is_stage('race')[:, None]
        self.achieved |= newly
        score = self.score()
        gain = score - self.score_now
        self.score_now = score
        pot = self.compute_potential()
        reward = pot - self.potential + SUCCESS_REWARD * newly
        self.potential = pot
        race = self.is_stage('race')[:, None]
        speedrun = self.is_stage('speedrun')[:, None]
        # race: zero-sum score change (my gain minus theirs); speedrun: score change minus a time penalty
        reward = reward + torch.where(race, SCORE_REWARD * (gain - gain.flip(-1)).float(), 0.0)
        reward = reward + torch.where(speedrun, SCORE_REWARD * gain.float() - TIME_PENALTY, 0.0)
        reward = torch.where(active, reward, 0.0)
        timeout = self.steps >= self.stage_steps[self.stage]
        done = (self.flipped() & active).any(-1) | timeout | ((self.achieved | ~active).all(-1) & ~race[:, 0])
        # race success is decided at the end: more blocks home than the other robot
        success = torch.where(race, score > score.flip(-1), self.achieved)
        info = {'success': success, 'score': score.clone(), 'stage': self.stage.clone(), 'start_home': self.start_home.clone()}
        if done.any():
            self.reset(done)
        return reward, done, info

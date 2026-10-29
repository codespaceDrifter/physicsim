# The warehouse course: a Unitree G1 learns to move boxes the way warehouse humanoids are paid to (Agility's Digit
# moves totes for GXO), one pass-gated stage at a time (ml/train_humanoid.py unlocks the next stage when the newest
# one passes its test; earlier stages keep being trained). every command is words ("... warehouse" last), the same
# skill words a language model (System 2) writes at deployment.
# stages (one per episode, robot/humanoid.py runs them), with their pass tests (test episodes, see robot/humanoid.py):
#   walk     "walk forward +1.00 left +0.00 turn +0.00": follow a velocity command (robot/humanoid.py).
#            pass: 10 m in 12 s, smoothly (the dash)
#   pick     "pick up the red box": the named box (on the floor) up off the floor.
#            pass: its bottom LIFT_HEIGHT up, held in both hands, for HOLD_STEPS in a row
#   track    "run a lap": one lap of the walled track, the way the robot faces at the start.
#            pass: the lap at TRACK_SPEED or faster without touching a wall
#   carry    "run a lap with the red box": the same lap carrying the named box (it starts in the hands, standing).
#            pass: the lap at CARRY_SPEED or faster, without touching a wall or losing the box
#   deliver  "put the red box on the blue pad": the named box from the floor onto the named pad (its own color's
#            SAME_PAD of the time, another's otherwise: commands name any pair). DELIVER_HELD of its episodes start
#            with the box already in the hands (like carry: a plan "pick up the red box", then "put it on ..." hands
#            the deliver skill a held box).
#            pass: the box resting on the pad, let go for RELEASE_STEPS
#   sort     "sort the boxes by color": every box onto the pad of its color.
#            pass: all of them, before the episode ends
# each stage passes when PASS[stage] of its tries pass.
#
# grasp: a box is squeezed between the palm sides of the two rubber hands (boxes, robot/g1.py GEOM_OVERRIDES) and held
# by friction. checked before training: the G1 reaches a floor box's side faces collision-free with its feet flat
# (pelvis 0.35 m up, pitched 60° forward, center of mass over the feet), and palms commanded 3 cm into a 1.5 kg box
# hold it up (25-30 N of squeeze).
#
# arena: open floor (walk and the box stages) or the track: a loop corridor around a central island box, closed by
# four outer walls with their corners cut by 45° walls. corridor widths are drawn per side every episode. bodies a
# stage doesn't use are parked AWAY. pads are flat colored squares on the floor (no collision): one per box color.
#
# the course's task state (after robot/humanoid.py's, per robot; zeros for what the stage doesn't use):
#   privileged  per box: position [forward, left, up] from the pelvis, its x axis and velocity in the heading frame,
#               on its pad, named (4 x 11); per pad: center (3), named (1) (4 x 4); each palm face's position (2 x 3);
#               each hand's grip force on boxes / 50 N (2); track: lap progress so far / lap length (1), lateral
#               offset from the centerline (+ = outward) (1), lap direction in the heading frame [forward, left] (2),
#               distance to the island and to the outer walls (2)
#   aux         per box: position (4 x 3); per pad: center (4 x 3); track: lateral offset, lap direction, the two
#               wall distances (5)
import math
import torch
from robot.humanoid import HumanoidCourse, WALK_WORDS, TOKENS, DT, SUBSTEPS, HANDS
from robot.g1 import g1, PREFIX, STAND
from physics.math3d import quat_rotate, quat_mul, quat_from_rotvec, const

NAME = 'warehouse'
STAGES = ['walk', 'pick', 'track', 'carry', 'deliver', 'sort']
# walk commands (the trainer's command curriculum raises the forward top speed)
COMMANDED = True
# episode length per stage in control steps (50 Hz)
STAGE_STEPS = {'walk': 1000, 'pick': 750, 'track': 1500, 'carry': 2000, 'deliver': 1500, 'sort': 4500}
# falls (robot/humanoid.py fallen()): {stage: (lowest pelvis height m, largest pelvis tilt rad)}: upright for walking,
# down into the floor-pick squat for the box stages
FALL_LIMITS = {'walk': (0.35, 1.0), 'pick': (0.2, 1.4), 'track': (0.35, 1.0), 'carry': (0.35, 1.0), 'deliver': (0.2, 1.4),
               'sort': (0.2, 1.4)}
# pass rate each stage's test needs (the share of its tries that pass)
PASS = {'walk': 0.9, 'pick': 0.8, 'track': 0.9, 'carry': 0.8, 'deliver': 0.8, 'sort': 0.8}
# boxes: {color name: color}; size [x (depth), y (height), z (width)] (m), mass (kg)
BOXES = {'red': '#d1495b', 'green': '#2a9d8f', 'blue': '#3a6ea5', 'yellow': '#e9c46a'}
BOX_SIZE = [0.25, 0.25, 0.3]
BOX_MASS = 1.5
# pads: one per box color, square, PAD_SIZE wide and PAD_THICK high
PAD_SIZE = 0.8
PAD_THICK = 0.005
# track (see the header): island half extents [x, z] (m), corridor width range per side, wall height / thickness,
# outer wall lengths (long ones along x, short ones along z: longer than the widest track, they overlap at the corners),
# corner cut (m along each wall from the corner)
ISLAND = (3.0, 1.0)
CORRIDOR = (1.4, 2.2)
WALL_HEIGHT = 0.5
WALL_THICK = 0.1
WALL_LONG = 2 * (ISLAND[0] + CORRIDOR[1]) + 1.0
WALL_SHORT = 2 * (ISLAND[1] + CORRIDOR[1]) + 1.0
CORNER_CUT = 0.8
# track start: on the centerline at a random point, up to START_LATERAL off it, facing along it either way
START_LATERAL = 0.2
# a wall touch: contact normal force on any robot part above this (N)
TOUCH_FORCE = 1.0
TRACK_SPEED = 1.0
CARRY_SPEED = 0.7
# box stage layout (robot at the origin): boxes and pads inside [-AREA, AREA]^2, boxes at least BOX_GAP apart and from
# the robot, pads PAD_GAP apart, a box's center at least OFF_PAD from every pad center (it starts off the pads); pick /
# deliver: the named box START_DIST ahead of the robot, within START_YAW of straight ahead
AREA = 3.5
BOX_GAP = 0.8
PAD_GAP = 1.2
OFF_PAD = 0.9
START_DIST = (0.8, 2.5)
START_YAW = 0.5
# carry start: standing, arms in the hold pose (palms flat on the box's side faces, found by a search over the arm
# joints through the model's kinematics, checked collision-free), the box between them at HOLD_BOX (robot at the
# origin facing +x)
HOLD_ARM = {'shoulder_pitch': -0.7665, 'shoulder_roll': 0.2253, 'shoulder_yaw': 0.2033, 'elbow': 1.0241, 'wrist_roll': -0.0815,
            'wrist_pitch': -0.3185, 'wrist_yaw': -0.1615}
# right arm = left arm mirrored: these joints turn the other way
MIRRORED = {'shoulder_roll', 'shoulder_yaw', 'wrist_roll', 'wrist_yaw'}
HOLD = {**STAND, **{f'left_{j}_joint': v for j, v in HOLD_ARM.items()},
        **{f'right_{j}_joint': -v if j in MIRRORED else v for j, v in HOLD_ARM.items()}}
HOLD_BOX = [0.436, 0.88, 0.0]
# palm face: the hand box's palm side (robot/g1.py): half its thickness from its center, along -y (left) / +y (right)
PALM_HALF = 0.0333
# grasp: each palm's target is the center of the box face on its side, GRASP_RAISE above the box's center
GRASP_RAISE = 0.05
# a hand grips a box with more than this force (N); full grip shaping at GRIP_FORCE
GRIP_MIN = 1.0
GRIP_FORCE = 10.0
# pick: bottom LIFT_HEIGHT up for HOLD_STEPS. carrying (deliver, sort): lifted CARRY_LIFT counts fully. deliver: let go
# (no hand touching it) for RELEASE_STEPS. a box rests on a pad: center inside it by PAD_IN, bottom within REST of
# the floor. carry: the box is lost when its bottom is below LOST_HEIGHT
LIFT_HEIGHT = 0.3
HOLD_STEPS = 100
CARRY_LIFT = 0.1
RELEASE_STEPS = 25
PAD_IN = 0.1
REST = 0.02
LOST_HEIGHT = 0.05
SAME_PAD = 0.5
DELIVER_HELD = 0.5
# out of the way: bodies a stage doesn't use sit this far off (m)
AWAY = 50.0
# locomotion reward weights per stage {term: (walk, pick, track, carry, deliver, sort)} (robot/humanoid.py TERMS):
# walk is the MuJoCo Playground G1 joystick recipe (smoothness is a training loss on the policy, ml/train_humanoid.py). the box stages let the trunk bend and the arms work (no upper pose, a looser
# orientation)
WEIGHTS = {'track_lin_vel': (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'track_ang_vel': (0.75, 0.0, 0.0, 0.0, 0.0, 0.0),
           'ang_vel_xy': (-0.15,) * 6, 'orientation': (-2.0, -0.5, -2.0, -2.0, -0.5, -0.5), 'dof_pos_limits': (-1.0,) * 6,
           'hip_pos': (-1.0,) * 6, 'feet_slip': (-0.25,) * 6, 'feet_phase': (1.0, 0.5, 1.0, 1.0, 0.5, 0.5),
           'feet_air_time': (2.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'stand_still': (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
           'alive': (0.15, 0.5, 0.5, 0.5, 0.5, 0.5), 'upper_pose': (-0.2, 0.0, -0.2, 0.0, 0.0, 0.0)}
# task reward (not clipped). track / carry: PROGRESS_W per meter of lap progress (carry: while holding the box),
# WALL_PENALTY per step touching a wall. the box stages: potential shaping (the change
# of task_potential(): the pelvis within REACH of the target box, the palms on its faces, gripping it, lifting it,
# and in deliver / sort carrying it to its pad, setting it down there and letting go), SUCCESS_REWARD for a pass
# (sort: per box on its pad), LOST_PENALTY for losing the box in carry. FALL_PENALTY for falling in every stage but walk
PROGRESS_W = 3.0
WALL_PENALTY = 0.5
APPROACH_W = 1.0
REACH = 0.5
HAND_W = 4.0
GRIP_W = 2.0
LIFT_W = 4.0
PAD_W = 1.0
PLACE_W = 2.0
SUCCESS_REWARD = 10.0
LOST_PENALTY = 10.0
FALL_PENALTY = 10.0
VOCAB = WALK_WORDS + [NAME, 'pick', 'up', 'the', 'box', 'run', 'a', 'lap', 'with', 'put', 'on', 'pad', 'sort', 'boxes',
                      'by', 'color'] + list(BOXES)
NUM_BOXES = len(BOXES)
NUM_PADS = len(BOXES)
# outer walls (in order): east (+x), west (-x), south (+z), north (-z); corners by (x side, z side) sign
SIDES = [(1, 0), (-1, 0), (0, 1), (0, -1)]
CORNERS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
# the centerline loop's edge directions (x, z), in loop order from its (x0, z0) corner
EDGE_DIRS = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]


def scene():
    # scene JSON: a ground plane, the G1 standing at the origin, the track, boxes and pads parked away (training
    # envs lay them out every episode)
    identity = [1.0, 0.0, 0.0, 0.0]
    still = {'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'density': 1000.0, 'static': True,
             'collide': True, 'group': 0, 'friction': 1.0, 'restitution': 0.0}
    box_shape = lambda size: [{'type': 'box', 'size': size, 'position': [0.0, 0.0, 0.0], 'rotation': identity}]
    ground = {'id': 'ground', 'shapes': [{'type': 'plane', 'position': [0.0, 0.0, 0.0], 'rotation': identity}],
              'position': [0.0, 0.0, 0.0], 'rotation': identity, 'color': '#9a9a9a', **still}
    bodies = [ground]
    bodies.append({'id': 'island', 'shapes': box_shape([2 * ISLAND[0], WALL_HEIGHT, 2 * ISLAND[1]]),
                   'position': [AWAY, WALL_HEIGHT / 2, AWAY], 'rotation': identity, 'color': '#457b9d', **still})
    for i, (sx, sz) in enumerate(SIDES):
        size = [WALL_LONG, WALL_HEIGHT, WALL_THICK] if sz else [WALL_THICK, WALL_HEIGHT, WALL_SHORT]
        bodies.append({'id': f'wall/{i}', 'shapes': box_shape(size), 'position': [AWAY, WALL_HEIGHT / 2, AWAY],
                       'rotation': identity, 'color': '#e9ecef', **still})
    # a corner wall runs from CORNER_CUT along one wall to CORNER_CUT along the other: its long axis turned 45° about
    # +y (+x turned by yaw points along (cos yaw, -sin yaw) in (x, z)), the sign by corner
    for i, (sx, sz) in enumerate(CORNERS):
        yaw = math.pi / 4 * sx * sz
        bodies.append({'id': f'corner/{i}', 'shapes': box_shape([CORNER_CUT * math.sqrt(2) + WALL_THICK, WALL_HEIGHT, WALL_THICK]),
                       'position': [AWAY, WALL_HEIGHT / 2, AWAY], 'rotation': [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0],
                       'color': '#e9ecef', **still})
    for k, (name, color) in enumerate(BOXES.items()):
        bodies.append({'id': f'pad/{name}', 'shapes': box_shape([PAD_SIZE, PAD_THICK, PAD_SIZE]),
                       'position': [AWAY + 2 * k, PAD_THICK / 2, -AWAY], 'rotation': identity, 'color': color,
                       **{**still, 'collide': False}})
        bodies.append({'id': f'box/{name}', 'shapes': box_shape(BOX_SIZE),
                       'position': [AWAY + 2 * k, BOX_SIZE[1] / 2, AWAY], 'rotation': identity,
                       'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0],
                       'density': BOX_MASS / (BOX_SIZE[0] * BOX_SIZE[1] * BOX_SIZE[2]), 'static': False, 'collide': True,
                       'group': 0, 'friction': 1.0, 'restitution': 0.0, 'color': color})
    bot = g1()
    return {'name': NAME, 'settings': {'dt': DT, 'substeps': SUBSTEPS, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0, 'max_contacts': 128},
            'bodies': bodies + bot['bodies'], 'joints': bot['joints'], 'cameras': bot['cameras']}


ACTUATORS = [a.removeprefix(PREFIX) for a in g1()['actuators']]
ACTION_DIM = len(ACTUATORS)
# gyro 3 + gravity 3 + joint angles + joint speeds + last action + gait clock 2
PROPRIO_DIM = 6 + 3 * ACTION_DIM + 2
# robot/humanoid.py's privileged state (noise-free proprio, stage one-hot, walk command 3, base velocity 3, height 1,
# feet 6), then the task's (see the header)
PRIV_DIM = PROPRIO_DIM + len(STAGES) + 13 + NUM_BOXES * 11 + NUM_PADS * 4 + 2 * 3 + 2 + 6
# base velocity 3 + height 1 + foot contacts 2 + foot heights 2, then the task's
AUX_DIM = 8 + NUM_BOXES * 3 + NUM_PADS * 3 + 5


class WarehouseCourse(HumanoidCourse):
    NAME, STAGES, STAGE_STEPS, FALL_LIMITS, WEIGHTS, VOCAB = NAME, STAGES, STAGE_STEPS, FALL_LIMITS, WEIGHTS, VOCAB

    # stage_probs: (S,) probability of each stage at reset, in STAGES order. vision: render the head camera
    def __init__(self, num_envs, device, stage_probs, vision, seed=0):
        super().__init__(scene(), num_envs, device, [PREFIX], ACTUATORS, stage_probs, vision, seed)
        w = self.world
        # (K,) boxes, (K,) pads (by color), (9,) track walls (island, outer walls, corners)
        self.boxes = torch.tensor([w.body_index[f'box/{n}'] for n in BOXES], device=device)
        self.pads = torch.tensor([w.body_index[f'pad/{n}'] for n in BOXES], device=device)
        self.island = w.body_index['island']
        self.outer = torch.tensor([w.body_index[f'wall/{i}'] for i in range(len(SIDES))], device=device)
        self.corners = torch.tensor([w.body_index[f'corner/{i}'] for i in range(len(CORNERS))], device=device)
        self.walls = torch.cat([torch.tensor([self.island], device=device), self.outer, self.corners])
        # (R, 2) each hand's box geom (its only collision shape), (2,) which way its palm side faces in its frame (y)
        self.palm_geom = torch.stack([(w.geom_body == h).nonzero()[:, 0] for h in self.hands.view(-1)]).view(self.R, 2)
        self.palm_side = const([-1.0, 1.0], device)
        # carry start: (P, 3), (P, 4) the robot's parts in the hold pose (built like the scene's robot)
        held = {b['id']: b for b in g1(pose=HOLD)['bodies']}
        self.hold_pos = torch.tensor([held[w.body_ids[i]]['position'] for i in self.parts[0].tolist()], device=device)
        self.hold_quat = torch.tensor([held[w.body_ids[i]]['rotation'] for i in self.parts[0].tolist()], device=device)
        # per env / robot: (E, R) named box, (E, R) named pad, (E, 4) track centerline rectangle [x0, x1, z0, z1]
        # (around the arena center (0, 0)), (E, R) lap direction (+1: increasing centerline position), (E, R) last
        # centerline position (m), (E, R) lap progress so far (m), (E, R) touched a wall this episode, (E, R) steps in
        # a row the pick has been held up / the delivered box let go, (E,) boxes on their pads (sort score), (E, R) task
        # potential, (E,) carry: the box is still held (not lost)
        self.target = torch.zeros(self.E, self.R, dtype=torch.long, device=device)
        self.target_pad = torch.zeros(self.E, self.R, dtype=torch.long, device=device)
        self.centerline = torch.zeros(self.E, 4, device=device)
        self.lap_dir = torch.ones(self.E, self.R, device=device)
        self.lap_pos = torch.zeros(self.E, self.R, device=device)
        self.progress = torch.zeros(self.E, self.R, device=device)
        self.touched = torch.zeros(self.E, self.R, dtype=torch.bool, device=device)
        self.streak = torch.zeros(self.E, self.R, dtype=torch.long, device=device)
        self.score = torch.zeros(self.E, dtype=torch.long, device=device)
        self.potential = torch.zeros(self.E, self.R, device=device)
        self.kept = torch.ones(self.E, dtype=torch.bool, device=device)
        self.reset(torch.ones(self.E, dtype=torch.bool, device=device))
        self.render()

    # ---- layout ----

    def scatter(self, n, count, taken, gaps, rand):
        # (n, count, 2) points in [-AREA, AREA]^2, each at least gaps[i] (for taken's i-th point) from the points in
        # taken (n, T, 2) and gaps[-1] from each other (rejection sampling, one point at a time)
        points = torch.zeros(n, 0, 2, device=self.device)
        gap = torch.tensor(gaps[:-1], device=self.device)
        for _ in range(count):
            new = torch.zeros(n, 2, device=self.device)
            todo = torch.ones(n, dtype=torch.bool, device=self.device)
            while todo.any():
                cand = (rand(n, 2) * 2 - 1) * AREA
                ok = (torch.linalg.norm(taken - cand[:, None], dim=-1) > gap).all(-1) & (torch.linalg.norm(points - cand[:, None], dim=-1) > gaps[-1]).all(-1)
                new = torch.where((todo & ok)[:, None], cand, new)
                todo = todo & ~ok
            points = torch.cat([points, new[:, None]], 1)
        return points

    def reset_task(self, mask, envs):
        # lay out the track, boxes, pads and the robot for the new episodes in envs (n,) (mask (E,))
        w = self.world
        n = len(envs)
        rand = lambda *shape: torch.rand(*shape, generator=self.gen, device=self.device)
        stage = self.stage[envs]
        is_ = lambda name: stage == STAGES.index(name)
        track = is_('track') | is_('carry')
        # track: corridor widths per side (east, west, south, north), walls around them; parked away otherwise
        width = CORRIDOR[0] + (CORRIDOR[1] - CORRIDOR[0]) * rand(n, 4)
        # (n, 3) offset of the track's bodies: none, or AWAY along x and z
        away = torch.where(track, 0.0, AWAY)[:, None] * const([1.0, 0.0, 1.0], self.device)
        # (n, 4) inner edge of each outer wall: x of east / west, z of south / north (signed)
        edge = torch.stack([ISLAND[0] + width[:, 0], -ISLAND[0] - width[:, 1], ISLAND[1] + width[:, 2], -ISLAND[1] - width[:, 3]], -1)
        height = torch.full((n,), WALL_HEIGHT / 2, device=self.device)
        zero = torch.zeros(n, device=self.device)
        w.pos[envs, self.island] = torch.stack([zero, height, zero], -1) + away
        for i, (sx, sz) in enumerate(SIDES):
            off = edge[:, i] + (sx + sz) * WALL_THICK / 2
            pos = torch.stack([off, height, zero], -1) if sx else torch.stack([zero, height, off], -1)
            w.pos[envs, self.outer[i]] = pos + away
        # a corner wall's center: CORNER_CUT / 2 in from the corner along both walls, then out by half its thickness
        out = WALL_THICK / 2 / math.sqrt(2)
        for i, (sx, sz) in enumerate(CORNERS):
            cx = edge[:, 0] if sx > 0 else edge[:, 1]
            cz = edge[:, 2] if sz > 0 else edge[:, 3]
            w.pos[envs, self.corners[i]] = torch.stack([cx - sx * (CORNER_CUT / 2 - out), height, cz - sz * (CORNER_CUT / 2 - out)], -1) + away
        # centerline: midway between the island and each outer wall
        self.centerline[envs] = torch.stack([-ISLAND[0] - width[:, 1] / 2, ISLAND[0] + width[:, 0] / 2,
                                             -ISLAND[1] - width[:, 3] / 2, ISLAND[1] + width[:, 2] / 2], -1)
        self.target[envs] = torch.randint(0, NUM_BOXES, (n, self.R), generator=self.gen, device=self.device)
        other = torch.randint(0, NUM_PADS, (n, self.R), generator=self.gen, device=self.device)
        self.target_pad[envs] = torch.where(rand(n, self.R) < SAME_PAD, self.target[envs], other)
        target = self.target[envs, 0]
        # robot: on the track's centerline facing along it (track, carry), else at the origin facing anywhere
        rect = self.centerline[envs]
        point, tangent = self.centerline_point(rect, rand(n) * self.perimeter(rect))
        lateral = (rand(n) * 2 - 1) * START_LATERAL
        direction = torch.where(rand(n) < 0.5, 1.0, -1.0)
        # lateral: along the tangent turned 90° (in (x, z): (tx, tz) -> (-tz, tx))
        start = point + lateral[:, None] * torch.stack([-tangent[:, 1], tangent[:, 0]], -1)
        heading = tangent * direction[:, None]
        xz = torch.where(track[:, None], start, torch.zeros(n, 2, device=self.device))
        yaw = torch.where(track, torch.atan2(-heading[:, 1], heading[:, 0]), (rand(n) * 2 - 1) * math.pi)
        # carry, and DELIVER_HELD of deliver, start in the hold pose with the named box in the hands
        holding = is_('carry') | (is_('deliver') & (rand(n) < DELIVER_HELD))
        pose = (torch.where(holding[:, None, None], self.hold_pos, self.world.init_pos[self.parts[0]]),
                torch.where(holding[:, None, None], self.hold_quat, self.world.init_quat[self.parts[0]]))
        self.place_robot(envs, 0, xz, yaw, pose)
        self.lap_dir[envs] = direction[:, None]
        self.lap_pos[envs] = self.track_coords(envs)[0]
        self.progress[envs] = 0.0
        self.touched[envs] = False
        # boxes and pads (box stages): pads anywhere; the named box START_DIST ahead of the robot (pick, deliver), the
        # rest anywhere clear of the robot, the pads and each other. +x turned by yaw points along (cos yaw, -sin yaw)
        boxes_on = is_('pick') | is_('deliver') | is_('sort')
        pads_on = is_('deliver') | is_('sort')
        robot = torch.zeros(n, 1, 2, device=self.device)
        ahead = yaw + (rand(n) * 2 - 1) * START_YAW
        dist = START_DIST[0] + (START_DIST[1] - START_DIST[0]) * rand(n)
        # (n, 2) the named box's spot in pick / deliver (sort scatters every box, a held start holds it; pads and boxes
        # still keep clear of this spot, which changes nothing)
        first = dist[:, None] * torch.stack([torch.cos(ahead), -torch.sin(ahead)], -1)
        named_first = is_('pick') | is_('deliver')
        pads = self.scatter(n, NUM_PADS, torch.stack([robot[:, 0], first], 1), [0.0, OFF_PAD, PAD_GAP], rand)
        rest = self.scatter(n, NUM_BOXES, torch.cat([robot, pads, first[:, None]], 1), [BOX_GAP] + [OFF_PAD] * NUM_PADS + [BOX_GAP, BOX_GAP], rand)
        k = torch.arange(NUM_BOXES, device=self.device)
        spots = torch.where((named_first[:, None] & (k == target[:, None]))[..., None], first[:, None], rest)
        level = lambda count, height: torch.full((n, count, 1), height, device=self.device)
        box_pos = torch.cat([spots[..., :1], level(NUM_BOXES, BOX_SIZE[1] / 2), spots[..., 1:]], -1)
        pad_pos = torch.cat([pads[..., :1], level(NUM_PADS, PAD_THICK / 2), pads[..., 1:]], -1)
        parked = lambda count, z: torch.stack([AWAY + 2 * k[:count].float(), torch.full((count,), 0.0, device=self.device), torch.full((count,), z, device=self.device)], -1)
        box_pos = torch.where(boxes_on[:, None, None], box_pos, parked(NUM_BOXES, AWAY) + const([0.0, BOX_SIZE[1] / 2, 0.0], self.device))
        pad_pos = torch.where(pads_on[:, None, None], pad_pos, parked(NUM_PADS, -AWAY) + const([0.0, PAD_THICK / 2, 0.0], self.device))
        box_yaw = rand(n, NUM_BOXES) * 2 * math.pi
        box_quat = quat_from_rotvec(torch.stack([torch.zeros_like(box_yaw), box_yaw, torch.zeros_like(box_yaw)], -1))
        # carry: the named box between the palms (HOLD_BOX turned with the robot), square to it
        q_yaw = quat_from_rotvec(torch.stack([zero, yaw, zero], -1))
        hold = torch.stack([xz[:, 0], zero, xz[:, 1]], -1) + quat_rotate(q_yaw, const(HOLD_BOX, self.device).expand(n, 3))
        held = holding[:, None] & (k == target[:, None])
        box_pos = torch.where(held[..., None], hold[:, None], box_pos)
        box_quat = torch.where(held[..., None], q_yaw[:, None], box_quat)
        w.pos[envs[:, None], self.boxes] = box_pos
        w.quat[envs[:, None], self.boxes] = box_quat
        w.velocity[envs[:, None], self.boxes] = 0.0
        w.pos[envs[:, None], self.pads] = pad_pos
        w.inv_inertia_world = w.compute_inv_inertia_world()
        self.streak[envs] = 0
        self.score[envs] = self.on_pad()[envs].sum(-1)
        self.kept[envs] = True
        self.potential = torch.where(mask[:, None], self.task_potential(), self.potential)

    def set_task(self, stage, box=0, pad=0):
        # switch every env's running episode to another stage's task in place (no new layout: the scene stays as it
        # is, ml/demo_humanoid.py): stage name, the named box and pad (color indices)
        self.stage[:] = STAGES.index(stage)
        self.target[:] = box
        self.target_pad[:] = pad
        self.steps[:] = 0
        self.streak[:] = 0
        self.progress[:] = 0.0
        self.touched[:] = False
        self.kept[:] = True
        self.score[:] = self.on_pad().sum(-1)
        self.lap_pos = self.track_coords()[0]
        self.potential = self.task_potential()

    # ---- the track ----

    def perimeter(self, rect):
        # (n, 4) centerline rectangles [x0, x1, z0, z1] -> (n,) their perimeter (m)
        return 2 * (rect[:, 1] - rect[:, 0] + rect[:, 3] - rect[:, 2])

    def centerline_point(self, rect, s):
        # (n, 4) rectangles, (n,) position along their loop (m, from (x0, z0) along +x, then +z, then -x, then -z) ->
        # (n, 2) the point (x, z), (n, 2) the loop's unit direction there (increasing s)
        lx, lz = rect[:, 1] - rect[:, 0], rect[:, 3] - rect[:, 2]
        # (n, 4) loop position where each edge starts, and its start corner (n, 4, 2)
        starts = torch.stack([torch.zeros_like(lx), lx, lx + lz, 2 * lx + lz], -1)
        corner = torch.stack([torch.stack([rect[:, 0], rect[:, 2]], -1), torch.stack([rect[:, 1], rect[:, 2]], -1),
                              torch.stack([rect[:, 1], rect[:, 3]], -1), torch.stack([rect[:, 0], rect[:, 3]], -1)], 1)
        edge = ((s[:, None] >= starts).sum(-1) - 1).clamp(0, 3)
        t = s - starts.gather(1, edge[:, None])[:, 0]
        tangent = const(EDGE_DIRS, self.device)[edge]
        return corner[torch.arange(len(s), device=self.device), edge] + t[:, None] * tangent, tangent

    def track_coords(self, envs=None):
        # robots' pelvis on the track of envs (n,) (all envs if None): (n, R) position along the centerline loop (m, at
        # its nearest point), (n, R) lateral offset (+ = outward of the centerline), (n, R, 2) the lap direction (x, z)
        # at the nearest point
        envs = torch.arange(self.E, device=self.device) if envs is None else envs
        x0, x1, z0, z1 = self.centerline[envs][:, None].unbind(-1)
        p = self.world.pos[envs][:, self.pelvis][..., [0, 2]]
        x, z = torch.maximum(torch.minimum(p[..., 0], x1), x0), torch.maximum(torch.minimum(p[..., 1], z1), z0)
        # (n, R, 4) distance to each edge (its nearest point), and that point's position along the loop
        dist = torch.stack([(p[..., 1] - z0).abs() + (p[..., 0] - x).abs(), (p[..., 0] - x1).abs() + (p[..., 1] - z).abs(),
                            (p[..., 1] - z1).abs() + (p[..., 0] - x).abs(), (p[..., 0] - x0).abs() + (p[..., 1] - z).abs()], -1)
        lx, lz = x1 - x0, z1 - z0
        along = torch.stack([x - x0, lx + z - z0, 2 * lx + lz - (x - x0), 2 * (lx + lz) - (z - z0)], -1)
        edge = dist.argmin(-1)
        s = along.gather(-1, edge[..., None])[..., 0] % (2 * (lx + lz))
        outside = (p[..., 0] < x0) | (p[..., 0] > x1) | (p[..., 1] < z0) | (p[..., 1] > z1)
        lateral = dist.amin(-1) * torch.where(outside, 1.0, -1.0)
        return s, lateral, const(EDGE_DIRS, self.device)[edge] * self.lap_dir[envs][..., None]

    # ---- boxes and hands ----

    def box_bottoms(self):
        # (E, K) height of each box's lowest corner
        w = self.world
        corners = const([[sx, sy, sz] for sx in (-1., 1.) for sy in (-1., 1.) for sz in (-1., 1.)], self.device) * const(BOX_SIZE, self.device) / 2
        return (w.pos[:, self.boxes, None, 1] + quat_rotate(w.quat[:, self.boxes, None], corners.expand(self.E, NUM_BOXES, 8, 3))[..., 1]).amin(-1)

    def on_pad(self):
        # (E, K) box resting on the pad of its color: center PAD_IN inside it, bottom within REST of the floor
        w = self.world
        d = (w.pos[:, self.boxes] - w.pos[:, self.pads]).abs()
        inside = (d[..., 0] < PAD_SIZE / 2 - PAD_IN) & (d[..., 2] < PAD_SIZE / 2 - PAD_IN)
        return inside & (self.box_bottoms() < REST)

    def palms(self):
        # (E, R, 2, 3) center of each hand's palm face (left, right)
        w = self.world
        g = self.palm_geom
        body = self.hands
        q = quat_mul(w.quat[:, body], w.geom_local_quat[g].expand(self.E, self.R, 2, 4))
        center = w.pos[:, body] + quat_rotate(w.quat[:, body], w.geom_local_pos[g].expand(self.E, self.R, 2, 3))
        side = (self.palm_side[:, None] * const([0.0, PALM_HALF, 0.0], self.device)).expand(self.E, self.R, 2, 3)
        return center + quat_rotate(q, side)

    def grip(self):
        # (E, R, 2, K) each hand's contact normal force on each box (N)
        return torch.stack([self.contact_force(self.hands, self.boxes[k:k + 1]) for k in range(NUM_BOXES)], -1)

    def target_box(self):
        # (E, R) the box the task steers to: the named one, or in sort the nearest one not on its pad yet
        w = self.world
        dist = torch.linalg.norm((w.pos[:, self.boxes][:, None] - w.pos[:, self.pelvis][:, :, None])[..., [0, 2]], dim=-1)
        dist = torch.where(self.on_pad()[:, None], math.inf, dist)
        return torch.where(self.is_stage('sort')[:, None], dist.argmin(-1), self.target)

    def task_potential(self):
        # (E, R) higher = closer to done (see the reward constants): the pick part (approach, palms on the target
        # box's faces, grip, lift: to LIFT_HEIGHT in pick, CARRY_LIFT when carrying it to a pad), then for deliver /
        # sort its pad (closer, and resting on it: the pick part counts as full there, so setting it down and letting
        # go costs nothing). carry: palms on the box and gripping it. 0 in walk and track
        w = self.world
        e = torch.arange(self.E, device=self.device)[:, None]
        k = self.target_box()
        box = w.pos[:, self.boxes][e, k]
        approach = (torch.linalg.norm((box - w.pos[:, self.pelvis])[..., [0, 2]], dim=-1) - REACH).clamp_min(0)
        # grasp targets: the pair of side faces across the robot (the box axis, x or z, closer to its left), each palm
        # on the face on its side, GRASP_RAISE above the box's center
        _, left = self.heading()
        q = w.quat[:, self.boxes][e, k]
        ax = quat_rotate(q, const([1.0, 0.0, 0.0], self.device).expand(self.E, self.R, 3))
        az = quat_rotate(q, const([0.0, 0.0, 1.0], self.device).expand(self.E, self.R, 3))
        use_x = (ax * left).sum(-1).abs() > (az * left).sum(-1).abs()
        axis = torch.where(use_x[..., None], ax * BOX_SIZE[0] / 2, az * BOX_SIZE[2] / 2)
        axis = axis * torch.sign((axis * left).sum(-1, keepdim=True) + 1e-9)
        raise_ = const([0.0, GRASP_RAISE, 0.0], self.device)
        targets = torch.stack([box + axis + raise_, box - axis + raise_], -2)
        hands = torch.linalg.norm(self.palms() - targets, dim=-1).sum(-1)
        force = self.grip().gather(-1, k[..., None, None].expand(self.E, self.R, 2, 1))[..., 0]
        grip = (force.amin(-1) / GRIP_FORCE).clamp(max=1.0)
        bottom = self.box_bottoms()[e, k]
        lift_to = torch.where(self.is_stage('pick')[:, None], LIFT_HEIGHT, CARRY_LIFT)
        lift = (bottom.clamp_min(0) / lift_to).clamp(max=1.0) * grip
        pick = GRIP_W * grip + LIFT_W * lift - APPROACH_W * approach - HAND_W * hands
        # deliver / sort: the target box's pad (its own color in sort)
        pad = w.pos[:, self.pads][e, torch.where(self.is_stage('sort')[:, None], k, self.target_pad)]
        to_pad = torch.linalg.norm((box - pad)[..., [0, 2]], dim=-1)
        resting = (to_pad < PAD_SIZE / 2 - PAD_IN) & (bottom < REST)
        place = torch.where(resting, GRIP_W + LIFT_W + PLACE_W, pick) - PAD_W * to_pad
        hold = GRIP_W * grip - HAND_W * hands
        out = torch.where((self.is_stage('deliver') | self.is_stage('sort'))[:, None], place, pick)
        out = torch.where(self.is_stage('carry')[:, None], hold, out)
        return torch.where((self.is_stage('walk') | self.is_stage('track'))[:, None], 0.0, out)

    # ---- observations, rewards ----

    def task_tokens(self):
        # (E, R, TOKENS) see the header (padded, then the course's name)
        w = self.word
        fixed = lambda name: torch.full((self.E, self.R), w[name], device=self.device)
        color, pad_color = self.target + w[list(BOXES)[0]], self.target_pad + w[list(BOXES)[0]]
        words = {'pick': [fixed('pick'), fixed('up'), fixed('the'), color, fixed('box')],
                 'track': [fixed('run'), fixed('a'), fixed('lap')],
                 'carry': [fixed('run'), fixed('a'), fixed('lap'), fixed('with'), fixed('the'), color, fixed('box')],
                 'deliver': [fixed('put'), fixed('the'), color, fixed('box'), fixed('on'), fixed('the'), pad_color, fixed('pad')],
                 'sort': [fixed('sort'), fixed('the'), fixed('boxes'), fixed('by'), fixed('color')]}
        out = torch.full((self.E, self.R, TOKENS), w['<pad>'], device=self.device)
        for name, ws in words.items():
            sentence = torch.stack(ws + [fixed('<pad>')] * (TOKENS - 1 - len(ws)) + [fixed(NAME)], -1)
            out = torch.where(self.is_stage(name)[:, None, None], sentence, out)
        return out

    def task_state(self):
        # (E, R, task privileged), (E, R, task aux): see the header
        w = self.world
        fwd, left = self.heading()
        # world vectors (E, R, K, 3) -> [forward, left, up] in each robot's heading frame
        in_heading = lambda v: torch.stack([(v * fwd[:, :, None]).sum(-1), (v * left[:, :, None]).sum(-1), v[..., 1]], -1)
        boxes = self.relative(w.pos[:, self.boxes][:, None].expand(self.E, self.R, NUM_BOXES, 3))
        axis = in_heading(quat_rotate(w.quat[:, self.boxes], const([1.0, 0.0, 0.0], self.device).expand(self.E, NUM_BOXES, 3))[:, None].expand(self.E, self.R, NUM_BOXES, 3))
        vel = in_heading(w.velocity[:, self.boxes, :3][:, None].expand(self.E, self.R, NUM_BOXES, 3))
        on_pad = self.on_pad()[:, None, :, None].expand(self.E, self.R, NUM_BOXES, 1).float()
        named = (self.target[..., None] == torch.arange(NUM_BOXES, device=self.device))[..., None].float()
        pads = self.relative(w.pos[:, self.pads][:, None].expand(self.E, self.R, NUM_PADS, 3))
        pad_named = (self.target_pad[..., None] == torch.arange(NUM_PADS, device=self.device))[..., None].float()
        palms = self.relative(self.palms())
        grip = self.grip().sum(-1) / 50.0
        # (E, 1, 1) which parts of the state the env's stage uses
        boxes_on = (self.is_stage('pick') | self.is_stage('carry') | self.is_stage('deliver') | self.is_stage('sort'))[:, None, None]
        pads_on = (self.is_stage('deliver') | self.is_stage('sort'))[:, None, None]
        track_on = (self.is_stage('track') | self.is_stage('carry'))[:, None, None]
        _, lateral, direction = self.track_coords()
        d = torch.stack([direction[..., 0] * fwd[..., 0] + direction[..., 1] * fwd[..., 2],
                         direction[..., 0] * left[..., 0] + direction[..., 1] * left[..., 2]], -1)
        p = w.pos[:, self.pelvis][..., [0, 2]]
        to_island = torch.linalg.norm((p.abs() - const(ISLAND, self.device)).clamp_min(0), dim=-1)
        # outer wall edges from the centerline rectangle: each side's wall is as far out of the centerline as the island
        # edge is in
        x0, x1, z0, z1 = self.centerline[:, None].unbind(-1)
        to_outer = torch.stack([2 * x1 - ISLAND[0] - p[..., 0], p[..., 0] - (2 * x0 + ISLAND[0]),
                                2 * z1 - ISLAND[1] - p[..., 1], p[..., 1] - (2 * z0 + ISLAND[1])], -1).amin(-1)
        lap = self.perimeter(self.centerline)[:, None]
        track = torch.cat([(self.progress / lap)[..., None], lateral[..., None], d, to_island[..., None], to_outer[..., None]], -1) * track_on
        box_state = torch.cat([boxes, axis, vel, on_pad, named], -1).flatten(-2) * boxes_on
        pad_state = torch.cat([pads, pad_named], -1).flatten(-2) * pads_on
        priv = torch.cat([box_state, pad_state, palms.flatten(-2) * boxes_on, grip * boxes_on, track], -1)
        aux = torch.cat([boxes.flatten(-2) * boxes_on, pads.flatten(-2) * pads_on, track[..., 1:]], -1)
        return priv, aux

    def task_step(self, fell):
        # for the step that just happened (fell (E, R)): (E, R) task reward, (E,) episodes finished, {name: scalar}
        # metrics: tries / passes of the stages' tests (counts this step), track speed, wall touches, task successes
        w = self.world
        e = torch.arange(self.E, device=self.device)
        stage = {name: self.is_stage(name) for name in STAGES}
        # track progress (track, carry)
        s, _, _ = self.track_coords()
        lap = self.perimeter(self.centerline)[:, None]
        # wrapped step along the loop, signed by the lap direction
        step = ((s - self.lap_pos + lap / 2) % lap - lap / 2) * self.lap_dir
        self.lap_pos = s
        touch = (self.contact_force(self.parts[0], self.walls) > TOUCH_FORCE).any(-1)[:, None]
        self.touched |= touch
        target = self.target[:, 0]
        bottom = self.box_bottoms()[e, target]
        force = self.grip()[e, 0, :, target]
        gripped = (force > GRIP_MIN).all(-1)
        # carry: the box is lost on the floor (or dropped below LOST_HEIGHT); progress counts only while it's held
        lost = stage['carry'] & self.kept & (bottom < LOST_HEIGHT)
        self.kept &= ~lost
        moving = stage['track'] | (stage['carry'] & gripped & self.kept)
        self.progress = self.progress + step * moving[:, None]
        done_lap = (self.progress >= lap).all(-1) & (stage['track'] | stage['carry'])
        clean = done_lap & ~self.touched.any(-1) & self.kept
        speed = torch.where(stage['carry'], CARRY_SPEED, TRACK_SPEED)
        fast = self.steps * DT <= lap[:, 0] / speed
        lap_reward = PROGRESS_W * step * moving[:, None] - WALL_PENALTY * touch * (stage['track'] | stage['carry'])[:, None]
        # box stages: shaping, then the stage's success
        pot = self.task_potential()
        shaping = pot - self.potential
        self.streak = torch.where((stage['pick'] & (bottom > LIFT_HEIGHT) & gripped)[:, None], self.streak + 1, 0)
        picked = stage['pick'] & (self.streak[:, 0] >= HOLD_STEPS)
        # deliver: the named box resting on the named pad (any color), let go
        pad = w.pos[e, self.pads[self.target_pad[:, 0]]]
        d = (w.pos[e, self.boxes[target]] - pad).abs()
        on_named = (d[:, 0] < PAD_SIZE / 2 - PAD_IN) & (d[:, 2] < PAD_SIZE / 2 - PAD_IN) & (bottom < REST)
        released = stage['deliver'] & on_named & (force < GRIP_MIN).all(-1)
        self.streak = torch.where(stage['deliver'][:, None], torch.where(released[:, None], self.streak + 1, 0), self.streak)
        delivered = stage['deliver'] & (self.streak[:, 0] >= RELEASE_STEPS)
        score = self.on_pad().sum(-1)
        gained = torch.where(stage['sort'], score - self.score, 0)
        self.score = torch.where(stage['sort'], score, self.score)
        sorted_ = stage['sort'] & (score == NUM_BOXES) & (self.grip().amax((1, 2, 3)) < GRIP_MIN)
        success = picked | delivered | sorted_ | (clean & fast)
        reward = lap_reward + shaping + (SUCCESS_REWARD * (success.float() + gained * stage['sort']) - LOST_PENALTY * lost)[:, None]
        reward = reward - FALL_PENALTY * (fell & ~stage['walk'][:, None])
        # the target may change (sort: another box is now the nearest one off its pad): restart the shaping from it
        self.potential = self.task_potential()
        finished = picked | delivered | sorted_ | done_lap | lost
        # a test episode's end (robot/humanoid.py ends it: fell, finished or out of time) counts a try
        ended = self.test & (fell.any(-1) | finished | (self.steps >= self.ep_steps))
        passed = self.test & success & ~fell.any(-1)
        count = lambda st: st.sum().clamp_min(1)
        metrics = {'track_speed': (step.mean(-1) / DT * stage['track']).sum() / count(stage['track']),
                   'wall_touch': (touch.any(-1) & stage['track']).float().sum() / count(stage['track']),
                   'pick_grip': (gripped & stage['pick']).float().sum() / count(stage['pick']),
                   'pick_lift': (bottom.clamp_min(0) * stage['pick']).sum() / count(stage['pick']),
                   'sort_placed': (score * stage['sort']).float().sum() / count(stage['sort'])}
        for name in STAGES[1:]:
            metrics[f'tried_{name}'] = (ended & stage[name]).float().sum()
            metrics[f'passed_{name}'] = (passed & stage[name]).float().sum()
        return reward, finished, metrics

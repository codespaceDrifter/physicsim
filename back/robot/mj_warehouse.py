# The warehouse course on MuJoCo (MuJoCo Warp: GPU MuJoCo, thousands of worlds batched), for the pure motor baseline:
# a Unitree G1 learns walk -> pick a box up off the floor -> deliver it onto a pad, with no language. its command is
# one task vector (the stage, the walk command, the box's and the pad's state), the way HumanoidBench policies observe
# objects. same pass-gated curriculum, locomotion recipe, pass tests and constants as the physicsim course
# (robot/humanoid.py, robot/warehouse.py); MuJoCo is z-up (x forward, y left, z up).
#   walk     follow a velocity command; pass: the dash (10 m in 12 s, smoothly)
#   pick     the box (on the floor ahead) squeezed between the palms and up; pass: its bottom LIFT_HEIGHT up, gripped by
#            both hands, for HOLD_STEPS in a row
#   deliver  the box onto the pad; pass: resting on it (center PAD_IN inside, bottom within REST of the floor), let go
#            for RELEASE_STEPS
#
# robot: MuJoCo Menagerie's g1_mjx.xml (robot/assets/unitree_g1), with: its collision geoms colliding with the floor
# and the box but not each other (contype 2, conaffinity 1: the file disables all contacts and expects explicit pairs),
# friction on (condim 3), the hands as their rubber-hand meshes' bounds (robot/g1.py GEOM_OVERRIDES), the sole capsules
# dropped (the foot boxes touch the floor), visual meshes dropped (training doesn't render). contact sensors give foot
# contact, trunk-on-floor (falls) and each palm's force on the box. its position actuators (the model's kp / kv) track
# target = rest + ACTION_SCALE * action, rest = robot/g1.py STAND.
#
# observations (observe()), one robot per env (axis R = 1, like the physicsim course):
#   proprio    (E, 1, PROPRIO_DIM)  gyro, gravity, joint angles / speeds, last action (robot/humanoid.py), gait clock
#   command    (E, 1, TASK_DIM)     stage one-hot (3), walk command (3), box: position from the pelvis in the heading
#                                   frame (3), its x and z axes in the heading frame (6), velocity in the heading frame
#                                   (3); pad: position from the pelvis in the heading frame (3). zeros for what the
#                                   stage doesn't use
#   privileged (E, 1, PRIV_DIM)     noise-free proprio, the command, base velocity (3), height (1), feet (contact,
#                                   height, speed: 6), palm faces from the pelvis (6), palm forces on the box / 50 N (2)
#   aux        (E, 1, AUX_DIM)      base velocity, height, foot contacts, foot heights (the state decoder's targets)
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
import mujoco_warp as mjw
import torch
import warp as wp
from robot.g1 import STAND, STAND_HEIGHT, GEOM_OVERRIDES
from robot.humanoid import (DT, ACTION_SCALE, FEET, HANDS, CONTACT_FORCE, PUSH_INTERVAL, PUSH_SPEED, NOISE, GYRO_SCALE,
                            JOINT_VEL_SCALE, COMMAND_RANGES, COMMAND_STEP, COMMAND_STEPS, STAND_PROB, GAIT_FREQ, GAIT_FAST,
                            SWING_HEIGHT, AIR_TIME, STILL, TRACKING_SIGMA, TERMS, SOFT_LIMIT, HIP_JOINTS, UPPER_JOINTS,
                            TEST_PROB, DASH_SPEED, DASH_STEPS, DASH_DIST, DASH_SMOOTH)
from robot.warehouse import (HOLD, MIRRORED, BOX_SIZE, BOX_MASS, PAD_SIZE, START_DIST, START_YAW, PALM_HALF, GRASP_RAISE, GRIP_MIN,
                             GRIP_FORCE, LIFT_HEIGHT, HOLD_STEPS, CARRY_LIFT, RELEASE_STEPS, PAD_IN, REST, APPROACH_W,
                             REACH, HAND_W, GRIP_W, LIFT_W, PAD_W, PLACE_W, SUCCESS_REWARD, FALL_PENALTY)
from physics.math3d import quat_rotate, quat_rotate_inv, quat_from_rotvec, quat_mul, const

MODEL = Path(__file__).parent / 'assets' / 'unitree_g1' / 'g1_mjx.xml'
NAME = 'warehouse'
STAGES = ['walk', 'pick', 'deliver']
# walk commands (the trainer's command curriculum raises the forward top speed)
COMMANDED = True
STAGE_STEPS = {'walk': 1000, 'pick': 750, 'deliver': 1500}
PASS = {'walk': 0.9, 'pick': 0.8, 'deliver': 0.8}
# falls: {stage: (lowest pelvis height m, largest pelvis tilt rad)} (robot/warehouse.py FALL_LIMITS)
FALL_LIMITS = {'walk': (0.35, 1.0), 'pick': (0.2, 1.4), 'deliver': (0.2, 1.4)}
# locomotion reward weights {term: (walk, pick, deliver)} (robot/warehouse.py WEIGHTS; turn tracking doubled: at 0.75
# the noise-free policy held ~+0.35 rad/s of yaw under a straight command, walking circles, and never passed the dash;
# the box stages charge orientation and base height only while stepping (on the way to the target, see rewards()): the
# floor grasp pitches the pelvis ~1 rad, and an orientation penalty there plus stepping in place made standing up worth
# more than holding the crouch; deliver tracks the carry velocity. base_height (unitree_rl_gym's G1: -10 toward its
# standing pelvis height): without it the walk ran hunched and the carry was a duck walk in a deep crouch
# robot/humanoid.py's locomotion terms, and base_height: (pelvis height - STAND_HEIGHT)^2
MJ_TERMS = TERMS + ['base_height']
WEIGHTS = {'base_height': (-10.0, -10.0, -10.0), 'track_lin_vel': (1.0, 0.0, 1.0), 'track_ang_vel': (1.5, 0.0, 0.0), 'ang_vel_xy': (-0.15,) * 3,
           'orientation': (-2.0, -1.0, -1.0), 'dof_pos_limits': (-1.0,) * 3, 'hip_pos': (-1.0,) * 3,
           'feet_slip': (-0.25,) * 3, 'feet_phase': (1.0, 0.5, 0.5), 'feet_air_time': (2.0, 0.0, 0.0),
           'stand_still': (-1.0, 0.0, 0.0), 'alive': (0.15, 0.5, 0.5), 'upper_pose': (-0.2, 0.0, 0.0)}
# physics substeps per control step (the model's timestep is 4 ms, control 50 Hz)
SUBSTEPS = 5
# reverse curriculum (the later part of a task practiced before the earlier one is mastered): START_CROUCH of pick
# episodes start in the floor grasp (crouched, palms on the box's side faces: FLOOR_GRASP, found by a search through
# the model's kinematics, collision-free, feet flat, center of mass over the feet), START_HELD of deliver episodes
# standing with the box in the hands (robot/warehouse.py HOLD). from scratch the policy walked up to the box and
# stood there: a 0.35 m squat with the trunk pitched 60° is too far to find by exploration. from the floor grasp it then
# squeezed the box (45-60 N) but never stood up with it, so START_LIFTED of pick episodes start part way up: a rung of
# the ladder from the floor grasp to the hold (legs, waist and pelvis pitch interpolated at u in LIFT_U, the arms in
# the hold so the palms stay a box width apart; u below LIFT_U's start puts the box into the thighs), the box between
# the palms. from standing starts the policy then gripped the box bent over (pelvis ~0.58 m, not the floor grasp's
# 0.385) and never lifted from there, so START_ARCHIVE of pick episodes reset to states its own pick episodes visited
# with both palms gripping the box still on the floor (an ARCHIVE_SIZE ring of qpos / qvel, ARCHIVE_PER_STEP added
# per step; Go-Explore's restart-from-visited-states): it practices lifting from the grips it actually reaches.
# pass-test episodes always start standing (the whole task)
START_CROUCH = 0.25
START_LIFTED = 0.25
START_ARCHIVE = 0.25
ARCHIVE_SIZE = 8192
ARCHIVE_PER_STEP = 64
# deliver starts: START_PLACED in the floor grasp with the box already on the pad (only letting go left: from held
# starts the policy hovered the box at CARRY_LIFT, where setting it down first loses the lift shaping),
# START_LOWERED on a lift rung with the pad under the box (setting it down: from held starts the policy squatted the
# box to 0.1 m over a pad and stopped), START_HELD standing with it in the hands, the rest standing at the box.
# placed and lowered starts put the pad within PLACED_PAD (m) of the box center
START_PLACED = 0.25
START_LOWERED = 0.25
START_HELD = 0.25
PLACED_PAD = 0.15
# ladder rungs' interpolation fractions (0: floor grasp, 1: hold)
LIFT_U = [0.375 + 0.0625 * i for i in range(11)]
# the floor grasp: pelvis pitch (rad, forward), leg / waist joints (both legs alike), left arm (the right mirrored)
FLOOR_GRASP = {'pelvis_pitch': 1.0419, 'hip_pitch': -2.4973, 'knee': 2.0983, 'ankle_pitch': -0.642, 'waist_pitch': 0.1187,
               'shoulder_pitch': -0.8558, 'shoulder_roll': 0.6076, 'shoulder_yaw': -0.1832, 'elbow': 0.3253,
               'wrist_roll': -0.4807, 'wrist_pitch': -0.42, 'wrist_yaw': 0.2059}
# heading commands (legged_gym): HEADING_PROB of walk commands, and every dash, hold a target heading: each step the
# turn command is HEADING_GAIN x the heading error (rad/s), clipped to the turn range. a turn rate of 0 alone drifts
# (small yaw errors add up: the policy walked circles and missed the dash's 10 m along its start heading)
HEADING_PROB = 0.5
HEADING_GAIN = 0.5
# deliver: the pad PAD_DIST from the box, any direction. held starts (reverse curriculum) take it from 0 up: from the
# held start with the pad 1.5-3 m off the policy squatted with the box and never walked it over or set it down
PAD_DIST = (1.5, 3.0)
# out of the way: the box and pad sit this far off in stages that don't use them (m)
AWAY = 50.0
# MuJoCo Warp buffers per world: contacts, constraint rows
NCONMAX = 64
NJMAX = 512
# constraint solver iterations (Newton) / line search iterations: the model's 5 / 8 are MuJoCo Playground's for
# feet-only contacts; whole-body contacts with the floor and the box need more (at 5 the solver hit its limit
# constantly)
SOLVER_ITERATIONS = 10
LS_ITERATIONS = 20
# robot geoms collide with the floor and the box (type 2, affinity 1), the floor and box with everything (1, 3)
ROBOT_CONTACT = {'contype': '2', 'conaffinity': '1', 'condim': '3', 'friction': '1 0.005 0.0001'}
WORLD_CONTACT = {'contype': '1', 'conaffinity': '3', 'condim': '3', 'friction': '1 0.005 0.0001'}
# the body that carries each hand's box geom, foot bodies, and bodies that may touch the floor without falling
HAND_GEOMS = ['left_hand_collision', 'right_hand_collision']
ACTUATORS = [a for a in ['left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
                         'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee', 'right_ankle_pitch',
                         'right_ankle_roll', 'waist_yaw', 'waist_roll', 'waist_pitch',
                         'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
                         'left_wrist_pitch', 'left_wrist_yaw', 'right_shoulder_pitch', 'right_shoulder_roll',
                         'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw']]
ACTUATORS = [a + '_joint' for a in ACTUATORS]
ACTION_DIM = len(ACTUATORS)
PROPRIO_DIM = 6 + 3 * ACTION_DIM + 2
TASK_DIM = len(STAGES) + 3 + 3 + 6 + 3 + 3
PRIV_DIM = PROPRIO_DIM + TASK_DIM + 3 + 1 + 6 + 6 + 2
AUX_DIM = 8


def mjcf(visual=False):
    # the scene as an MJCF string: the G1 (see the header), a floor, the box (free), the pad (mocap, visual only),
    # contact sensors. visual: keep the G1's meshes and a checkered floor (rendering videos; same qpos layout)
    root = ET.parse(MODEL).getroot()
    asset = root.find('asset')
    if visual:
        root.find('compiler').set('meshdir', str(MODEL.parent / 'assets'))
        ET.SubElement(asset, 'texture', {'name': 'grid', 'type': '2d', 'builtin': 'checker', 'width': '512', 'height': '512',
                                         'rgb1': '0.55 0.55 0.55', 'rgb2': '0.45 0.45 0.45'})
        ET.SubElement(asset, 'material', {'name': 'grid', 'texture': 'grid', 'texrepeat': '1 1', 'texuniform': 'true'})
        ET.SubElement(asset, 'texture', {'type': 'skybox', 'builtin': 'gradient', 'width': '512', 'height': '512',
                                         'rgb1': '0.75 0.82 0.9', 'rgb2': '0.95 0.95 0.95'})
        look = ET.SubElement(root, 'visual')
        ET.SubElement(look, 'headlight', {'ambient': '0.45 0.45 0.45', 'diffuse': '0.6 0.6 0.6', 'specular': '0.1 0.1 0.1'})
        ET.SubElement(root.find('worldbody'), 'light', {'directional': 'true', 'pos': '0 0 5', 'dir': '-0.3 -0.3 -1', 'diffuse': '0.5 0.5 0.5'})
    else:
        # no visual meshes: drop the mesh assets and every visual geom
        for mesh in asset.findall('mesh'):
            asset.remove(mesh)
    for parent in root.iter():
        for g in list(parent.findall('geom')):
            if (g.get('class') == 'visual' and not visual) or g.get('class') == 'foot_capsule':
                parent.remove(g)
    for d in root.iter('default'):
        if d.get('class') == 'collision':
            d.find('geom').attrib.update(ROBOT_CONTACT)
        if d.get('class') == 'visual' and not visual:
            d.find('geom').attrib.pop('type')
    for g in root.iter('geom'):
        if g.get('name') in GEOM_OVERRIDES:
            g.attrib.pop('fromto')
            g.attrib.update(GEOM_OVERRIDES[g.get('name')])
    world = root.find('worldbody')
    world.find('body').set('pos', f'0 0 {STAND_HEIGHT}')
    ET.SubElement(world, 'geom', {'name': 'floor', 'type': 'plane', 'size': '0 0 0.05', **WORLD_CONTACT, **({'material': 'grid'} if visual else {})})
    box = ET.SubElement(world, 'body', {'name': 'box', 'pos': f'{AWAY} 0 {BOX_SIZE[1] / 2}'})
    ET.SubElement(box, 'freejoint', {'name': 'box'})
    # BOX_SIZE is [depth, height, width] (physicsim, y-up): here x depth, y width, z height
    ET.SubElement(box, 'geom', {'name': 'box', 'type': 'box', 'size': f'{BOX_SIZE[0] / 2} {BOX_SIZE[2] / 2} {BOX_SIZE[1] / 2}',
                                'mass': str(BOX_MASS), 'rgba': '0.82 0.29 0.36 1', **WORLD_CONTACT})
    pad = ET.SubElement(world, 'body', {'name': 'pad', 'mocap': 'true', 'pos': f'{-AWAY} 0 0'})
    ET.SubElement(pad, 'geom', {'type': 'box', 'size': f'{PAD_SIZE / 2} {PAD_SIZE / 2} 0.0025', 'contype': '0',
                                'conaffinity': '0', 'rgba': '0.23 0.43 0.65 1'})
    sensor = root.find('sensor')
    # feet on the floor; trunk bodies (everything with a collision geom but the feet and hands) on the floor; palms on
    # the box: contact sensors (data "found force": the number of contacts, then the net force in the world frame)
    contact = lambda name, body: ET.SubElement(sensor, 'contact', {'name': name, 'body1': body, 'data': 'found force', 'reduce': 'netforce', 'num': '1', **({'geom2': 'floor'} if name.startswith(('foot', 'trunk')) else {'body2': 'box'})})
    for f in FEET:
        contact(f'foot/{f}', f)
    bodies = sorted({b.get('name') for b in root.iter('body') if any(g.get('class') in ('collision', 'foot_box') or g.get('name', '').endswith('collision') for g in b.findall('geom'))})
    for b in bodies:
        if b not in FEET + HANDS:
            contact(f'trunk/{b}', b)
    for h in HANDS:
        contact(f'palm/{h}', h)
    return ET.tostring(root, encoding='unicode')


class MjWarehouseCourse:
    NAME, STAGES = NAME, STAGES

    # num_envs worlds on device (cuda), stage_probs (S,) per stage at reset, vision: unused (state-based), seed
    def __init__(self, num_envs, device, stage_probs, vision=False, seed=0):
        assert not vision
        self.E, self.R, self.A = num_envs, 1, ACTION_DIM
        self.device = device
        self.vision = False
        self.gen = torch.Generator(device=device).manual_seed(seed)
        wp.init()
        self.mjm = mujoco.MjModel.from_xml_string(mjcf())
        mjm = self.mjm
        mjm.opt.iterations, mjm.opt.ls_iterations = SOLVER_ITERATIONS, LS_ITERATIONS
        assert abs(mjm.opt.timestep * SUBSTEPS - DT) < 1e-9
        assert [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mjm.nu)] == ACTUATORS
        mjd = mujoco.MjData(mjm)
        # the standing pose: pelvis at STAND_HEIGHT, joints at STAND
        jid = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name, angle in STAND.items():
            mjd.qpos[mjm.jnt_qposadr[jid(name)]] = angle
        mujoco.mj_forward(mjm, mjd)
        # the start poses (qpos facing +x at the origin) and where their box goes (from the pelvis, box yaw 0)
        crouch_q, crouch_box = self.start_pose(mjm, self.floor_grasp(mjd.qpos.copy(), jid), on_floor=True)
        held_q, held_box = self.start_pose(mjm, self.held(mjd.qpos.copy(), jid), on_floor=False)
        # the lift ladder (see START_LIFTED): (K, nq) rung poses, (K, 3) their box centers from the pelvis
        arm = [mjm.jnt_qposadr[j] for j in range(mjm.njnt) if any(s in mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) for s in ('shoulder', 'elbow', 'wrist'))]
        pitch = lambda q: 2 * math.atan2(q[5], q[3])
        hands = {mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in HAND_GEOMS}
        box_body, box_q = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, 'box'), mjm.jnt_qposadr[jid('box')]
        lift_q, lift_box = [], []  # [(nq,)], [(3,)]
        for u in LIFT_U:
            q = crouch_q + (held_q - crouch_q) * u
            q[arm] = held_q[arm]
            p = pitch(crouch_q) + (pitch(held_q) - pitch(crouch_q)) * u
            q[3:7] = [math.cos(p / 2), 0.0, math.sin(p / 2), 0.0]
            q, box = self.start_pose(mjm, q, on_floor=False)
            check = mujoco.MjData(mjm)
            check.qpos[:] = q
            check.qpos[box_q:box_q + 7] = [*(q[:3] + box), 1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(mjm, check)
            # the box touches nothing but the palms
            touching = [c for c in check.contact[:check.ncon] if box_body in (mjm.geom_bodyid[c.geom1], mjm.geom_bodyid[c.geom2])
                        and not {c.geom1, c.geom2} & hands and c.dist < 0]
            assert not touching, f'lift rung u={u}: the box overlaps the robot or floor'
            lift_q.append(q)
            lift_box.append(box)
        self.m = mjw.put_model(mjm)
        # an unconverged step is still a valid (softer) step, so MuJoCo doesn't print it per step either; contact /
        # constraint buffer overflows stay loud
        self.m.opt.warn_overflow &= ~(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        self.d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=NCONMAX, njmax=NJMAX)
        # torch views of the warp state (shared memory: writes go straight into the simulation)
        t = wp.to_torch
        d = self.d
        self.qpos, self.qvel, self.ctrl = t(d.qpos), t(d.qvel), t(d.ctrl)
        self.xpos, self.xquat = t(d.xpos), t(d.xquat)
        self.geom_xpos, self.geom_xmat = t(d.geom_xpos), t(d.geom_xmat)
        self.mocap_pos = t(d.mocap_pos)
        self.sensordata = t(d.sensordata)
        # (nq,) / (nv,) the standing state every world starts from
        self.init_qpos = torch.tensor(mjd.qpos, dtype=torch.float32, device=device)
        self.crouch_qpos, self.crouch_box = torch.tensor(crouch_q, dtype=torch.float32, device=device), torch.tensor(crouch_box, dtype=torch.float32, device=device)
        self.held_qpos, self.held_box = torch.tensor(held_q, dtype=torch.float32, device=device), torch.tensor(held_box, dtype=torch.float32, device=device)
        self.lift_qpos, self.lift_box = torch.tensor(np.array(lift_q), dtype=torch.float32, device=device), torch.tensor(np.array(lift_box), dtype=torch.float32, device=device)
        body = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        self.pelvis, self.box = body('pelvis'), body('box')
        self.feet = torch.tensor([body(f) for f in FEET], device=device)
        self.hand_geoms = torch.tensor([mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in HAND_GEOMS], device=device)
        # free joints: the robot's (qpos 0:7, qvel 0:6) and the box's
        self.box_q, self.box_v = mjm.jnt_qposadr[jid('box')], mjm.jnt_dofadr[jid('box')]
        assert mjm.jnt_qposadr[jid('floating_base_joint')] == 0
        # (A,) qpos / qvel index of each actuated joint, rest angles, soft limits around the rest pose
        self.act_q = torch.tensor([mjm.jnt_qposadr[jid(a)] for a in ACTUATORS], device=device)
        self.act_v = torch.tensor([mjm.jnt_dofadr[jid(a)] for a in ACTUATORS], device=device)
        self.rest = self.init_qpos[self.act_q]
        lo = torch.tensor([mjm.jnt_range[jid(a)][0] for a in ACTUATORS], device=device)
        hi = torch.tensor([mjm.jnt_range[jid(a)][1] for a in ACTUATORS], device=device)
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_LIMIT
        self.limit_lo, self.limit_hi = mid - half - self.rest, mid + half - self.rest
        self.hip = torch.tensor([ACTUATORS.index(n) for n in HIP_JOINTS], device=device)
        self.upper = torch.tensor([ACTUATORS.index(n) for n in UPPER_JOINTS], device=device)
        # sensor addresses: gyro, and the contact sensors' [found, fx, fy, fz]
        sens = lambda name: int(mjm.sensor_adr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, name)])
        self.gyro_adr = sens('gyro_pelvis')
        self.foot_adr = torch.tensor([sens(f'foot/{f}') for f in FEET], device=device)
        self.trunk_adr = torch.tensor([mjm.sensor_adr[i] for i in range(mjm.nsensor)
                                       if mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SENSOR, i).startswith('trunk/')], device=device)
        self.palm_adr = torch.tensor([sens(f'palm/{h}') for h in HANDS], device=device)
        self.foot_rest = self.xpos[0, self.feet, 2].clone()
        S = len(STAGES)
        self.stage_probs = torch.tensor(stage_probs, dtype=torch.float32, device=device)
        self.weights = torch.tensor([[WEIGHTS[k][s] for k in MJ_TERMS] for s in range(S)], device=device)
        self.stage_steps = torch.tensor([STAGE_STEPS[s] for s in STAGES], device=device)
        self.fall_height = torch.tensor([FALL_LIMITS[s][0] for s in STAGES], device=device)
        self.fall_tilt = torch.tensor([FALL_LIMITS[s][1] for s in STAGES], device=device)
        self.cmd_lo = torch.tensor([round(lo / COMMAND_STEP) for lo, hi in COMMAND_RANGES], device=device)
        self.cmd_hi = torch.tensor([round(hi / COMMAND_STEP) for lo, hi in COMMAND_RANGES], device=device)
        E = num_envs
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)
        # per env (robot axis R = 1 where the trainer expects it): stage, steps, episode length, walk command, the last
        # three actions, gait phase, push timer, feet air time / last contact / last position, test / dash flags, dash
        # start / heading / jerk sum, the pick / release streak, task potential
        self.stage, self.steps, self.ep_steps = zeros(E, dtype=torch.long), zeros(E, dtype=torch.long), zeros(E, dtype=torch.long)
        self.command = zeros(E, 1, 3)
        self.last_action, self.prev_action, self.prev2_action = zeros(E, 1, self.A), zeros(E, 1, self.A), zeros(E, 1, self.A)
        self.phase = zeros(E, 1)
        self.push_timer = torch.randint(PUSH_INTERVAL // 2, PUSH_INTERVAL * 3 // 2, (E,), generator=self.gen, device=device)
        self.pushes = True
        self.auto_reset = True
        # fraction of episodes that are pass tests (a recording sets 1: every episode the whole task, mean actions)
        self.test_prob = TEST_PROB
        self.air_time, self.last_contact = zeros(E, 1, 2), zeros(E, 1, 2, dtype=torch.bool)
        self.last_feet = zeros(E, 2, 3)
        self.test, self.dash = zeros(E, dtype=torch.bool), zeros(E, dtype=torch.bool)
        self.dash_start, self.dash_heading, self.dash_jerk = zeros(E, 2), zeros(E, 2), zeros(E)
        self.streak = zeros(E, dtype=torch.long)
        # heading commands: (E,) on, (E,) target yaw (rad)
        self.heading_mode, self.target_yaw = zeros(E, dtype=torch.bool), zeros(E)
        self.potential = zeros(E)
        # the visited-state archive (see START_ARCHIVE): (ARCHIVE_SIZE, nq) / (ARCHIVE_SIZE, nv), states stored, next slot
        self.archive_q, self.archive_v = zeros(ARCHIVE_SIZE, self.qpos.shape[1]), zeros(ARCHIVE_SIZE, self.qvel.shape[1])
        self.archive_n, self.archive_i = 0, 0
        self.graph = None
        self.reset(torch.ones(E, dtype=torch.bool, device=device))

    def floor_grasp(self, q, jid):
        # (nq,) qpos q set to the floor grasp (FLOOR_GRASP), pelvis pitched forward about its +y
        g = FLOOR_GRASP
        for side in ('left', 'right'):
            q[self.mjm.jnt_qposadr[jid(f'{side}_hip_pitch_joint')]] = g['hip_pitch']
            q[self.mjm.jnt_qposadr[jid(f'{side}_knee_joint')]] = g['knee']
            q[self.mjm.jnt_qposadr[jid(f'{side}_ankle_pitch_joint')]] = g['ankle_pitch']
            for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'):
                q[self.mjm.jnt_qposadr[jid(f'{side}_{j}_joint')]] = -g[j] if side == 'right' and j in MIRRORED else g[j]
        q[self.mjm.jnt_qposadr[jid('waist_pitch_joint')]] = g['waist_pitch']
        q[3:7] = [math.cos(g['pelvis_pitch'] / 2), 0.0, math.sin(g['pelvis_pitch'] / 2), 0.0]
        return q

    def held(self, q, jid):
        # (nq,) qpos q with the arms in the hold pose (robot/warehouse.py HOLD), standing
        for name, angle in HOLD.items():
            q[self.mjm.jnt_qposadr[jid(name)]] = angle
        return q

    def start_pose(self, mjm, q, on_floor):
        # a start pose qpos (nq,): set down so the lowest foot corner touches the floor -> (nq,) that qpos, (3,) the box
        # center (from the pelvis): between the palm faces, at their height or (on_floor) resting on the floor
        d = mujoco.MjData(mjm)
        d.qpos[:] = q
        mujoco.mj_forward(mjm, d)
        feet = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f'{s}_foot_box_collision') for s in ('left', 'right')]
        corners = [d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (mjm.geom_size[g] * [sx, sy, sz])
                   for g in feet for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        q = q.copy()
        q[2] -= min(c[2] for c in corners)
        d.qpos[:] = q
        mujoco.mj_forward(mjm, d)
        hands = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in HAND_GEOMS]
        faces = [d.geom_xpos[g] + side * PALM_HALF * d.geom_xmat[g].reshape(3, 3)[:, 1] for g, side in zip(hands, (-1, 1))]
        box = (faces[0] + faces[1]) / 2 - q[:3]
        if on_floor:
            box[2] = BOX_SIZE[1] / 2 - q[2]
        return q, box

    # ---- simulation ----

    def physics(self):
        # SUBSTEPS physics steps, then forward kinematics / collisions / sensors of the new state (for observing)
        for _ in range(SUBSTEPS):
            mjw.step(self.m, self.d)
        mjw.forward(self.m, self.d)

    def enable_cuda_graph(self):
        # the physics as one CUDA graph (capture warms up: re-reset after). torch and warp share one non-default stream
        # from here on (the default stream can't be captured; one stream keeps torch's writes and the graph ordered)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        wp.set_stream(wp.stream_from_torch(stream))
        self.physics()
        with wp.ScopedCapture() as capture:
            self.physics()
        self.graph = capture.graph
        self.reset(torch.ones(self.E, dtype=torch.bool, device=self.device))

    def simulate(self):
        if self.graph is None:
            self.physics()
        else:
            wp.capture_launch(self.graph)

    def is_stage(self, name):
        return self.stage == STAGES.index(name)

    def reset(self, mask):
        # new episodes in envs where mask (E,): standing pose at the origin facing a random way, a stage, a walk command,
        # the box and pad laid out for the stage
        envs = mask.nonzero().view(-1)
        n = len(envs)
        if n == 0:
            return
        rand = lambda *shape: torch.rand(*shape, generator=self.gen, device=self.device)
        stage = torch.multinomial(self.stage_probs, n, replacement=True, generator=self.gen)
        self.stage[envs] = stage
        yaw = (rand(n) * 2 - 1) * math.pi
        zero = torch.zeros(n, device=self.device)
        test = rand(n) < self.test_prob
        self.test[envs] = test
        # start pose: crouched at the box (START_CROUCH of pick, START_PLACED of deliver: the box on the pad), on a lift
        # rung (START_LIFTED of pick, START_LOWERED of deliver: over the pad), holding it (START_HELD of deliver), else
        # (and in pass tests) standing
        pick, start = (stage == STAGES.index('pick')) & ~test, rand(n)
        deliver_start = (stage == STAGES.index('deliver')) & ~test
        placed = deliver_start & (start < START_PLACED)
        lowered = deliver_start & (start >= START_PLACED) & (start < START_PLACED + START_LOWERED)
        held = deliver_start & (start >= START_PLACED + START_LOWERED) & (start < START_PLACED + START_LOWERED + START_HELD)
        crouch = pick & (start < START_CROUCH) | placed
        lifted = pick & (start >= START_CROUCH) & (start < START_CROUCH + START_LIFTED) | lowered
        archived = pick & (start >= START_CROUCH + START_LIFTED) & (start < START_CROUCH + START_LIFTED + START_ARCHIVE) & (self.archive_n > 0)
        rung = torch.randint(len(LIFT_U), (n,), generator=self.gen, device=self.device)
        base = torch.where(crouch[:, None], self.crouch_qpos, torch.where(held[:, None], self.held_qpos, self.init_qpos))
        base = torch.where(lifted[:, None], self.lift_qpos[rung], base)
        q = base.clone()
        q_yaw = quat_from_rotvec(torch.stack([zero, zero, yaw], -1))
        q[:, 3:7] = quat_mul(q_yaw, base[:, 3:7])
        # the box: pick / deliver START_DIST ahead within START_YAW, any heading; else away. +x turned by yaw points
        # along (cos yaw, sin yaw)
        ahead = yaw + (rand(n) * 2 - 1) * START_YAW
        dist = START_DIST[0] + (START_DIST[1] - START_DIST[0]) * rand(n)
        uses_box = stage > 0
        bx = torch.where(uses_box, dist * torch.cos(ahead), AWAY)
        by = torch.where(uses_box, dist * torch.sin(ahead), 0.0)
        q[:, self.box_q:self.box_q + 3] = torch.stack([bx, by, torch.full_like(bx, BOX_SIZE[1] / 2)], -1)
        q[:, self.box_q + 3:self.box_q + 7] = quat_from_rotvec(torch.stack([zero, zero, rand(n) * 2 * math.pi], -1))
        # crouched / lifted / held starts: the box between the palms, square to the robot (the pelvis stays at the origin)
        offset = torch.where(crouch[:, None], self.crouch_box, torch.where(lifted[:, None], self.lift_box[rung], self.held_box))
        start_box = torch.cat([quat_rotate(q_yaw, offset)[:, :2], offset[:, 2:3]], -1)
        in_hands = crouch | lifted | held
        q[:, self.box_q:self.box_q + 3] = torch.where(in_hands[:, None], start_box + torch.cat([q[:, :2], base[:, 2:3]], -1), q[:, self.box_q:self.box_q + 3])
        q[:, self.box_q + 3:self.box_q + 7] = torch.where(in_hands[:, None], q_yaw, q[:, self.box_q + 3:self.box_q + 7])
        self.qpos[envs] = q
        self.qvel[envs] = 0.0
        if archived.any():
            # (n,) an archived state per env (only the archived envs take it)
            pick_from = torch.randint(self.archive_n, (n,), generator=self.gen, device=self.device)
            self.qpos[envs] = torch.where(archived[:, None], self.archive_q[pick_from], q)
            self.qvel[envs] = torch.where(archived[:, None], self.archive_v[pick_from], 0.0)
        # the pad: deliver PAD_DIST from the box (held starts: 0 to PAD_DIST's top; placed, lowered: 0 to PLACED_PAD), any
        # direction; else away
        pad_dir = rand(n) * 2 * math.pi
        pad_lo = torch.where(held | placed | lowered, 0.0, PAD_DIST[0])
        pad_hi = torch.where(placed | lowered, PLACED_PAD, PAD_DIST[1])
        pad_dist = pad_lo + (pad_hi - pad_lo) * rand(n)
        deliver = stage == STAGES.index('deliver')
        px = torch.where(deliver, q[:, self.box_q] + pad_dist * torch.cos(pad_dir), -AWAY)
        py = torch.where(deliver, q[:, self.box_q + 1] + pad_dist * torch.sin(pad_dir), 0.0)
        self.mocap_pos[envs, 0] = torch.stack([px, py, torch.full_like(px, 0.0025)], -1)
        self.last_action[envs] = 0.0
        self.prev_action[envs] = 0.0
        self.prev2_action[envs] = 0.0
        self.phase[envs] = 0.0
        self.steps[envs] = 0
        self.air_time[envs] = 0.0
        self.last_contact[envs] = False
        self.streak[envs] = 0
        self.dash[envs] = (stage == 0) & test
        self.ep_steps[envs] = torch.where(self.dash[envs], DASH_STEPS, self.stage_steps[stage])
        self.dash_jerk[envs] = 0.0
        self.new_command(mask)
        self.dash_start[envs] = q[:, :2]
        self.dash_heading[envs] = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
        # kinematics of the new state (positions read by observe())
        mjw.forward(self.m, self.d)
        self.last_feet[envs] = self.xpos[envs][:, self.feet]
        self.potential = torch.where(mask, self.task_potential(), self.potential)

    def new_command(self, mask):
        # a grid walk command for envs where mask (E,) (zero outside walk, the dash's in a dash)
        steps = self.cmd_lo + torch.floor(torch.rand(self.E, 1, 3, generator=self.gen, device=self.device) * (self.cmd_hi - self.cmd_lo + 1)).long()
        stand = (torch.rand(self.E, 1, 1, generator=self.gen, device=self.device) < STAND_PROB) | ~self.is_stage('walk')[:, None, None]
        cmd = torch.where(stand, 0.0, steps * COMMAND_STEP)
        cmd = torch.where(self.dash[:, None, None], const([DASH_SPEED, 0.0, 0.0], self.device), cmd)
        self.command = torch.where(mask[:, None, None], cmd, self.command)
        # heading mode: dashes hold their start heading, HEADING_PROB of the others a random one
        rand = torch.rand(self.E, generator=self.gen, device=self.device)
        heading = (rand < HEADING_PROB) | self.dash
        fwd, _ = self.heading()
        yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
        offset = torch.where(self.dash, 0.0, (torch.rand(self.E, generator=self.gen, device=self.device) * 2 - 1) * math.pi)
        self.heading_mode = torch.where(mask, heading & self.is_stage('walk'), self.heading_mode)
        self.target_yaw = torch.where(mask, yaw + offset, self.target_yaw)
        self.steer()

    def steer(self):
        # heading mode: the turn command from the heading error
        fwd, _ = self.heading()
        error = torch.remainder(self.target_yaw - torch.atan2(fwd[:, 1], fwd[:, 0]) + math.pi, 2 * math.pi) - math.pi
        turn = (HEADING_GAIN * error).clamp(COMMAND_RANGES[2][0], COMMAND_RANGES[2][1])
        self.command[:, 0, 2] = torch.where(self.heading_mode, turn, self.command[:, 0, 2])

    # ---- the robot's state ----

    def quat(self):
        # (E, 4) pelvis orientation (the free joint's)
        return self.qpos[:, 3:7]

    def gravity(self):
        # (E, 3) unit gravity in the pelvis frame
        return quat_rotate_inv(self.quat(), const([0.0, 0.0, -1.0], self.device).expand(self.E, 3))

    def gyro(self):
        # (E, 3) pelvis angular velocity in the pelvis frame (the IMU's gyro)
        return self.sensordata[:, self.gyro_adr:self.gyro_adr + 3]

    def heading(self):
        # (E, 2) forward and (E, 2) left unit vectors of the pelvis on the ground (x, y)
        fwd = quat_rotate(self.quat(), const([1.0, 0.0, 0.0], self.device).expand(self.E, 3))[:, :2]
        fwd = fwd / torch.linalg.norm(fwd, dim=-1, keepdim=True).clamp_min(1e-6)
        return fwd, torch.stack([-fwd[:, 1], fwd[:, 0]], -1)

    def in_heading(self, v):
        # (E, ..., 3) world vectors -> [forward, left, up] in the heading frame
        fwd, left = self.heading()
        shape = (self.E,) + (1,) * (v.dim() - 2) + (2,)
        return torch.stack([(v[..., :2] * fwd.view(shape)).sum(-1), (v[..., :2] * left.view(shape)).sum(-1), v[..., 2]], -1)

    def base_velocity(self):
        # (E, 3) [forward, left] ground velocity in the heading frame, yaw rate (rad/s, + = left)
        v = self.in_heading(self.qvel[:, :3])
        # the free joint's angular velocity is in the body frame: to the world frame
        w = quat_rotate(self.quat(), self.qvel[:, 3:6])
        return torch.stack([v[:, 0], v[:, 1], w[:, 2]], -1)

    def height(self):
        # (E,) pelvis height
        return self.qpos[:, 2]

    def joint_pos(self):
        return self.qpos[:, self.act_q] - self.rest

    def joint_vel(self):
        return self.qvel[:, self.act_v]

    def foot_contact(self):
        # (E, 2) foot touching the floor with more than CONTACT_FORCE
        f = self.sensordata[:, self.foot_adr[:, None] + torch.arange(1, 4, device=self.device)]
        return torch.linalg.norm(f, dim=-1) > CONTACT_FORCE

    def foot_height(self):
        return self.xpos[:, self.feet, 2] - self.foot_rest

    def palms(self):
        # (E, 2, 3) each hand's palm face center (the hand box's -y face on the left hand, +y on the right)
        side = const([-PALM_HALF, PALM_HALF], self.device)
        y_axis = self.geom_xmat[:, self.hand_geoms, :, 1]
        return self.geom_xpos[:, self.hand_geoms] + side[None, :, None] * y_axis

    def grip(self):
        # (E, 2) each palm's contact force on the box (N)
        f = self.sensordata[:, self.palm_adr[:, None] + torch.arange(1, 4, device=self.device)]
        return torch.linalg.norm(f, dim=-1)

    def box_bottom(self):
        # (E,) height of the box's lowest corner
        corners = const([[sx, sy, sz] for sx in (-1., 1.) for sy in (-1., 1.) for sz in (-1., 1.)], self.device) * \
            const([BOX_SIZE[0], BOX_SIZE[2], BOX_SIZE[1]], self.device) / 2
        q = self.xquat[:, self.box][:, None].expand(self.E, 8, 4)
        return (self.xpos[:, self.box, None, 2] + quat_rotate(q, corners.expand(self.E, 8, 3))[..., 2]).amin(-1)

    def fallen(self):
        # (E,) a trunk part on the floor, or the pelvis below / tipped past the stage's limits
        trunk = self.sensordata[:, self.trunk_adr] > 0
        tipped = -self.gravity()[:, 2] < torch.cos(self.fall_tilt[self.stage])
        return trunk.any(-1) | (self.height() < self.fall_height[self.stage]) | tipped

    def proprio(self, noise):
        # (E, PROPRIO_DIM) see the header
        parts = {'gyro': self.gyro() * GYRO_SCALE, 'gravity': self.gravity(), 'joint_pos': self.joint_pos(),
                 'joint_vel': self.joint_vel() * JOINT_VEL_SCALE}
        if noise:
            parts = {k: v + (torch.rand(v.shape, generator=self.gen, device=self.device) * 2 - 1) * NOISE[k] for k, v in parts.items()}
        clock = torch.stack([torch.sin(2 * math.pi * self.phase[:, 0]), torch.cos(2 * math.pi * self.phase[:, 0])], -1)
        return torch.cat(list(parts.values()) + [self.last_action[:, 0], clock], -1)

    def task(self):
        # (E, TASK_DIM) see the header
        S = len(STAGES)
        onehot = (self.stage[:, None] == torch.arange(S, device=self.device)).float()
        pelvis = self.qpos[:, :3]
        box = self.in_heading(self.xpos[:, self.box] - pelvis)
        axes = quat_rotate(self.xquat[:, self.box][:, None].expand(self.E, 2, 4), const([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], self.device).expand(self.E, 2, 3))
        box_v = self.in_heading(self.qvel[:, self.box_v:self.box_v + 3])
        pad = self.in_heading(self.mocap_pos[:, 0] - pelvis)
        uses_box = (self.stage > 0)[:, None].float()
        uses_pad = self.is_stage('deliver')[:, None].float()
        return torch.cat([onehot, self.command[:, 0], box * uses_box, self.in_heading(axes).flatten(1) * uses_box, box_v * uses_box, pad * uses_pad], -1)

    def observe(self, noise=True):
        # {"proprio", "command", "privileged", "aux"}, each (E, 1, dim): see the header
        vel, height = self.base_velocity(), self.height()[:, None]
        contact = self.foot_contact().float()
        foot_speed = torch.linalg.norm(self.xpos[:, self.feet, :2] - self.last_feet[..., :2], dim=-1) / DT
        feet = torch.cat([contact, self.foot_height(), foot_speed], -1)
        task = self.task()
        palms = self.in_heading(self.palms() - self.qpos[:, None, :3]).flatten(1)
        priv = torch.cat([self.proprio(noise=False), task, vel, height, feet, palms, self.grip() / 50.0], -1)
        out = {'proprio': self.proprio(noise) if noise else self.proprio(False), 'command': task, 'privileged': priv,
               'aux': torch.cat([vel, height, feet[:, :4]], -1)}
        return {k: v[:, None] for k, v in out.items()}

    # ---- rewards ----

    def rewards(self):
        # {name in MJ_TERMS: (E,) unweighted locomotion term} (robot/humanoid.py rewards()). updates the feet's air time
        vel = self.base_velocity()
        q = self.joint_pos()
        contact = self.foot_contact()
        legs = torch.stack([self.phase[:, 0], (self.phase[:, 0] + 0.5) % 1.0], -1)
        x = (2 * legs) % 1.0
        bezier = x ** 3 + 3 * x ** 2 * (1 - x)
        # (E,) the box stages step only on the way to their target (the box; in deliver the pad once the box is held):
        # within PLANT_DIST of it the reference keeps both feet planted (a crouch can't step). PLANT_DIST (m) is past
        # REACH and the floor grasp's 0.5 m pelvis-to-box distance
        PLANT_DIST = 0.7
        held = self.is_stage('deliver') & (self.grip() > GRIP_MIN).all(-1)
        target = torch.where(held[:, None], self.mocap_pos[:, 0], self.xpos[:, self.box])
        far = torch.linalg.norm((target - self.qpos[:, :3])[:, :2], dim=-1) > PLANT_DIST
        stepping = self.is_stage('walk') | far
        ref = SWING_HEIGHT * torch.where(legs < 0.5, bezier, 1 - bezier) * stepping[:, None]
        touchdown = contact & ~self.last_contact[:, 0]
        air_time = self.air_time[:, 0] + DT
        air = ((air_time - AIR_TIME[0]).clamp(max=AIR_TIME[1] - AIR_TIME[0]) * touchdown).sum(-1)
        self.air_time = torch.where(contact, 0.0, air_time)[:, None]
        self.last_contact = contact[:, None]
        cmd = self.command[:, 0]
        moving = torch.linalg.norm(cmd, dim=-1) > STILL
        # deliver carrying the box to a pad beyond PLANT_DIST: velocity tracking toward the pad at CARRY_TOWARD (m/s,
        # heading frame; the policy squatted with the box and the pad shaping alone, 1 per m, never got it walking).
        # (E, 2) the tracked [forward, left] velocity, (E,) where linear tracking counts (walk, carrying)
        CARRY_TOWARD = 0.5
        carrying = held & far
        to_pad = self.in_heading(self.mocap_pos[:, 0] - self.qpos[:, :3])[:, :2]
        carry_cmd = CARRY_TOWARD * to_pad / torch.linalg.norm(to_pad, dim=-1, keepdim=True).clamp_min(1e-6)
        lin_cmd = torch.where(carrying[:, None], carry_cmd, cmd[:, :2])
        tracked = self.is_stage('walk') | carrying
        foot_speed = torch.linalg.norm(self.xpos[:, self.feet, :2] - self.last_feet[..., :2], dim=-1) / DT
        return {'track_lin_vel': torch.exp(-((lin_cmd - vel[:, :2]) ** 2).sum(-1) / TRACKING_SIGMA) * tracked,
                'track_ang_vel': torch.exp(-(cmd[:, 2] - vel[:, 2]) ** 2 / TRACKING_SIGMA),
                'ang_vel_xy': (self.gyro()[:, :2] ** 2).sum(-1),
                'orientation': (self.gravity()[:, :2] ** 2).sum(-1) * stepping,
                'base_height': (self.qpos[:, 2] - STAND_HEIGHT) ** 2 * stepping,
                'dof_pos_limits': ((self.limit_lo - q).clamp_min(0) + (q - self.limit_hi).clamp_min(0)).sum(-1),
                'hip_pos': (q[:, self.hip] ** 2).sum(-1),
                'feet_slip': (foot_speed ** 2 * contact).sum(-1),
                'feet_phase': torch.exp(-((self.foot_height() - ref) ** 2).sum(-1) / 0.01),
                'feet_air_time': air * moving,
                'stand_still': q.abs().sum(-1) * ~moving,
                'alive': torch.ones(self.E, device=self.device),
                'upper_pose': (q[:, self.upper] ** 2).sum(-1)}

    def task_potential(self):
        # (E,) robot/warehouse.py task_potential() for the one box and pad: approach, palms on the box's side faces
        # across the robot, grip, lift (LIFT_HEIGHT in pick, CARRY_LIFT in deliver), then in deliver the pad (closer;
        # over it the lift counts full and lowering the box pays; resting on it the pick part counts full, and letting
        # go pays). 0 in walk
        box = self.xpos[:, self.box]
        approach = (torch.linalg.norm((box - self.qpos[:, :3])[:, :2], dim=-1) - REACH).clamp_min(0)
        _, left2 = self.heading()
        left = torch.cat([left2, torch.zeros(self.E, 1, device=self.device)], -1)
        q = self.xquat[:, self.box]
        ax = quat_rotate(q, const([1.0, 0.0, 0.0], self.device).expand(self.E, 3))
        ay = quat_rotate(q, const([0.0, 1.0, 0.0], self.device).expand(self.E, 3))
        use_x = (ax * left).sum(-1).abs() > (ay * left).sum(-1).abs()
        axis = torch.where(use_x[:, None], ax * BOX_SIZE[0] / 2, ay * BOX_SIZE[2] / 2)
        axis = axis * torch.sign((axis * left).sum(-1, keepdim=True) + 1e-9)
        up = const([0.0, 0.0, GRASP_RAISE], self.device)
        targets = torch.stack([box + axis + up, box - axis + up], 1)
        hands = torch.linalg.norm(self.palms() - targets, dim=-1).sum(-1)
        grip = (self.grip().amin(-1) / GRIP_FORCE).clamp(max=1.0)
        bottom = self.box_bottom()
        lift_to = torch.where(self.is_stage('pick'), LIFT_HEIGHT, CARRY_LIFT)
        to_pad = torch.linalg.norm((box - self.mocap_pos[:, 0])[:, :2], dim=-1)
        # (E,) deliver with the box over the pad: setting it down. below CARRY_LIFT the lift shaping drained before
        # resting paid, and from the held start the policy never lowered the box onto a pad even 0.5 m off; here the lift
        # counts full and lowering pays PLACE_W from LOWER_FROM (m, the held start's box bottom) down to the floor, so it
        # meets the resting value
        LOWER_FROM = 0.75
        lowering = self.is_stage('deliver') & (to_pad < PAD_SIZE / 2 - PAD_IN)
        lift = torch.where(lowering, 1.0, (bottom.clamp_min(0) / lift_to).clamp(max=1.0)) * grip
        lower = PLACE_W * (1 - (bottom / LOWER_FROM).clamp(0.0, 1.0)) * lowering
        pick = GRIP_W * grip + LIFT_W * lift - APPROACH_W * approach - HAND_W * hands
        resting = (to_pad < PAD_SIZE / 2 - PAD_IN) & (bottom < REST)
        # resting on the pad, letting go pays per newton (RELEASE_W at no force on either palm, 0 from RELEASE_FORCE):
        # the policy squeezed with 40-80 N and success's < GRIP_MIN on both palms was too far to find by noise
        RELEASE_W, RELEASE_FORCE = 2.0, 80.0
        release = RELEASE_W * (1 - (self.grip().amax(-1) / RELEASE_FORCE).clamp(max=1.0))
        place = torch.where(resting, GRIP_W + LIFT_W + PLACE_W + release, pick + lower) - PAD_W * to_pad
        out = torch.where(self.is_stage('deliver'), place, pick)
        return torch.where(self.is_stage('walk'), 0.0, out)

    # ---- the step ----

    def step(self, action):
        # action (E, 1, A) -> reward (E, 1), done (E,), timeout (E,), info {"terms": {name: (E,)}, "metrics": {...}}
        # (robot/humanoid.py step()); done envs are already reset
        self.ctrl[:] = self.rest + ACTION_SCALE * action[:, 0]
        self.prev2_action, self.prev_action, self.last_action = self.prev_action, self.last_action, action
        if self.pushes:
            self.push_timer -= 1
            due = self.push_timer <= 0
            kick = (torch.rand(self.E, 2, generator=self.gen, device=self.device) * 2 - 1) * PUSH_SPEED * due[:, None]
            self.qvel[:, :2] += kick
            self.push_timer = torch.where(due, torch.randint(PUSH_INTERVAL // 2, PUSH_INTERVAL * 3 // 2, (self.E,), generator=self.gen, device=self.device), self.push_timer)
        self.simulate()
        self.steps += 1
        self.steer()
        walk = self.is_stage('walk')
        speed = torch.where(walk, torch.linalg.norm(self.command[:, 0, :2], dim=-1), torch.linalg.norm(self.base_velocity()[:, :2], dim=-1))
        self.phase = (self.phase + (GAIT_FREQ[0] + (GAIT_FREQ[1] - GAIT_FREQ[0]) * (speed / GAIT_FAST).clamp(max=1.0))[:, None] * DT) % 1.0
        raw = self.rewards()
        weighted = torch.stack([raw[k] for k in MJ_TERMS], -1) * self.weights[self.stage]
        reward = (weighted.sum(-1) * DT).clamp_min(0.0)
        fell = self.fallen()
        # the box stages: shaping, then success
        pot = self.task_potential()
        reward = reward + torch.where(walk, 0.0, pot - self.potential)
        self.potential = pot
        bottom, grip = self.box_bottom(), self.grip()
        gripped = (grip > GRIP_MIN).all(-1)
        pick, deliver = self.is_stage('pick'), self.is_stage('deliver')
        held = pick & (bottom > LIFT_HEIGHT) & gripped
        to_pad = (self.xpos[:, self.box, :2] - self.mocap_pos[:, 0, :2]).abs()
        on_pad = (to_pad < PAD_SIZE / 2 - PAD_IN).all(-1) & (bottom < REST)
        released = deliver & on_pad & (grip < GRIP_MIN).all(-1)
        self.streak = torch.where(held | released, self.streak + 1, 0)
        success = (pick & (self.streak >= HOLD_STEPS)) | (deliver & (self.streak >= RELEASE_STEPS))
        reward = reward + SUCCESS_REWARD * success - FALL_PENALTY * (fell & ~walk)
        # metrics, walk commands, episode ends
        vel = self.base_velocity()
        trained = walk & ~self.test
        lin_err = torch.linalg.norm(self.command[:, 0, :2] - vel[:, :2], dim=-1)
        jerk = ((self.last_action - 2 * self.prev_action + self.prev2_action) ** 2).mean((-2, -1))
        count = lambda m: m.sum().clamp_min(1)
        metrics = {'walk_lin_err': (lin_err * trained).sum() / count(trained),
                   'action_jerk': ((jerk * walk).sum() / count(walk)).sqrt(), 'falls': fell.float().mean(),
                   'pick_grip': (gripped & pick).float().sum() / count(pick), 'pick_lift': (bottom.clamp_min(0) * pick).sum() / count(pick)}
        resample = (self.steps % COMMAND_STEPS == 0) & walk & ~self.dash
        if resample.any():
            self.new_command(resample)
        timeout = (self.steps >= self.ep_steps) & ~fell & ~success
        done = fell | timeout | success
        # pass tests: the dash (walk), success (pick, deliver), counted when a test episode ends
        self.dash_jerk += jerk * self.dash
        covered = ((self.qpos[:, :2] - self.dash_start) * self.dash_heading).sum(-1)
        dash_pass = self.dash & timeout & (covered >= DASH_DIST) & ((self.dash_jerk / self.steps).sqrt() < DASH_SMOOTH)
        ended = self.test & done
        for i, name in enumerate(STAGES):
            metrics[f'tried_{name}'] = (ended & (self.stage == i)).float().sum()
            metrics[f'passed_{name}'] = ((dash_pass if name == 'walk' else ended & success) & (self.stage == i)).float().sum()
        info = {'terms': {k: weighted[:, i] for i, k in enumerate(MJ_TERMS)}, 'metrics': metrics}
        self.last_feet = self.xpos[:, self.feet].clone()
        # archive pick states with both palms gripping the box still on the floor (training episodes still running)
        found = (pick & ~self.test & gripped & (bottom < REST) & ~done).nonzero().view(-1)
        if len(found):
            found = found[torch.randperm(len(found), generator=self.gen, device=self.device)[:ARCHIVE_PER_STEP]]
            slots = (self.archive_i + torch.arange(len(found), device=self.device)) % ARCHIVE_SIZE
            self.archive_q[slots], self.archive_v[slots] = self.qpos[found], self.qvel[found]
            self.archive_i = (self.archive_i + len(found)) % ARCHIVE_SIZE
            self.archive_n = min(ARCHIVE_SIZE, self.archive_n + len(found))
        if done.any() and self.auto_reset:
            self.reset(done)
        return reward[:, None], done, timeout, info

# Unitree G1 humanoid (29 joints, 1.32 m, 35 kg), imported from MuJoCo Menagerie's g1_mjx.xml (BSD-3, Unitree):
# its masses, inertias, joint axes/limits, motor torque limits, PD gains and simplified collision capsules.
# frame: +x forward, +y up, +z to the robot's right (like LeKiwi). built standing with its feet on the y = 0 ground.
import math
from pathlib import Path
from robot.mjcf import load_mjcf

ASSETS = Path(__file__).parent / 'assets'
MODEL = ASSETS / 'unitree_g1' / 'g1_mjx.xml'
PREFIX = 'g1/'
# robot parts never collide with each other
GROUP = 1
# the menagerie floor contacts use friction 1
FRICTION = 1.0
# each foot touches the floor with its box; the three thin sole capsules overlapping it would only add contact points
SKIP_GEOM_CLASSES = ('foot_capsule',)
# each hand touches things with a box instead of the model's capsule: the rubber mitten is flat, and a capsule touches
# an object along a line it can pivot about (a box squeezed between two capsules spins). the box is the rubber hand
# mesh's bounds in the hand's frame (x along the fingers, palm facing -y on the left hand, +y on the right): 13 cm
# along the fingers, 6.7 cm thick, 10.6 cm wide. its palm side sticks out 1 cm past the wrist capsule, so a squeezed
# object meets the palm, not the wrist
GEOM_OVERRIDES = {'left_hand_collision': {'type': 'box', 'size': '0.0659 0.0333 0.0532', 'pos': '0.1074 -0.0116 0.0102'},
                  'right_hand_collision': {'type': 'box', 'size': '0.0659 0.0333 0.0532', 'pos': '0.1074 0.0116 0.0102'}}
# pelvis frame height in the standing pose: the menagerie "knees_bent" keyframe (0.755) raised until the foot boxes clear the floor
STAND_HEIGHT = 0.759
# {joint name: angle} standing pose, the menagerie "knees_bent" keyframe: knees bent, arms slightly out and bent
STAND = {'left_hip_pitch_joint': -0.312, 'left_knee_joint': 0.669, 'left_ankle_pitch_joint': -0.363,
         'right_hip_pitch_joint': -0.312, 'right_knee_joint': 0.669, 'right_ankle_pitch_joint': -0.363,
         'waist_pitch_joint': 0.073,
         'left_shoulder_pitch_joint': 0.2, 'left_shoulder_roll_joint': 0.22, 'left_elbow_joint': 1.0,
         'right_shoulder_pitch_joint': 0.2, 'right_shoulder_roll_joint': -0.22, 'right_elbow_joint': 1.0}
# head camera: the G1's RealSense D435 sits in the head, looking forward and down. position in the torso body's frame
# is set after import (see g1()); fov and resolution of the policy's view
CAMERA_FOV = 90
CAMERA_RES = 64
CAMERA_PITCH = math.radians(30)
# head camera position relative to the torso's center of mass, in our world axes at the standing pose (x forward, y up)
CAMERA_OFFSET = [0.08, 0.27, 0.0]

def quat_mul(a, b):
    # hamilton product of two python [w, x, y, z] lists
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def rotate_inv(q, v):
    # rotate python vector v by the inverse of unit quaternion q
    w, x, y, z = q[0], -q[1], -q[2], -q[3]
    u = [x, y, z]
    uv = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    uuv = [u[1] * uv[2] - u[2] * uv[1], u[2] * uv[0] - u[0] * uv[2], u[0] * uv[1] - u[1] * uv[0]]
    return [v[i] + 2 * (w * uv[i] + uuv[i]) for i in range(3)]


def g1(prefix=PREFIX, group=GROUP, color='#c9ccd1', pose=STAND):
    # {"bodies": [...], "joints": [...], "cameras": [...], "actuators": [joint id], "links": {...}} in the scene JSON
    # format, at the origin facing +x, built in pose ({joint name: angle}, legs bent so it stands; the rest pose its
    # actions are offsets from). several robots in one scene need different prefixes and collision groups
    robot = load_mjcf(MODEL, ASSETS, prefix, group, color, pose, [0.0, STAND_HEIGHT, 0.0], FRICTION, SKIP_GEOM_CLASSES, GEOM_OVERRIDES)
    torso = next(b for b in robot['bodies'] if b['id'] == prefix + 'torso_link')
    # camera looks down its local -z with +y up (three.js): turn -z onto +x, then pitch down about the right axis (+z)
    yaw = [math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0]
    pitch = [math.cos(-CAMERA_PITCH / 2), 0.0, 0.0, math.sin(-CAMERA_PITCH / 2)]
    world_rot = quat_mul(pitch, yaw)
    camera = {'id': prefix + 'head_cam', 'body': prefix + 'torso_link',
              'position': rotate_inv(torso['rotation'], CAMERA_OFFSET), 'rotation': quat_mul([torso['rotation'][0], *[-v for v in torso['rotation'][1:]]], world_rot),
              'fov': CAMERA_FOV, 'width': CAMERA_RES, 'height': CAMERA_RES}
    return {'bodies': robot['bodies'], 'joints': robot['joints'], 'cameras': [camera],
            'actuators': [prefix + j for j in robot['actuators']], 'links': robot['links']}

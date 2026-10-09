# ToFuBot: a 4-wheel cart (front wheels steer, rear wheels drive) with a turret + 2-joint arm and a parallel gripper.
# frame: +x forward, +y up, +z to the robot's right. built standing on the y = 0 ground, chassis centered at x = z = 0.
import math

PREFIX = 'tofubot/'
# robot parts never collide with each other (Isaac/MuJoCo default for RL robots)
GROUP = 1

WHEEL_RADIUS = 0.05
WHEEL_WIDTH = 0.03
# wheel centers: front / rear axle x, left / right z
AXLE_X = 0.10
TRACK_Z = 0.115
CHASSIS_SIZE = [0.30, 0.06, 0.18]
CHASSIS_Y = 0.07
TURRET_X = 0.03
TURRET_RADIUS = 0.04
TURRET_HEIGHT = 0.02
# shoulder pivot height above ground
SHOULDER_Y = CHASSIS_Y + CHASSIS_SIZE[1] / 2 + TURRET_HEIGHT + 0.015
UPPER_ARM = 0.16
FOREARM = 0.14
FINGER_SIZE = [0.02, 0.05, 0.01]
# finger opening per side: 0 = closed (fingers touch at the center), max = fully open
FINGER_OPEN = 0.035
# camera on the chassis front, local offset (x forward, y up), pitch down, vertical fov
CAMERA_LOCAL = [0.14, 0.10, 0.0]
CAMERA_PITCH = math.radians(40)
CAMERA_FOV = 100
CAMERA_RES = 64

# action order shared by the env, the expert, the policy and the live server. kind: what the action value means
#   steer (rad target), wheel (rad/s target), arm (rad target), finger (m target)
ACTUATORS = [
    ('steer_fl', 'steer'), ('steer_fr', 'steer'),
    ('drive_rl', 'wheel'), ('drive_rr', 'wheel'),
    ('turret', 'arm'), ('shoulder', 'arm'), ('elbow', 'arm'),
    ('finger_l', 'finger'), ('finger_r', 'finger'),
]
STEER_LIMIT = 0.6
WHEEL_MAX_SPEED = 15.0
# {joint: (lower, upper)} revolute limits in rad (inside (-π, π))
ARM_LIMITS = {'turret': (-2.6, 2.6), 'shoulder': (-0.6, 1.9), 'elbow': (-0.2, 2.6)}


def quat_axis_angle(axis, angle):
    # [w, x, y, z] rotation of `angle` rad about unit `axis`
    s = math.sin(angle / 2)
    return [math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def quat_mul(a, b):
    # hamilton product of two python [w, x, y, z] lists
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def tofubot():
    # {"bodies": [...], "joints": [...], "cameras": [...]} in the scene JSON format, ids prefixed with "tofubot/"
    P = PREFIX
    identity = [1.0, 0.0, 0.0, 0.0]
    # cylinder axis is local +y, wheels spin about z: rotate +90° about x
    wheel_rot = quat_axis_angle([1, 0, 0], math.pi / 2)

    def part(name, shape, pos, density, color, rotation=identity, collide=True, friction=0.8):
        return {'id': P + name, 'shape': shape, 'position': pos, 'rotation': rotation, 'velocity': [0.0, 0.0, 0.0],
                'angular_velocity': [0.0, 0.0, 0.0], 'density': density, 'static': False, 'collide': collide,
                'group': GROUP, 'friction': friction, 'restitution': 0.0, 'color': color}

    def joint(name, kind, a, b, anchor, axis, limits=None, drive=None):
        return {'id': P + name, 'type': kind, 'body_a': P + a, 'body_b': P + b, 'anchor': anchor, 'axis': axis,
                'limits': limits, 'drive': drive}

    def position_drive(k, c, f):
        return {'mode': 'position', 'stiffness': k, 'damping': c, 'max_force': f, 'target': 0.0}

    wheel = {'type': 'cylinder', 'radius': WHEEL_RADIUS, 'height': WHEEL_WIDTH}
    tofu = '#f2ead3'
    dark = '#2b2b2b'
    accent = '#e8833a'
    upper_y = SHOULDER_Y + UPPER_ARM / 2
    elbow_y = SHOULDER_Y + UPPER_ARM
    forearm_y = elbow_y + FOREARM / 2
    tip_y = elbow_y + FOREARM
    bodies = [
        part('chassis', {'type': 'box', 'size': CHASSIS_SIZE}, [0.0, CHASSIS_Y, 0.0], 460.0, tofu),
        # steering knuckles: small hidden bodies between chassis and front wheels (steer joint, then spin joint)
        part('knuckle_fl', {'type': 'box', 'size': [0.03, 0.03, 0.03]}, [AXLE_X, WHEEL_RADIUS, -TRACK_Z], 2000.0, dark, collide=False),
        part('knuckle_fr', {'type': 'box', 'size': [0.03, 0.03, 0.03]}, [AXLE_X, WHEEL_RADIUS, TRACK_Z], 2000.0, dark, collide=False),
        part('wheel_fl', wheel, [AXLE_X, WHEEL_RADIUS, -TRACK_Z], 500.0, dark, wheel_rot, friction=1.0),
        part('wheel_fr', wheel, [AXLE_X, WHEEL_RADIUS, TRACK_Z], 500.0, dark, wheel_rot, friction=1.0),
        part('wheel_rl', wheel, [-AXLE_X, WHEEL_RADIUS, -TRACK_Z], 500.0, dark, wheel_rot, friction=1.0),
        part('wheel_rr', wheel, [-AXLE_X, WHEEL_RADIUS, TRACK_Z], 500.0, dark, wheel_rot, friction=1.0),
        part('turret', {'type': 'cylinder', 'radius': TURRET_RADIUS, 'height': TURRET_HEIGHT},
             [TURRET_X, CHASSIS_Y + CHASSIS_SIZE[1] / 2 + TURRET_HEIGHT / 2, 0.0], 1000.0, accent),
        part('upper_arm', {'type': 'box', 'size': [0.03, UPPER_ARM, 0.03]}, [TURRET_X, upper_y, 0.0], 700.0, tofu),
        part('forearm', {'type': 'box', 'size': [0.025, FOREARM, 0.025]}, [TURRET_X, forearm_y, 0.0], 700.0, tofu),
        # fingers start closed: inner faces touching at z = 0
        part('finger_l', {'type': 'box', 'size': FINGER_SIZE}, [TURRET_X, tip_y + FINGER_SIZE[1] / 2, -FINGER_SIZE[2] / 2], 4000.0, accent, friction=1.2),
        part('finger_r', {'type': 'box', 'size': FINGER_SIZE}, [TURRET_X, tip_y + FINGER_SIZE[1] / 2, FINGER_SIZE[2] / 2], 4000.0, accent, friction=1.2),
    ]
    steer = position_drive(5.0, 0.2, 1.0)
    wheel_drive = {'mode': 'velocity', 'stiffness': 0.0, 'damping': 0.5, 'max_force': 0.4, 'target': 0.0}
    joints = [
        # steer about +y: positive turns the wheel's heading toward -z = left
        joint('steer_fl', 'revolute', 'chassis', 'knuckle_fl', [AXLE_X, WHEEL_RADIUS, -TRACK_Z], [0, 1, 0], [-STEER_LIMIT, STEER_LIMIT], dict(steer)),
        joint('steer_fr', 'revolute', 'chassis', 'knuckle_fr', [AXLE_X, WHEEL_RADIUS, TRACK_Z], [0, 1, 0], [-STEER_LIMIT, STEER_LIMIT], dict(steer)),
        # wheel spin about -z: positive rolls forward (+x)
        joint('spin_fl', 'revolute', 'knuckle_fl', 'wheel_fl', [AXLE_X, WHEEL_RADIUS, -TRACK_Z], [0, 0, -1]),
        joint('spin_fr', 'revolute', 'knuckle_fr', 'wheel_fr', [AXLE_X, WHEEL_RADIUS, TRACK_Z], [0, 0, -1]),
        joint('drive_rl', 'revolute', 'chassis', 'wheel_rl', [-AXLE_X, WHEEL_RADIUS, -TRACK_Z], [0, 0, -1], None, dict(wheel_drive)),
        joint('drive_rr', 'revolute', 'chassis', 'wheel_rr', [-AXLE_X, WHEEL_RADIUS, TRACK_Z], [0, 0, -1], None, dict(wheel_drive)),
        joint('turret', 'revolute', 'chassis', 'turret', [TURRET_X, CHASSIS_Y + CHASSIS_SIZE[1] / 2, 0.0], [0, 1, 0],
              list(ARM_LIMITS['turret']), position_drive(10.0, 0.5, 2.0)),
        # arm pitch about -z: positive leans forward (+x). zero pose = arm straight up
        joint('shoulder', 'revolute', 'turret', 'upper_arm', [TURRET_X, SHOULDER_Y, 0.0], [0, 0, -1],
              list(ARM_LIMITS['shoulder']), position_drive(40.0, 1.5, 4.0)),
        joint('elbow', 'revolute', 'upper_arm', 'forearm', [TURRET_X, elbow_y, 0.0], [0, 0, -1],
              list(ARM_LIMITS['elbow']), position_drive(20.0, 0.8, 2.0)),
        # fingers slide outward: left along -z, right along +z. value 0 = closed, FINGER_OPEN = open
        joint('finger_l', 'prismatic', 'forearm', 'finger_l', [TURRET_X, tip_y, -FINGER_SIZE[2] / 2], [0, 0, -1],
              [0.0, FINGER_OPEN], position_drive(100.0, 3.0, 3.0)),
        joint('finger_r', 'prismatic', 'forearm', 'finger_r', [TURRET_X, tip_y, FINGER_SIZE[2] / 2], [0, 0, 1],
              [0.0, FINGER_OPEN], position_drive(100.0, 3.0, 3.0)),
    ]
    # three.js camera convention: looks down local -z, up = local +y. yaw -90° so -z faces +x, then pitch down about +z
    cam_rot = quat_mul(quat_axis_angle([0, 0, 1], -CAMERA_PITCH), quat_axis_angle([0, 1, 0], -math.pi / 2))
    cameras = [{'id': P + 'cam', 'body': P + 'chassis', 'position': [CAMERA_LOCAL[0], CAMERA_LOCAL[1], CAMERA_LOCAL[2]],
                'rotation': cam_rot, 'fov': CAMERA_FOV, 'width': CAMERA_RES, 'height': CAMERA_RES}]
    return {'bodies': bodies, 'joints': joints, 'cameras': cameras}

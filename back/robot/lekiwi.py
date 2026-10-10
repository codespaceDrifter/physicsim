# LeKiwi (Hugging Face / SIGRobotics-UIUC): a round base on 3 omni wheels ("kiwi drive") with an SO-101 arm.
# frame: +x forward, +y up, +z to the robot's right. built standing on the y = 0 ground, base centered at x = z = 0.
#
# arm: joint positions, axes, limits and link masses from SO-ARM100/Simulation/SO101/so101_new_calib.urdf (forward
# kinematics of its zero pose), so joint angles match the real arm's calibration. links are boxes between joints.
# base: dimensions from the LeKiwi CAD (4" omni wheels at 120°, two plates 5 cm apart); masses from the real parts.
# every motor is a Feetech STS3215 servo: ~2 N·m usable torque, ~5.8 rad/s top speed at 12 V.
import math

PREFIX = 'lekiwi/'
# robot parts never collide with each other (Isaac/MuJoCo default for RL robots)
GROUP = 1

STS3215_TORQUE = 2.0
STS3215_SPEED = 5.8
WHEEL_RADIUS = 0.0508
WHEEL_WIDTH = 0.025
# horizontal distance from the base center to each wheel center
WHEEL_RING = 0.10
# angle of each wheel around the base, from +x toward +z (right). one wheel at the back, two at the front corners
WHEEL_ANGLES = {'wheel_back': math.pi, 'wheel_left': -math.pi / 3, 'wheel_right': math.pi / 3}
BASE_RADIUS = 0.125
BASE_HEIGHT = 0.06
# base bottom just above the wheel tops (the wheels hang under the bottom plate)
BASE_Y = 0.135
BASE_TOP = BASE_Y + BASE_HEIGHT / 2
# arm base_link origin on the top plate (forward offset from the base center)
ARM_MOUNT_X = 0.0
# front camera at the front edge of the base, below the folded arm so the arm stays out of its view
FRONT_CAMERA_LOCAL = [0.125, -0.025, 0.0]
FRONT_CAMERA_PITCH = math.radians(15)
CAMERA_FOV = 90
CAMERA_RES = 64

# SO-101 zero pose in URDF coordinates (z up, y left), from forward kinematics of so101_new_calib.urdf
URDF_JOINTS = {
    # name: (position, axis)
    'shoulder_pan': ((0.0388, 0.0, 0.0624), (0.0, 0.0, -1.0)),
    'shoulder_lift': ((0.0692, -0.0183, 0.1166), (0.0, 1.0, 0.0)),
    'elbow_flex': ((0.0972, -0.0183, 0.2292), (0.0, 1.0, 0.0)),
    'wrist_flex': ((0.2321, -0.0183, 0.2344), (0.0, 1.0, 0.0)),
    'wrist_roll': ((0.2932, -0.0002, 0.2344), (-1.0, 0.0, 0.0)),
    'gripper': ((0.3166, 0.0176, 0.2555), (0.0, -0.999, -0.049)),
}
# gripper tool point (between the jaw tips) in URDF coordinates
URDF_TOOL = (0.3914, 0.0, 0.2265)
# {joint: (lower, upper)} rad, from the URDF
ARM_LIMITS = {'shoulder_pan': (-1.91986, 1.91986), 'shoulder_lift': (-1.74533, 1.74533), 'elbow_flex': (-1.69, 1.69),
              'wrist_flex': (-1.65806, 1.65806), 'wrist_roll': (-2.74385, 2.84121), 'gripper': (-0.174533, 1.74533)}
# link masses (kg) from the URDF
ARM_MASSES = {'arm_base': 0.147, 'shoulder': 0.100, 'upper_arm': 0.103, 'forearm': 0.104, 'wrist': 0.079, 'gripper': 0.087, 'jaw': 0.012}
# the moving jaw is simulated at 60 g instead of its 12 g: through the servo's 345:1 gearbox the motor's rotor makes the
# jaw act far heavier about its hinge (MuJoCo calls this armature). at 12 g, soft contacts let a 5 N squeeze sink ~3 cm
# into the block and it squirts out
JAW_EFFECTIVE_MASS = 0.06

# folded rest pose the robot is built in and starts each episode in (arm tucked back over the base, so the
# center of mass stays inside the wheel triangle). the URDF zero pose sticks 39 cm forward and tips the base
REST = {'shoulder_pan': 0.0, 'shoulder_lift': -1.7, 'elbow_flex': 1.6, 'wrist_flex': 0.7, 'wrist_roll': 0.0, 'gripper': 0.0}
# battery (5.2 Ah LiPo) at the back of the top plate: real counterweight to the arm
BATTERY_MASS = 0.3

# action order shared by the env, the expert, the policy and the live server. kind: what the action value means
#   wheel (rad/s target), arm (rad target)
ACTUATORS = [('wheel_back', 'wheel'), ('wheel_left', 'wheel'), ('wheel_right', 'wheel'),
             ('shoulder_pan', 'arm'), ('shoulder_lift', 'arm'), ('elbow_flex', 'arm'), ('wrist_flex', 'arm'),
             ('wrist_roll', 'arm'), ('gripper', 'arm')]


def urdf_to_world(p):
    # URDF arm frame (x forward, y left, z up) -> our frame (x forward, y up, z right), mounted on the base top
    return [ARM_MOUNT_X + p[0], BASE_TOP + p[2], -p[1]]


def urdf_axis(a):
    return [a[0], a[2], -a[1]]


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


def rotate(q, v):
    # rotate python vector v by unit quaternion q
    w, x, y, z = q
    u = [x, y, z]
    uv = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    uuv = [u[1] * uv[2] - u[2] * uv[1], u[2] * uv[0] - u[0] * uv[2], u[0] * uv[1] - u[1] * uv[0]]
    return [v[i] + 2 * (w * uv[i] + uuv[i]) for i in range(3)]


def quat_y_to(d):
    # rotation taking local +y onto unit vector d (shortest arc)
    n = math.sqrt(sum(v * v for v in d))
    d = [v / n for v in d]
    dot = d[1]
    if dot < -0.999999:
        return [0.0, 1.0, 0.0, 0.0]
    # axis = y × d, quaternion = [1 + dot, axis] normalized
    q = [1.0 + dot, d[2], 0.0, -d[0]]
    qn = math.sqrt(sum(v * v for v in q))
    return [v / qn for v in q]


def lekiwi():
    # {"bodies": [...], "joints": [...], "cameras": [...]} in the scene JSON format, ids prefixed with "lekiwi/"
    P = PREFIX
    identity = [1.0, 0.0, 0.0, 0.0]
    white = '#f2efe6'
    dark = '#2b2b2b'
    accent = '#e8833a'

    def part(name, shape, pos, mass, volume, color, rotation=identity, friction=0.8):
        return {'id': P + name, 'shape': shape, 'position': [float(v) for v in pos], 'rotation': rotation,
                'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'density': mass / volume,
                'static': False, 'collide': True, 'group': GROUP, 'friction': friction, 'restitution': 0.0, 'color': color}

    def joint(name, kind, a, b, anchor, axis, limits=None, drive=None):
        return {'id': P + name, 'type': kind, 'body_a': P + a, 'body_b': P + b, 'anchor': [float(v) for v in anchor],
                'axis': [float(v) for v in axis], 'limits': limits, 'drive': drive}

    def servo(stiffness, damping):
        # STS3215 in position mode: a stiff spring-damper toward the commanded angle, capped at its torque
        return {'mode': 'position', 'stiffness': stiffness, 'damping': damping, 'max_force': STS3215_TORQUE,
                'max_velocity': STS3215_SPEED, 'target': 0.0}

    def link(name, start, end, width, depth, mass, color):
        # box spanning two points (world): local y along start -> end, width along local x, depth along local z
        d = [e - s for s, e in zip(start, end)]
        length = math.sqrt(sum(v * v for v in d))
        center = [(s + e) / 2 for s, e in zip(start, end)]
        return part(name, {'type': 'box', 'size': [width, length, depth]}, center, mass, width * length * depth, color, quat_y_to(d))

    # ---- base ----
    bodies = [part('base', {'type': 'cylinder', 'radius': BASE_RADIUS, 'height': BASE_HEIGHT}, [0.0, BASE_Y, 0.0], 1.5,
                   math.pi * BASE_RADIUS ** 2 * BASE_HEIGHT, white)]
    joints = []
    for name, angle in WHEEL_ANGLES.items():
        # axle points radially outward; the cylinder's local y is its axle
        axle = [math.cos(angle), 0.0, math.sin(angle)]
        center = [WHEEL_RING * axle[0], WHEEL_RADIUS, WHEEL_RING * axle[2]]
        bodies.append(part(name, {'type': 'cylinder', 'radius': WHEEL_RADIUS, 'height': WHEEL_WIDTH, 'omni': True}, center, 0.1,
                           math.pi * WHEEL_RADIUS ** 2 * WHEEL_WIDTH, dark, quat_y_to(axle), friction=1.0))
        # STS3215 in velocity mode
        joints.append(joint(name, 'revolute', 'base', name, center, axle, None,
                            {'mode': 'velocity', 'stiffness': 0.0, 'damping': 1.0, 'max_force': STS3215_TORQUE, 'target': 0.0}))

    # ---- SO-101 arm ----
    J = {n: urdf_to_world(p) for n, (p, a) in URDF_JOINTS.items()}
    tool = urdf_to_world(URDF_TOOL)
    bodies.append(part('arm_base', {'type': 'box', 'size': [0.07, 0.06, 0.05]}, urdf_to_world((0.0137, 0.0, 0.03)), ARM_MASSES['arm_base'],
                       0.07 * 0.06 * 0.05, white))
    bodies.append(part('shoulder', {'type': 'box', 'size': [0.05, 0.05, 0.045]}, urdf_to_world((0.0696, 0.0, 0.0877)), ARM_MASSES['shoulder'],
                       0.05 * 0.05 * 0.045, white))
    bodies.append(link('upper_arm', J['shoulder_lift'], J['elbow_flex'], 0.03, 0.035, ARM_MASSES['upper_arm'], white))
    bodies.append(link('forearm', J['elbow_flex'], J['wrist_flex'], 0.03, 0.035, ARM_MASSES['forearm'], white))
    bodies.append(link('wrist', J['wrist_flex'], J['wrist_roll'], 0.03, 0.035, ARM_MASSES['wrist'], white))
    # fixed jaw: the gripper body itself, from the wrist roll axis to the tool point, below the moving jaw
    bodies.append(part('gripper', {'type': 'box', 'size': [tool[0] - J['wrist_roll'][0], 0.016, 0.03]},
                       [(tool[0] + J['wrist_roll'][0]) / 2, tool[1], 0.0], ARM_MASSES['gripper'], 0.098 * 0.016 * 0.03, accent))
    # moving jaw: hinged above the fixed jaw, closed (touching) at angle 0, opens upward for positive angles
    jaw_len = tool[0] - J['gripper'][0]
    bodies.append(part('jaw', {'type': 'box', 'size': [jaw_len, 0.012, 0.03]}, [J['gripper'][0] + jaw_len / 2, tool[1] + 0.008 + 0.006 + 0.001, 0.0],
                       JAW_EFFECTIVE_MASS, jaw_len * 0.012 * 0.03, accent))
    joints.append(joint('arm_mount', 'fixed', 'base', 'arm_base', urdf_to_world((0.0, 0.0, 0.0)), [1.0, 0.0, 0.0]))
    bodies.append(part('battery', {'type': 'box', 'size': [0.05, 0.03, 0.12]}, [-0.08, BASE_TOP + 0.015, 0.0], BATTERY_MASS, 0.05 * 0.03 * 0.12, dark))
    joints.append(joint('battery_mount', 'fixed', 'base', 'battery', [-0.08, BASE_TOP, 0.0], [1.0, 0.0, 0.0]))
    # stiff like the real servo's position loop (sim sag under gravity < 1 cm at the tip)
    chain = [('shoulder_pan', 'arm_base', 'shoulder', servo(100.0, 2.0)), ('shoulder_lift', 'shoulder', 'upper_arm', servo(100.0, 2.0)),
             ('elbow_flex', 'upper_arm', 'forearm', servo(100.0, 2.0)), ('wrist_flex', 'forearm', 'wrist', servo(50.0, 1.0)),
             ('wrist_roll', 'wrist', 'gripper', servo(20.0, 0.4)),
             # gripper capped well below the servo's torque: a real 2 N·m squeeze would crush into a sim block
             ('gripper', 'gripper', 'jaw', {'mode': 'position', 'stiffness': 2.0, 'damping': 0.05, 'max_force': 0.25,
                                             'max_velocity': STS3215_SPEED, 'target': 0.0})]
    for name, a, b, drive in chain:
        joints.append(joint(name, 'revolute', a, b, J[name], urdf_axis(URDF_JOINTS[name][1]), list(ARM_LIMITS[name]), drive))

    # fold the arm into REST: from the base outward, each joint rotates every link and joint after it about its axis
    body_by_id = {x['id']: x for x in bodies}  # {id: body dict}
    joint_by_id = {x['id']: x for x in joints}  # {id: joint dict}
    for i, (name, a, b, drive) in enumerate(chain):
        jd = joint_by_id[P + name]
        jd['initial'] = REST[name]
        jd['drive']['target'] = REST[name]
        q = quat_axis_angle(jd['axis'], REST[name])
        pivot = jd['anchor']
        for later in [body_by_id[P + c[2]] for c in chain[i:]]:
            later['position'] = [pivot[k] + r for k, r in enumerate(rotate(q, [later['position'][k] - pivot[k] for k in range(3)]))]
            later['rotation'] = quat_mul(q, later['rotation'])
        for later in [joint_by_id[P + c[0]] for c in chain[i + 1:]]:
            later['anchor'] = [pivot[k] + r for k, r in enumerate(rotate(q, [later['anchor'][k] - pivot[k] for k in range(3)]))]
            later['axis'] = rotate(q, later['axis'])

    # ---- cameras (three.js convention: look down local -z, up = local +y) ----
    # front camera on the base's front edge, looking forward and slightly down
    front_rot = quat_mul(quat_axis_angle([0, 0, 1], -FRONT_CAMERA_PITCH), quat_axis_angle([0, 1, 0], -math.pi / 2))
    # wrist camera on the side of the gripper (like the SO-101's wrist mount), looking along the jaws past their tips
    wrist_rot = quat_mul(quat_axis_angle([0, 1, 0], math.radians(15)),
                         quat_mul(quat_axis_angle([0, 0, 1], -math.radians(10)), quat_axis_angle([0, 1, 0], -math.pi / 2)))
    cameras = [{'id': P + 'front_cam', 'body': P + 'base', 'position': FRONT_CAMERA_LOCAL, 'rotation': front_rot,
                'fov': CAMERA_FOV, 'width': CAMERA_RES, 'height': CAMERA_RES},
               {'id': P + 'wrist_cam', 'body': P + 'gripper', 'position': [-0.035, 0.02, 0.04], 'rotation': wrist_rot,
                'fov': CAMERA_FOV, 'width': CAMERA_RES, 'height': CAMERA_RES}]
    return {'bodies': bodies, 'joints': joints, 'cameras': cameras}

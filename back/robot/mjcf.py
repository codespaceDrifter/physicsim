# MuJoCo MJCF -> our scene format (bodies, joints). handles what robot models in MuJoCo Menagerie use: nested bodies
# with hinge joints (armature included), <default> classes (childclass / class inheritance), <inertial>, primitive
# collision geoms (sphere, box, capsule by fromto or size), mesh visual geoms and <position> actuators.
#
# visuals: each body gets "visuals": [{"mesh": STL path under mesh_root, "position", "rotation" (the mesh frame in
# the body frame), "color"}], drawn by viewers in place of the collision shapes. the physics ignores them.
#
# frames: MJCF is z-up (x forward, y left); our world is y-up (x forward, y up, z right). positions and directions map
# with CONVERT, body orientations as R_ours = CONVERT @ R_mjcf (the body's own axes are unchanged).
# each of our bodies sits at its link's center of mass with its principal axes (inertia stays diagonal), and the link's
# collision geoms are stored relative to that frame.
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# (3, 3) MJCF world coordinates -> ours: x -> x, y (left) -> -z, z (up) -> y
CONVERT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# (3, 3) our capsule axis is local y, MuJoCo's is local z: this rotation (+90° about x) takes y onto z
Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def quat_to_matrix(q):
    # [w, x, y, z] -> (3, 3) rotation matrix (columns = rotated axes)
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    # (3, 3) rotation matrix -> [w, x, y, z] with w >= 0 (Shepperd's method: branch on the largest diagonal term)
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) * (1 if q[0] >= 0 else -1)
    return [float(v) for v in q / np.linalg.norm(q)]


def axis_angle(axis, angle):
    # (3, 3) rotation of `angle` rad about unit `axis` (Rodrigues)
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def rotation_y_to(d):
    # (3, 3) shortest rotation taking +y onto unit vector d
    d = np.asarray(d, dtype=float) / np.linalg.norm(d)
    axis = np.cross([0.0, 1.0, 0.0], d)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3) if d[1] > 0 else axis_angle([1.0, 0.0, 0.0], math.pi)
    return axis_angle(axis, math.atan2(s, d[1]))


def floats(text):
    return [float(v) for v in text.split()]


def parse_defaults(root):
    # {class name: {tag: {attribute: value}}} with every class's attributes merged over its parent's.
    # the unnamed top-level <default> is class "main"
    classes = {}  # {str: {str: {str: str}}}

    def walk(elem, parent):
        name = elem.get('class', 'main')
        merged = {tag: dict(attrs) for tag, attrs in (classes[parent].items() if parent else [])}  # {tag: {attr: value}}
        for child in elem:
            if child.tag != 'default':
                merged.setdefault(child.tag, {}).update(child.attrib)
        classes[name] = merged
        for child in elem:
            if child.tag == 'default':
                walk(child, name)

    top = root.find('default')
    classes['main'] = {}
    if top is not None:
        walk(top, None)
    return classes


def load_mjcf(path, mesh_root, prefix, group, color, qpos, base_pos, friction, skip_geom_classes=(), geom_overrides=None):
    # path: the .xml file. mesh_root: visual mesh names are their paths relative to this directory. prefix: id prefix
    # of every body / joint. group: collision group (robot parts never collide with each other). color: the robot's
    # color (its bodies, and visual meshes that keep their class's default material; meshes given a material of their
    # own keep its color). qpos: {joint name: angle} the pose to build the robot in (missing joints = 0); joint
    # "initial" values and drive targets are set to it. base_pos: [x, y, z] (ours) of the root body's frame.
    # skip_geom_classes: collision geoms of these classes are left out. geom_overrides: {geom name: {MJCF attribute:
    # value}} replaces those attributes of a geom (a "fromto" is dropped unless given), e.g. a different shape.
    # returns {"bodies": [body dict], "joints": [joint dict], "actuators": [joint name in actuator order],
    #          "links": {body name: ([x, y, z], [w, x, y, z]) link frame (MJCF axes) in our world, for mounting things}}
    root = ET.parse(path).getroot()
    classes = parse_defaults(root)
    compiler = root.find('compiler')
    mesh_dir = Path(path).parent / (compiler.get('meshdir', '') if compiler is not None else '')
    asset = root.find('asset')
    # {mesh name: path under mesh_root}, {material name: "#rrggbb"}
    mesh_files = {m.get('name'): str((mesh_dir / m.get('file')).relative_to(mesh_root)) for m in asset.findall('mesh')}
    materials = {m.get('name'): '#' + ''.join(f'{round(255 * v):02x}' for v in floats(m.get('rgba'))[:3]) for m in asset.findall('material')}

    def attrs(elem, cls):
        # element attributes over its class defaults
        cls = elem.get('class', cls)
        assert cls in classes, f'unknown default class {cls}'
        out = dict(classes[cls].get(elem.tag, {}))  # {attr: value}
        out.update(elem.attrib)
        return out, cls

    # {joint name: (kp, kv)} from <position> actuators; [joint name] in actuator order
    drives = {}  # {str: (float, float)}
    actuators = []  # [str]
    for act in root.find('actuator'):
        assert act.tag == 'position', f'only position actuators are supported, got {act.tag}'
        a, _ = attrs(act, 'main')
        drives[a['joint']] = (float(a.get('kp', 1.0)), float(a.get('kv', 0.0)))
        actuators.append(a['joint'])

    bodies = []  # [body dict]
    joints = []  # [joint dict]
    links = {}  # {body name: (position, quaternion)}
    to_ours = lambda p: [float(v) for v in CONVERT @ p]

    def walk(elem, parent_id, parent_pos, parent_R, cls):
        name = elem.get('name')
        cls = elem.get('childclass', cls)
        pos = parent_pos + parent_R @ np.array(floats(elem.get('pos', '0 0 0')))
        R = parent_R @ quat_to_matrix(floats(elem.get('quat', '1 0 0 0')))
        hinges = [j for j in elem if j.tag == 'joint']
        assert len(hinges) <= 1, f'body {name}: one joint per body supported'
        assert parent_id is not None or elem.find('freejoint') is not None, f'root body {name} must be free'
        if hinges:
            j, _ = attrs(hinges[0], cls)
            assert j.get('type', 'hinge') == 'hinge', f"joint {j['name']}: only hinges are supported"
            anchor = pos + R @ np.array(floats(j.get('pos', '0 0 0')))
            axis = R @ np.array(floats(j['axis']))
            # pose the subtree: rotate this body (and so everything below it) by the joint angle about the joint axis
            angle = qpos.get(j['name'], 0.0)
            rot = axis_angle(axis, angle)
            pos = anchor + rot @ (pos - anchor)
            R = rot @ R
            lo, hi = floats(j['range']) if 'range' in j else (None, None)
            kp, kv = drives[j['name']]
            joints.append({'id': prefix + j['name'], 'type': 'revolute', 'body_a': parent_id, 'body_b': prefix + name,
                           'anchor': to_ours(anchor), 'axis': to_ours(axis), 'limits': [lo, hi] if lo is not None else None,
                           'initial': angle,
                           'drive': {'mode': 'position', 'stiffness': kp, 'damping': kv,
                                     'max_force': floats(j['actuatorfrcrange'])[1], 'target': angle}})

        links[name] = (to_ours(pos), matrix_to_quat(CONVERT @ R))
        # body frame = center of mass with principal axes
        inertial = elem.find('inertial')
        assert inertial is not None, f'body {name} has no <inertial>'
        com = pos + R @ np.array(floats(inertial.get('pos', '0 0 0')))
        R_com = R @ quat_to_matrix(floats(inertial.get('quat', '1 0 0 0')))
        # (3, 3) inertia in world axes. the joint's armature (motor rotor inertia reflected through the gearbox) is added
        # about the joint axis: the link resists turning about its hinge like the real geared motor does
        inertia = R_com @ np.diag(floats(inertial.get('diaginertia'))) @ R_com.T
        if hinges:
            inertia += float(j.get('armature', 0.0)) * np.outer(axis, axis) / (axis @ axis)
        # principal axes again, matched to the model's (same order and signs where the armature barely turns them, so
        # e.g. the pelvis frame stays the IMU's x forward, y left, z up), right-handed
        values, vectors = np.linalg.eigh(inertia)
        order = [int(np.argmax(np.abs(R_com[:, k] @ vectors))) for k in range(3)]
        assert sorted(order) == [0, 1, 2], f'body {name}: ambiguous principal axes after adding armature'
        principal = values[order]
        R_com = vectors[:, order] * np.sign(np.sum(vectors[:, order] * R_com, axis=0))
        R_com[:, 2] *= np.sign(np.linalg.det(R_com))
        shapes = []  # [shape dict] relative to the center-of-mass frame (links without collision geoms have none)
        visuals = []  # [visual dict] the same
        for g in elem.findall('geom'):
            a, gcls = attrs(g, cls)
            if g.get('name') in (geom_overrides or {}):
                a.pop('fromto', None)
                a.update(geom_overrides[g.get('name')])
            if a.get('type') == 'mesh':
                g_R = R_com.T @ R @ quat_to_matrix(floats(a.get('quat', '1 0 0 0')))
                g_pos = R_com.T @ (pos + R @ np.array(floats(a.get('pos', '0 0 0'))) - com)
                visuals.append({'mesh': mesh_files[a['mesh']], 'position': [float(v) for v in g_pos], 'rotation': matrix_to_quat(g_R),
                                'color': materials[a['material']] if 'material' in g.attrib else color})
                continue
            if a.get('group') == '2' or gcls in skip_geom_classes:
                continue
            size = floats(a['size'])
            if 'fromto' in a:
                ft = floats(a['fromto'])
                p0, p1 = np.array(ft[:3]), np.array(ft[3:])
                g_pos = (p0 + p1) / 2
                g_R = rotation_y_to(p1 - p0)
                length = float(np.linalg.norm(p1 - p0))
            else:
                g_pos = np.array(floats(a.get('pos', '0 0 0')))
                g_R = quat_to_matrix(floats(a.get('quat', '1 0 0 0')))
            kind = a.get('type', 'sphere')
            if kind == 'capsule' and 'fromto' not in a:
                # MuJoCo capsule along its local z, size = [radius, half length]
                g_R = g_R @ Y_TO_Z
                length = 2 * size[1]
            # geom frame in the body's COM frame
            local_pos = R_com.T @ (pos + R @ g_pos - com)
            local_R = R_com.T @ R @ g_R
            shape = {'type': kind, 'position': [float(v) for v in local_pos], 'rotation': matrix_to_quat(local_R)}
            if kind == 'sphere':
                shape['radius'] = size[0]
            elif kind == 'capsule':
                shape.update(radius=size[0], length=length)
            elif kind == 'box':
                shape['size'] = [2 * v for v in size]
            else:
                raise AssertionError(f'geom type {kind} not supported (body {name})')
            shapes.append(shape)
        bodies.append({'id': prefix + name, 'shapes': shapes, 'visuals': visuals, 'position': to_ours(com),
                       'rotation': matrix_to_quat(CONVERT @ R_com), 'velocity': [0.0, 0.0, 0.0],
                       'angular_velocity': [0.0, 0.0, 0.0], 'mass': float(inertial.get('mass')),
                       'inertia': [float(v) for v in principal], 'static': False, 'collide': True, 'group': group,
                       'friction': friction, 'restitution': 0.0, 'color': color})
        for child in elem.findall('body'):
            walk(child, prefix + name, pos, R, cls)

    for top in root.find('worldbody').findall('body'):
        # parent frame placed so the root body's own frame lands on base_pos
        walk(top, None, CONVERT.T @ np.array(base_pos, dtype=float) - np.array(floats(top.get('pos', '0 0 0'))), np.eye(3), 'main')
    return {'bodies': bodies, 'joints': joints, 'actuators': actuators, 'links': links}


# ---- compiled MuJoCo models -> our scenes ----
# any MjModel (scenes built from includes, assets, procedural objects) shown in our viewer: every visible geom
# (group < GEOM_GROUPS_SHOWN, alpha > 0) becomes one static body of ours at the geom's frame, so a geom keeps its own
# color and meshes and primitives mix freely. primitives become shapes (box: full extents; capsule, cylinder: our axis
# is local y, MuJoCo's local z), meshes become visuals whose triangles come from the compiled model, named
# "mujoco/{mesh name}_{content hash}.stl" (a path under robot/assets/: the same mesh in many scenes is one file).
# planes (normal: MuJoCo local z, ours local y) take their frame turned by Y_TO_Z, since viewers draw a plane body along
# its own y.
# MuJoCo geom types (mjtGeom): plane 0, sphere 2, capsule 3, cylinder 5, box 6, mesh 7
GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_BOX, GEOM_MESH = 0, 2, 3, 5, 6, 7
# MuJoCo's default visualizer shows geom groups 0-2; menagerie models put collision geoms in group 3
GEOM_GROUPS_SHOWN = 3


def geom_scene(m, d, name):
    # MjModel m, MjData d (after kinematics: the bodies' poses) -> (scene: our scene JSON, meshes: {mesh name: ((V, 3)
    # float32, (T, 3) uint32)}, geoms: [int] the geom ids in scene body order)
    def rgba(g):
        return m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    geoms = [g for g in range(m.ngeom) if m.geom_group[g] < GEOM_GROUPS_SHOWN and rgba(g)[3] > 0]  # [int]
    identity = [1.0, 0.0, 0.0, 0.0]
    axis_y = matrix_to_quat(Y_TO_Z)
    bodies = []  # [scene body]
    meshes = {}  # {str: (vertices, triangles)}
    for g in geoms:
        kind, size = int(m.geom_type[g]), m.geom_size[g]
        # (some models' rgba exceed 1)
        color = '#' + ''.join(f'{int(round(min(max(c, 0.0), 1.0) * 255)):02x}' for c in rgba(g)[:3])
        body = {'id': f'geom{g}', 'shapes': [], 'position': [0.0, 0.0, 0.0], 'rotation': identity, 'color': color,
                'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'density': 1000.0, 'static': True,
                'collide': True, 'group': 0, 'friction': 1.0, 'restitution': 0.0}
        if kind == GEOM_MESH:
            mesh = int(m.geom_dataid[g])
            v0, nv, f0, nf = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh], m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            vertices, triangles = m.mesh_vert[v0:v0 + nv].astype(np.float32), m.mesh_face[f0:f0 + nf].astype(np.uint32)
            mesh_name = f'mujoco/{m.mesh(mesh).name}_{hashlib.sha1(vertices.tobytes() + triangles.tobytes()).hexdigest()[:8]}.stl'
            meshes[mesh_name] = (vertices, triangles)
            # the viewer draws visuals in place of shapes: a stand-in shape keeps the body valid
            body['shapes'] = [{'type': 'sphere', 'radius': 0.01, 'position': [0.0, 0.0, 0.0], 'rotation': identity}]
            body['visuals'] = [{'mesh': mesh_name, 'position': [0.0, 0.0, 0.0], 'rotation': identity, 'color': color}]
        elif kind == GEOM_PLANE:
            body['shapes'] = [{'type': 'plane', 'position': [0.0, 0.0, 0.0], 'rotation': identity}]
        elif kind == GEOM_SPHERE:
            body['shapes'] = [{'type': 'sphere', 'radius': float(size[0]), 'position': [0.0, 0.0, 0.0], 'rotation': identity}]
        elif kind == GEOM_BOX:
            body['shapes'] = [{'type': 'box', 'size': [float(2 * s) for s in size], 'position': [0.0, 0.0, 0.0], 'rotation': identity}]
        elif kind == GEOM_CAPSULE:
            body['shapes'] = [{'type': 'capsule', 'radius': float(size[0]), 'length': float(2 * size[1]), 'position': [0.0, 0.0, 0.0], 'rotation': axis_y}]
        elif kind == GEOM_CYLINDER:
            body['shapes'] = [{'type': 'cylinder', 'radius': float(size[0]), 'height': float(2 * size[1]), 'position': [0.0, 0.0, 0.0], 'rotation': axis_y}]
        else:
            assert False, f'geom {g} ({m.geom(g).name}): MuJoCo geom type {kind} not supported'
        bodies.append(body)
    for body, pose in zip(bodies, geom_poses(m, d, geoms)):
        body['position'], body['rotation'] = pose[:3].tolist(), pose[3:].tolist()
    scene = {'name': name, 'settings': {'dt': float(m.opt.timestep), 'substeps': 1, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0, 'max_contacts': 128},
             'bodies': bodies, 'joints': [], 'cameras': []}
    return scene, meshes, geoms


def geom_poses(m, d, geoms):
    # MjModel m, MjData d (after kinematics), geoms from geom_scene -> (N, 7) float32 [px, py, pz, qw, qx, qy, qz] per
    # scene body, in our axes
    planes = m.geom_type[geoms] == GEOM_PLANE
    out = np.zeros((len(geoms), 7), dtype=np.float32)
    for i, g in enumerate(geoms):
        R = CONVERT @ d.geom_xmat[g].reshape(3, 3)
        out[i, :3] = CONVERT @ d.geom_xpos[g]
        out[i, 3:] = matrix_to_quat(R @ Y_TO_Z if planes[i] else R)
    return out

# Base for LeKiwi tasks: one or more robots in a batched world. actions, observations, cameras, physics stepping.
# every tensor the tasks expose has a robot axis R: obs rgb (E, R, 6, H, W), proprio (E, R, 9), actions (E, R, 9).
import torch
from physics.world import World
from physics.render import Renderer
from physics.math3d import quat_mul, quat_rotate, quat_from_rotvec, const
from robot.lekiwi import ACTUATORS, STS3215_SPEED, ARM_LIMITS

# sized for the SO-101 gripper
BLOCK_SIZE = 0.035
DT = 1 / 60
SUBSTEPS = 4
# physics steps per control step: 6 * (1/60) s = 10 Hz control
ACTION_REPEAT = 6
# joint readings fed to the policy: arm joint angles + wheel speeds
PROPRIO_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
WHEELS = ['wheel_back', 'wheel_left', 'wheel_right']
PROPRIO_DIM = len(PROPRIO_JOINTS) + len(WHEELS)
# cameras stacked on the channel axis of each robot's "rgb" observation: front RGB, then wrist RGB
CAMERAS = ['front_cam', 'wrist_cam']
ACTION_DIM = len(ACTUATORS)


def static_box(id, size, pos, color, collide=True):
    # scene JSON body for an immovable box (ground markers, walls)
    return {'id': id, 'shape': {'type': 'box', 'size': size}, 'position': pos, 'rotation': [1.0, 0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0, 0.0], 'angular_velocity': [0.0, 0.0, 0.0], 'density': 1000.0, 'static': True,
            'collide': collide, 'group': 0, 'friction': 0.8, 'restitution': 0.0, 'color': color}


class RobotEnv:
    # prefixes: id prefix of each robot in the scene ("lekiwi/", "lekiwi2/", ...), in robot axis order
    def __init__(self, scene_json, num_envs, device, prefixes, seed):
        self.E = num_envs
        self.R = len(prefixes)
        self.device = device
        self.world = World(scene_json, num_envs, device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        w = self.world
        # [LongTensor] body indices of each robot's parts (moved together when a robot is placed)
        self.parts = [torch.tensor([i for i, bid in enumerate(w.body_ids) if bid.startswith(p)], device=device) for p in prefixes]
        # (R,) base body, (R, 2) the two gripper jaws (their mean is the grip point)
        self.base = torch.tensor([w.body_index[p + 'base'] for p in prefixes], device=device)
        self.jaws = torch.tensor([[w.body_index[p + 'gripper'], w.body_index[p + 'jaw']] for p in prefixes], device=device)
        # (R, A) joint of each actuator, (R, 6) arm joints read as proprio, (R, 3) wheel joints
        self.act_joints = torch.tensor([[w.joints.index[p + n] for n, kind in ACTUATORS] for p in prefixes], device=device)
        self.proprio_joints = torch.tensor([[w.joints.index[p + n] for n in PROPRIO_JOINTS] for p in prefixes], device=device)
        self.wheel_joints = torch.tensor([[w.joints.index[p + n] for n in WHEELS] for p in prefixes], device=device)
        # [[Renderer]] each robot's cameras in CAMERAS order
        cams = {c['id']: c for c in w.scene['cameras']}
        self.cameras = [[Renderer(w, cams[p + n]) for n in CAMERAS] for p in prefixes]
        # (A,) affine map from action in [-1, 1] to joint target: target = lo + (action + 1) / 2 * (hi - lo)
        lo = [-STS3215_SPEED if kind == 'wheel' else ARM_LIMITS[n][0] for n, kind in ACTUATORS]  # [float]
        hi = [STS3215_SPEED if kind == 'wheel' else ARM_LIMITS[n][1] for n, kind in ACTUATORS]  # [float]
        self.act_lo = torch.tensor(lo, device=device)
        self.act_hi = torch.tensor(hi, device=device)
        # (P,) range of each proprio joint (its action range), to normalize readings to [-1, 1]
        proprio_act = [[n for n, kind in ACTUATORS].index(n) for n in PROPRIO_JOINTS]
        self.proprio_lo = self.act_lo[proprio_act]
        self.proprio_hi = self.act_hi[proprio_act]

    def enable_cuda_graph(self):
        # physics step and every camera as CUDA graphs, then a fresh reset (capture warm-up stepped the world)
        self.world.enable_cuda_graph()
        for cams in self.cameras:
            for cam in cams:
                cam.enable_cuda_graph()
        self.reset(torch.ones(self.E, dtype=torch.bool, device=self.device))

    def action_to_targets(self, action):
        # (..., A) in [-1, 1] -> (..., A) joint targets
        return self.act_lo + (action.clamp(-1, 1) + 1) / 2 * (self.act_hi - self.act_lo)

    def targets_to_action(self, targets):
        # inverse of action_to_targets, used to record the expert's targets as actions
        return ((targets - self.act_lo) / (self.act_hi - self.act_lo) * 2 - 1).clamp(-1, 1)

    def apply(self, action):
        # (E, R, A) actions -> joint targets, then one control step of physics
        w = self.world
        w.joints.target[:, self.act_joints.reshape(-1)] = self.action_to_targets(action).reshape(self.E, -1)
        for _ in range(ACTION_REPEAT):
            w.step()

    def observe(self):
        # {"rgb": (E, R, 3 * len(CAMERAS), H, W) uint8 (front, wrist), "proprio": (E, R, PROPRIO_DIM) float}
        w = self.world
        q = (w.joints.value[:, self.proprio_joints] - self.proprio_lo) / (self.proprio_hi - self.proprio_lo) * 2 - 1
        wheels = w.joints.speeds()[:, self.wheel_joints] / STS3215_SPEED
        rgb = torch.stack([torch.cat([cam.render().permute(0, 3, 1, 2) for cam in cams], dim=1) for cams in self.cameras], dim=1)
        return {'rgb': rgb, 'proprio': torch.cat([q, wheels], -1)}

    def grip(self):
        # (E, R, 3) grip point of each robot (between the jaws)
        return self.world.pos[:, self.jaws].mean(2)

    def flipped(self):
        # (E, R) base up axis tipped past 60°
        up = quat_rotate(self.world.quat[:, self.base], const([0.0, 1.0, 0.0], self.device).expand(self.E, self.R, 3))
        return up[..., 1] < 0.5

    def place_robot(self, envs, r, xz, yaw):
        # move robot r rigidly in envs (n,): its base to (x, z) (n, 2), turned by yaw (n,) about +y from its scene pose
        w = self.world
        parts = self.parts[r]
        n, P = len(envs), len(parts)
        q_yaw = quat_from_rotvec(torch.stack([torch.zeros_like(yaw), yaw, torch.zeros_like(yaw)], -1))[:, None].expand(n, P, 4)
        base0 = w.init_pos[self.base[r]]
        rel = quat_rotate(q_yaw, (w.init_pos[parts] - base0).expand(n, P, 3))
        target = torch.stack([xz[:, 0], torch.full_like(xz[:, 0], float(base0[1])), xz[:, 1]], -1)
        w.pos[envs[:, None], parts] = rel + target[:, None]
        w.quat[envs[:, None], parts] = quat_mul(q_yaw, w.init_quat[parts].expand(n, P, 4))

    def sample_spots(self, n, k, lo, hi, min_gap, taken):
        # (n, k, 2) random xz points in the box [lo, hi] (2,), each at least min_gap from the others and from taken
        # (n, T, 2) points (rejection sampling per point)
        spots = taken
        for _ in range(k):
            new = torch.zeros(n, 2, device=self.device)
            todo = torch.ones(n, dtype=torch.bool, device=self.device)
            while todo.any():
                cand = lo + (hi - lo) * torch.rand(n, 2, generator=self.gen, device=self.device)
                ok = (torch.linalg.norm(spots - cand[:, None], dim=-1) > min_gap).all(-1)
                new = torch.where((todo & ok)[:, None], cand, new)
                todo = todo & ~ok
            spots = torch.cat([spots, new[:, None]], 1)
        return spots[:, taken.shape[1]:]

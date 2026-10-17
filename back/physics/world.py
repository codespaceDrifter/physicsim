# Batched rigid body world: E environments that share one scene structure, stepped in parallel.
#
# step() follows Box2D v3's "soft step" solver (Catto, Solver2D 2024):
#   collide once per step, then for each substep:
#     integrate velocities -> warm start -> solve (with position bias) -> integrate positions -> relax (no bias)
#   then restitution, and keep the accumulated impulses to warm start the next step.
import math
import torch
from physics.math3d import quat_mul, quat_from_rotvec, quat_to_matrix
from physics.joints import Joints
from physics.contacts import Contacts

# shape type codes, ordered so that contact pairs are always (lower code, higher code)
PLANE, SPHERE, BOX, CYLINDER = 0, 1, 2, 3
SHAPE_CODES = {'plane': PLANE, 'sphere': SPHERE, 'box': BOX, 'cylinder': CYLINDER}


def soft_params(hertz, damping_ratio, h):
    # soft constraint coefficients (Box2D b2MakeSoft): spring of `hertz` with `damping_ratio`, stepped implicitly with dt h.
    # bias_rate scales position error into a target velocity, mass_scale / impulse_scale soften the impulse.
    omega = 2.0 * math.pi * hertz
    a1 = 2.0 * damping_ratio + h * omega
    a2 = h * omega * a1
    a3 = 1.0 / (1.0 + a2)
    return omega / a1, a2 * a3, a3


class World:
    def __init__(self, scene: dict, num_envs: int, device):
        self.scene = scene
        self.E = num_envs
        self.device = device
        settings = scene['settings']
        self.dt = settings['dt']
        self.substeps = settings['substeps']
        self.h = self.dt / self.substeps
        # (3,) uniform gravity acceleration
        self.gravity = torch.tensor(settings['gravity'], device=device, dtype=torch.float32)
        # n-body gravitational constant, 0 = off
        self.G = settings['G']
        self.time = 0.0

        bodies = scene['bodies']
        self.N = len(bodies)
        # ["ground", "tofubot/chassis", ...] - scene body order, also the order of every (N, ...) tensor
        self.body_ids = [b['id'] for b in bodies]
        assert len(set(self.body_ids)) == self.N, 'duplicate body ids'
        # {"ground": 0, ...}
        self.body_index = {bid: i for i, bid in enumerate(self.body_ids)}

        # (N,) shape type code per body
        self.shape_type = torch.tensor([SHAPE_CODES[b['shape']['type']] for b in bodies], device=device)
        # (N, 3) box half extents (zeros for other shapes)
        self.half = torch.tensor([[s / 2 for s in b['shape']['size']] if b['shape']['type'] == 'box' else [0., 0., 0.]
                                  for b in bodies], device=device)
        # (N,) sphere / cylinder radius (0 for other shapes)
        self.radius = torch.tensor([b['shape'].get('radius', 0.0) for b in bodies], device=device)
        # (N,) cylinder half height (0 for other shapes)
        self.half_height = torch.tensor([b['shape']['height'] / 2 if b['shape']['type'] == 'cylinder' else 0.0
                                         for b in bodies], device=device)
        # [bool] python copy for build-time logic, (N,) tensor for the solver
        self.scene_static = [b['static'] for b in bodies]
        self.static = torch.tensor(self.scene_static, device=device)
        for b in bodies:
            assert b['shape']['type'] != 'plane' or b['static'], f"plane {b['id']} must be static"

        # mass properties. inertia is diagonal in the body frame for all our primitives.
        masses = []  # [float]
        inertias = []  # [[Ixx, Iyy, Izz]]
        for b in bodies:
            s = b['shape']
            if s['type'] == 'box':
                x, y, z = s['size']
                m = x * y * z * b['density']
                I = [m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12]
            elif s['type'] == 'sphere':
                m = 4 / 3 * math.pi * s['radius'] ** 3 * b['density']
                I = [0.4 * m * s['radius'] ** 2] * 3
            elif s['type'] == 'cylinder':
                r, hgt = s['radius'], s['height']
                m = math.pi * r * r * hgt * b['density']
                I = [m * (3 * r * r + hgt * hgt) / 12, 0.5 * m * r * r, m * (3 * r * r + hgt * hgt) / 12]
            else:
                m, I = 0.0, [0.0, 0.0, 0.0]
            masses.append(m)
            inertias.append(I)
        # (N,) mass (0 for static, which is never used: static bodies use inv_mass 0)
        self.mass = torch.tensor(masses, device=device)
        # (N,) 1/m, 0 for static bodies
        self.inv_mass = torch.where(self.static, 0.0, 1.0 / self.mass.clamp_min(1e-12))
        # (N, 3) body-frame principal inertia, and its inverse (0 for static)
        self.inertia = torch.tensor(inertias, device=device)
        self.inv_inertia = torch.where(self.static.unsqueeze(-1), 0.0, 1.0 / self.inertia.clamp_min(1e-12))

        # initial state from the scene, (N, ...) -> broadcast to every env on reset
        self.init_pos = torch.tensor([b['position'] for b in bodies], device=device, dtype=torch.float32)
        self.init_quat = torch.tensor([b['rotation'] for b in bodies], device=device, dtype=torch.float32)
        self.init_quat = self.init_quat / torch.linalg.norm(self.init_quat, dim=-1, keepdim=True)
        # (N, 6) [linear velocity, angular velocity]
        self.init_velocity = torch.tensor([b['velocity'] + b['angular_velocity'] for b in bodies], device=device, dtype=torch.float32)
        # (E, N, 3) position, (E, N, 4) orientation, (E, N, 6) velocity [v, ω] - one 6-vector per body so a constraint
        # reads both bodies with one gather and writes them with one index_add
        self.pos = self.init_pos.repeat(self.E, 1, 1)
        self.quat = self.init_quat.repeat(self.E, 1, 1)
        self.velocity = self.init_velocity.repeat(self.E, 1, 1)
        # (E, 1) env index for per-env gathers
        self.env_index = torch.arange(self.E, device=device).unsqueeze(-1)
        # (E, N, 3, 3) world-space inverse inertia, refreshed whenever orientations change
        self.inv_inertia_world = self.compute_inv_inertia_world()

        self.joints = Joints(self, scene['joints'])
        self.contacts = Contacts(self, bodies, self.joints.connected_pairs)

        # contact spring: Box2D v3 uses min(30 Hz, substep rate / 4). soft contact stiffness is relative to the effective
        # mass, so a light block squeezed by a gripper sinks in by about F / (m_eff ω²): 1.5 cm at 30 Hz. 120 Hz
        # keeps that to millimeters. above substep rate / 4 contacts jitter, so 120 Hz needs 8 substeps at 60 Hz.
        self.contact_soft = soft_params(min(120.0, 0.25 / self.h), 10.0, self.h)
        # joints: Box2D v3 default 60 Hz, damping ratio 2
        self.joint_soft = soft_params(60.0, 2.0, self.h)
        # joint passes per solve/relax. the joint solve is one exact global solve, so one pass is enough
        self.joint_iterations = 1
        # jacobi contact passes per solve/relax. mass splitting converges slower than gauss-seidel, so it gets a few.
        self.contact_iterations = 2

    def compute_inv_inertia_world(self):
        # I⁻¹_world = R diag(I⁻¹_body) Rᵀ. (E, N, 3, 3)
        R = quat_to_matrix(self.quat)
        return ((R * self.inv_inertia.unsqueeze(-2)).unsqueeze(-1) * R.transpose(-1, -2).unsqueeze(-3)).sum(-2)

    def inv_mass_jt(self, idx, J):
        # W = M⁻¹ Jᵀ for constraint rows J (E, K, R, 6) acting on bodies idx (K,) -> (E, K, 6, R).
        # a row impulse λ (E, K, R) changes the body velocity by W λ.
        # idx may be per env (E, K) or shared (K,): the (E, 1) env index broadcasts against both
        lin = self.inv_mass[idx][..., None, None] * J[..., :3].transpose(-1, -2)
        ang = (self.inv_inertia_world[self.env_index, idx].unsqueeze(-1) * J[..., 3:].transpose(-1, -2).unsqueeze(-3)).sum(-2)
        return torch.cat([lin, ang], dim=-2)

    def integrate_velocities(self, h):
        dynamic = ~self.static
        # (E, N, 3) acceleration: uniform gravity + optional n-body gravity
        acc = self.gravity.expand(self.E, self.N, 3).clone()
        if self.G != 0.0:
            # (E, N, N, 3) delta[i, j] = pos[j] - pos[i]
            delta = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)
            dist_sq = (delta * delta).sum(-1) + torch.eye(self.N, device=self.device)
            # (E, N, N) G m_j / r³, zero on the diagonal and for static bodies
            coef = self.G * (self.mass * dynamic) / dist_sq.pow(1.5) * (1 - torch.eye(self.N, device=self.device))
            acc = acc + (coef.unsqueeze(-1) * delta).sum(2)
        vel = self.velocity[..., :3] + acc * h * dynamic.unsqueeze(-1)

        # gyroscopic term ω × Iω, integrated implicitly with one Newton step in the body frame
        # (Bullet btRigidBody::computeGyroscopicImpulseImplicit_Body). explicit integration blows up for spinning bodies.
        R = quat_to_matrix(self.quat)
        # (E, N, 3) ω in body frame
        w = (R * self.velocity[..., 3:].unsqueeze(-1)).sum(-2)
        I = torch.where(dynamic.unsqueeze(-1), self.inertia, 1.0)
        Iw = I * w
        # residual of I(ω' - ω) + h ω' × Iω' = 0 evaluated at ω' = ω
        f = h * torch.cross(w, Iw, dim=-1)
        # (E, N, 3, 3) jacobian I + h (skew(ω) I - skew(Iω))
        zero = torch.zeros_like(w[..., 0])

        def skew(v):
            # (E, N, 3) -> (E, N, 3, 3) cross product matrix
            return torch.stack([zero, -v[..., 2], v[..., 1], v[..., 2], zero, -v[..., 0], -v[..., 1], v[..., 0], zero], -1).reshape(v.shape + (3,))

        jac = torch.diag_embed(I) + h * (skew(w) @ torch.diag_embed(I) - skew(Iw))
        w = w - torch.linalg.solve_ex(jac, f.unsqueeze(-1))[0].squeeze(-1)
        angvel = torch.where(dynamic.unsqueeze(-1), (R * w.unsqueeze(-2)).sum(-1), 0.0)
        self.velocity = torch.cat([vel, angvel], dim=-1)

    def integrate_positions(self, h):
        self.pos = self.pos + self.velocity[..., :3] * h
        # exact exponential map: q' = exp(h ω) ⊗ q (world-frame ω), renormalized against float drift
        q = quat_mul(quat_from_rotvec(self.velocity[..., 3:] * h), self.quat)
        self.quat = q / torch.linalg.norm(q, dim=-1, keepdim=True)
        self.inv_inertia_world = self.compute_inv_inertia_world()

    def enable_cuda_graph(self):
        # record one step() as a CUDA graph and replay it from then on: one launch instead of thousands of tiny
        # kernels. tensors that persist between steps (or are read outside) live at fixed addresses (self.graph_state).
        assert self.device != 'cpu' and str(self.device).startswith('cuda'), 'CUDA graphs need a cuda device'
        # [(object, attribute)] state carried between steps or read by callers
        self.graph_slots = [(self, n) for n in ('pos', 'quat', 'velocity', 'inv_inertia_world')] + \
                           [(self.joints, n) for n in ('lam', 'lam_lower', 'lam_upper', 'lam_drive', 'value')] + \
                           [(self.contacts, n) for n in ('slot', 'lam', 'valid', 'anchor_a', 'anchor_b', 'sliding')]
        self.graph = None
        # warm up on a side stream (required before capture), then capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(stream)
        # {(id(object), attribute): tensor} fixed-address copies of the state
        self.graph_state = {(id(o), n): getattr(o, n).clone() for o, n in self.graph_slots}
        for o, n in self.graph_slots:
            setattr(o, n, self.graph_state[(id(o), n)])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.step_eager()
            for o, n in self.graph_slots:
                self.graph_state[(id(o), n)].copy_(getattr(o, n))
                setattr(o, n, self.graph_state[(id(o), n)])
        self.graph = graph

    def step(self):
        if getattr(self, 'graph', None) is None:
            self.step_eager()
            self.time += self.dt
            return
        self.sync_graph_state()
        self.graph.replay()
        self.time += self.dt

    def sync_graph_state(self):
        # eager code (resets, edits) may have replaced state tensors: copy them into the captured buffers, so CUDA
        # graphs (the step, camera renders) read the current state
        if getattr(self, 'graph', None) is None:
            return
        for o, n in self.graph_slots:
            fixed = self.graph_state[(id(o), n)]
            if getattr(o, n) is not fixed:
                fixed.copy_(getattr(o, n))
                setattr(o, n, fixed)

    def step_eager(self):
        self.joints.prepare()
        self.contacts.detect()
        for _ in range(self.substeps):
            self.integrate_velocities(self.h)
            self.joints.warm_start()
            self.contacts.warm_start()
            for _ in range(self.joint_iterations):
                self.joints.solve(use_bias=True)
            for _ in range(self.contact_iterations):
                self.contacts.solve(use_bias=True)
            self.integrate_positions(self.h)
            # positions moved: refresh the joint errors for the relax pass and the next substep
            self.joints.prepare(errors_only=True)
            self.contacts.update_separation()
            for _ in range(self.joint_iterations):
                self.joints.solve(use_bias=False)
            for _ in range(self.contact_iterations):
                self.contacts.solve(use_bias=False)
        # restitution is a Jacobi pass too: iterate so mass splitting converges to the full bounce.
        # skipped when nothing in the scene can bounce
        if self.contacts.bouncy:
            for _ in range(2 * self.contact_iterations):
                self.contacts.apply_restitution()

    def reset(self, env_mask):
        # reset envs where env_mask (E,) is True to the scene's initial state
        m = env_mask.view(-1, 1, 1)
        self.pos = torch.where(m, self.init_pos, self.pos)
        self.quat = torch.where(m, self.init_quat, self.quat)
        self.velocity = torch.where(m, self.init_velocity, self.velocity)
        self.inv_inertia_world = self.compute_inv_inertia_world()
        self.joints.reset(env_mask)
        self.contacts.reset(env_mask)

    def poses(self):
        # (E, N, 7) [px, py, pz, qw, qx, qy, qz], the recording / websocket pose format
        return torch.cat([self.pos, self.quat], dim=-1)

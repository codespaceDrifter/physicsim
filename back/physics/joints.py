# Joint constraints: revolute, prismatic, fixed. batched over envs, solved Gauss-Seidel by graph color.
#
# every joint has a frame on body A and on body B (anchor point + orientation whose local +x is the joint axis).
# it has 6 constraint rows, all expressed along body A's joint-frame axes u_x, u_y, u_z:
#   rows 0..2 linear  : (pB - pA) · u_i       anchors stay together
#   rows 3..5 angular : rotation of B's frame relative to A's frame about u_i
# each joint type locks some rows and leaves one free row that the drive and limits act on:
#   revolute  locks [x y z | . y z], free = row 3 (rotation about the axis)
#   prismatic locks [. y z | x y z], free = row 0 (sliding along the axis)
#   fixed     locks everything
# joints of the same color share no dynamic body, so a whole color is solved in parallel with exact Gauss-Seidel
# between colors (the "graph coloring" PhysX/Box2D use on GPUs/SIMD). joints are stored sorted by color so a
# color is a contiguous slice.
import math
import torch
from physics.math3d import quat_mul, quat_conj, quat_rotate, quat_rotate_inv, quat_to_matrix, quat_from_two_axes, wrap_angle, const

LOCKS = {'revolute': [1, 1, 1, 0, 1, 1], 'prismatic': [0, 1, 1, 1, 1, 1], 'fixed': [1, 1, 1, 1, 1, 1]}
FREE_ROW = {'revolute': 3, 'prismatic': 0, 'fixed': -1}


class Joints:
    def __init__(self, world, defs):
        self.world = world
        dev = world.device
        self.J = len(defs)
        for d in defs:
            assert d['type'] in LOCKS, f"unknown joint type {d['type']}"
            assert d['body_a'] in world.body_index and d['body_b'] in world.body_index, f"joint {d['id']} references a missing body"
            assert d['body_a'] != d['body_b'], f"joint {d['id']} connects a body to itself"
            assert d['type'] != 'fixed' or (d['drive'] is None and d['limits'] is None), f"fixed joint {d['id']} can't have a drive or limits"
            if d['limits'] is not None and d['type'] == 'revolute':
                # the angle wraps at ±π
                assert -math.pi < d['limits'][0] <= d['limits'][1] < math.pi, f"revolute limits of {d['id']} must be inside (-pi, pi)"
            if d['drive'] is not None:
                assert d['drive']['mode'] in ('position', 'velocity'), f"unknown drive mode in {d['id']}"
                assert d['drive']['damping'] > 0 or d['drive']['stiffness'] > 0, f"drive of {d['id']} needs stiffness or damping"
                assert d['drive']['mode'] == 'position' or d['drive']['stiffness'] == 0, f"velocity drive of {d['id']} must have stiffness 0"

        # greedy graph coloring: a joint gets the lowest color not used by a joint sharing one of its dynamic bodies
        color_bodies = []  # [{body index}] dynamic bodies already used by each color
        colors = []  # [[joint def]]
        for d in defs:
            dyn = {world.body_index[d[k]] for k in ('body_a', 'body_b') if not world.scene_static[world.body_index[d[k]]]}
            c = next((c for c, used in enumerate(color_bodies) if not (used & dyn)), len(colors))
            if c == len(colors):
                color_bodies.append(set())
                colors.append([])
            color_bodies[c] |= dyn
            colors[c].append(d)
        # [(start, end)] slice of each color in the color-sorted joint order
        self.color_slices = []
        start = 0
        for c in colors:
            self.color_slices.append((start, start + len(c)))
            start += len(c)
        defs = [d for c in colors for d in c]

        # ["tofubot/shoulder", ...] - internal (color-sorted) joint order, the order of every (.., J) tensor
        self.ids = [d['id'] for d in defs]
        assert len(set(self.ids)) == self.J, 'duplicate joint ids'
        # {"tofubot/shoulder": 0, ...}
        self.index = {jid: i for i, jid in enumerate(self.ids)}
        # (J,) body indices
        self.body_a = torch.tensor([world.body_index[d['body_a']] for d in defs], device=dev, dtype=torch.long)
        self.body_b = torch.tensor([world.body_index[d['body_b']] for d in defs], device=dev, dtype=torch.long)
        # {(a, b)} body index pairs (a < b) joined by a joint, they never collide with each other
        self.connected_pairs = {tuple(sorted((world.body_index[d['body_a']], world.body_index[d['body_b']]))) for d in defs}

        # joint frames from the world anchor/axis at scene definition time -> local frames on A and B
        anchor = torch.tensor([d['anchor'] for d in defs], device=dev, dtype=torch.float32).view(-1, 3)
        axis = torch.tensor([d.get('axis', [1, 0, 0]) for d in defs], device=dev, dtype=torch.float32).view(-1, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        # (J, 4) world orientation of the joint frame (local +x = axis)
        q_joint = quat_from_two_axes(axis)
        pa, qa = world.init_pos[self.body_a], world.init_quat[self.body_a]
        pb, qb = world.init_pos[self.body_b], world.init_quat[self.body_b]
        # (J, 3) anchor in each body's frame, (J, 4) joint frame orientation in each body's frame
        self.local_ra = quat_rotate_inv(qa, anchor - pa)
        self.local_rb = quat_rotate_inv(qb, anchor - pb)
        self.local_qa = quat_mul(quat_conj(qa), q_joint)
        self.local_qb = quat_mul(quat_conj(qb), q_joint)

        # (J, 6) which rows are rigidly locked
        self.lock = torch.tensor([LOCKS[d['type']] for d in defs], device=dev, dtype=torch.float32).view(-1, 6)
        self.is_revolute = torch.tensor([d['type'] == 'revolute' for d in defs], device=dev, dtype=torch.bool)
        self.is_prismatic = torch.tensor([d['type'] == 'prismatic' for d in defs], device=dev, dtype=torch.bool)
        # (J, 6) one-hot of the free row (all zero for fixed joints)
        self.free = torch.tensor([[float(i == FREE_ROW[d['type']]) for i in range(6)] for d in defs], device=dev).view(-1, 6)

        # limits [lower, upper]
        self.has_limit = torch.tensor([d['limits'] is not None for d in defs], device=dev, dtype=torch.bool)
        self.lower = torch.tensor([d['limits'][0] if d['limits'] is not None else 0.0 for d in defs], device=dev)
        self.upper = torch.tensor([d['limits'][1] if d['limits'] is not None else 0.0 for d in defs], device=dev)

        # drives: implicit spring-damper toward a target position (stiffness > 0) or target velocity (stiffness = 0)
        self.has_drive = torch.tensor([d['drive'] is not None for d in defs], device=dev, dtype=torch.bool)
        self.drive_is_position = torch.tensor([d['drive'] is not None and d['drive']['mode'] == 'position' for d in defs], device=dev, dtype=torch.bool)
        self.stiffness = torch.tensor([d['drive']['stiffness'] if d['drive'] else 0.0 for d in defs], device=dev)
        self.damping = torch.tensor([d['drive']['damping'] if d['drive'] else 1.0 for d in defs], device=dev)
        self.max_force = torch.tensor([d['drive']['max_force'] if d['drive'] else 0.0 for d in defs], device=dev)
        # (J,) initial targets, (E, J) live targets written in place by controllers
        self.init_target = torch.tensor([d['drive']['target'] if d['drive'] else 0.0 for d in defs], device=dev)
        self.target = self.init_target.repeat(world.E, 1)

        # accumulated impulses for warm starting
        # (E, J, 6) locked rows, (E, J) lower limit, upper limit, drive
        self.lam = torch.zeros(world.E, self.J, 6, device=dev)
        self.lam_lower = torch.zeros(world.E, self.J, device=dev)
        self.lam_upper = torch.zeros(world.E, self.J, device=dev)
        self.lam_drive = torch.zeros(world.E, self.J, device=dev)

    def prepare(self):
        # everything that depends only on positions, recomputed after every position update:
        #   J_a, J_b (E, J, 6, 6) jacobian rows per side, W_a, W_b = M⁻¹ Jᵀ
        #   C (E, J, 6) position error per row, value (E, J) joint angle / slide
        #   K (E, J, 6, 6) = J M⁻¹ Jᵀ, K_inv inverse of the locked block, proj / k_free for the free row
        if self.J == 0:
            return
        w = self.world
        a, b = self.body_a, self.body_b
        qa, qb = w.quat[:, a], w.quat[:, b]
        # (E, J, 3) lever arms from body centers to the anchors
        ra = quat_rotate(qa, self.local_ra)
        rb = quat_rotate(qb, self.local_rb)
        # (E, J, 4) joint frames in world
        qfa = quat_mul(qa, self.local_qa)
        qfb = quat_mul(qb, self.local_qb)
        # (E, J, 3, 3) rows = A's joint frame axes u_x, u_y, u_z in world
        axes = quat_to_matrix(qfa).transpose(-1, -2)
        c_lin = (axes * ((w.pos[:, b] + rb) - (w.pos[:, a] + ra)).unsqueeze(-2)).sum(-1)
        # (E, J, 4) rotation of B's frame relative to A's frame, short way round (w >= 0)
        q_rel = quat_mul(quat_conj(qfa), qfb)
        q_rel = q_rel * torch.where(q_rel[..., :1] < 0, -1.0, 1.0)
        # swing-twist split about the joint axis x: q_rel = swing ⊗ twist
        twist = q_rel * const([1.0, 1.0, 0.0, 0.0], q_rel.device)
        twist = twist / torch.linalg.norm(twist, dim=-1, keepdim=True).clamp_min(1e-9)
        swing = quat_mul(q_rel, quat_conj(twist))
        # (E, J, 3) small-angle rotation error: revolute ignores the twist, others lock all 3
        c_ang = 2.0 * torch.where(self.is_revolute.view(1, -1, 1), swing[..., 1:], q_rel[..., 1:])
        angle = 2.0 * torch.atan2(q_rel[..., 1], q_rel[..., 0])
        self.value = torch.where(self.is_revolute, angle, torch.where(self.is_prismatic, c_lin[..., 0], 0.0))
        self.C = torch.cat([c_lin, c_ang], dim=-1)
        zeros = torch.zeros_like(axes)
        # (E, J, 6, 6) jacobian rows of each side: row velocity = J_a V_a + J_b V_b, V = [v, ω]
        self.J_a = torch.cat([torch.cat([-axes, -torch.cross(ra.unsqueeze(-2).expand_as(axes), axes, dim=-1)], dim=-1),
                              torch.cat([zeros, -axes], dim=-1)], dim=-2)
        self.J_b = torch.cat([torch.cat([axes, torch.cross(rb.unsqueeze(-2).expand_as(axes), axes, dim=-1)], dim=-1),
                              torch.cat([zeros, axes], dim=-1)], dim=-2)
        # (E, J, 6, 6) velocity change of each body per unit row impulse
        self.W_a = w.inv_mass_jt(a, self.J_a)
        self.W_b = w.inv_mass_jt(b, self.J_b)
        # (E, J, 6, 6) effective inverse mass K = J M⁻¹ Jᵀ
        self.K = (self.J_a.unsqueeze(-1) * self.W_a.unsqueeze(-3)).sum(-2) + (self.J_b.unsqueeze(-1) * self.W_b.unsqueeze(-3)).sum(-2)
        lock = self.lock
        # inverse of the locked block, unlocked rows masked to identity so it stays invertible
        self.K_inv = torch.linalg.inv_ex(self.K * (lock.unsqueeze(-1) * lock.unsqueeze(-2)) + torch.diag_embed(1.0 - lock))[0] * (lock.unsqueeze(-1) * lock.unsqueeze(-2))
        # drive and limit impulses act along the free row, corrected so they don't pull on the locked rows:
        # row-impulse direction = e_free - K_LL⁻¹ K_L,free (Schur complement). without this, a strong drive
        # pushing into a limit gets undone every pass by the anchor rows.
        free = self.free.expand_as(self.K[..., 0])
        self.proj = free - (self.K_inv * (self.K * free.unsqueeze(-2)).sum(-1).unsqueeze(-2)).sum(-1)
        # (E, J, 6) change of every row velocity per unit impulse along proj
        self.K_proj = (self.K * self.proj.unsqueeze(-2)).sum(-1)
        # (E, J) inverse effective mass along proj
        self.k_free = (self.proj * self.K_proj).sum(-1).clamp_min(1e-12)

    def row_velocity(self, sl):
        # (E, Jc, 6) velocity of every row of joints in slice sl
        V = self.world.velocity
        # small matrix-vector products as multiply + sum: batched cuBLAS gemv is very slow on tiny matrices
        return (self.J_a[:, sl] * V[:, self.body_a[sl]].unsqueeze(-2)).sum(-1) + (self.J_b[:, sl] * V[:, self.body_b[sl]].unsqueeze(-2)).sum(-1)

    def speeds(self):
        # (E, J) velocity of every joint along its free row (rad/s or m/s), 0 for fixed joints
        return (self.row_velocity(slice(None)) * self.free).sum(-1)

    def apply(self, sl, mu):
        # apply row impulses mu (E, Jc, 6) of joints in slice sl
        V = self.world.velocity
        V.index_add_(1, self.body_a[sl], (self.W_a[:, sl] * mu.unsqueeze(-2)).sum(-1))
        V.index_add_(1, self.body_b[sl], (self.W_b[:, sl] * mu.unsqueeze(-2)).sum(-1))

    def warm_start(self):
        if self.J == 0:
            return
        # limits act along the projected free direction: lower pushes +, upper pushes -
        self.apply(slice(None), self.lam + (self.lam_lower - self.lam_upper).unsqueeze(-1) * self.proj)
        # drive impulses restart from zero every substep (their target moves with the position)
        self.lam_drive.zero_()

    def solve(self, use_bias):
        w = self.world
        h = w.h
        bias_rate, mass_scale, impulse_scale = w.joint_soft
        for start, end in self.color_slices:
            sl = slice(start, end)
            # (E, Jc, 6) row velocities. joints in a color share no dynamic body, so an impulse mu on a joint changes
            # only its own row velocities, by K mu. track that here and touch the bodies once at the end.
            rv = self.row_velocity(sl)
            proj, K_proj, k_free, value = self.proj[:, sl], self.K_proj[:, sl], self.k_free[:, sl], self.value[:, sl]

            # 1. drive: implicit spring-damper  F = k (x_target - x) + c (v_target - v)  (PhysX drive model).
            # solving for the impulse at the end-of-substep velocity gives the soft constraint
            #   λ = -(v - v_eff + γ λ_acc) / (K + γ),  γ = 1 / (h (c + h k)),  v_eff = (k (x_t - x) + c v_t) / (c + h k)
            k, c = self.stiffness[sl], self.damping[sl]
            target = self.target[:, sl]
            pos_err = torch.where(self.is_revolute[sl], wrap_angle(target - value), target - value)
            x_err = torch.where(self.drive_is_position[sl], pos_err, 0.0)
            v_target = torch.where(self.drive_is_position[sl], 0.0, target)
            gamma = 1.0 / (h * (c + h * k))
            v_eff = (k * x_err + c * v_target) / (c + h * k)
            old = self.lam_drive[:, sl]
            max_imp = self.max_force[sl] * h
            new = old - ((rv * proj).sum(-1) - v_eff + gamma * old) / (k_free + gamma)
            new = torch.where(self.has_drive[sl], torch.maximum(torch.minimum(new, max_imp), -max_imp), 0.0)
            d = new - old
            self.lam_drive[:, sl] = new
            mu = d.unsqueeze(-1) * proj
            rv = rv + d.unsqueeze(-1) * K_proj

            # 2. limits: one-sided rows on the free axis. lower: x - lo >= 0 pushes +, upper: hi - x >= 0 pushes -
            for sign, gap, lam_buf in ((1.0, value - self.lower[sl], self.lam_lower), (-1.0, self.upper[sl] - value, self.lam_upper)):
                # speculative when not yet touching (only stop the approach), soft push-out when violated
                spec = gap > 0
                bias = torch.where(spec, gap / h, bias_rate * gap if use_bias else 0.0)
                ms = torch.where(spec, 1.0, mass_scale if use_bias else 1.0)
                imp_s = torch.where(spec, 0.0, impulse_scale if use_bias else 0.0)
                old = lam_buf[:, sl]
                new = torch.clamp_min(old - ms * (sign * (rv * proj).sum(-1) + bias) / k_free - imp_s * old, 0.0)
                new = torch.where(self.has_limit[sl], new, 0.0)
                d = sign * (new - old)
                lam_buf[:, sl] = new
                mu = mu + d.unsqueeze(-1) * proj
                rv = rv + d.unsqueeze(-1) * K_proj

            # 3. locked rows as one 6x6 block: Δλ = -K_LL⁻¹ (J v + bias)
            rhs = rv + (bias_rate * self.C[:, sl] if use_bias else 0.0)
            ms, imp_s = (mass_scale, impulse_scale) if use_bias else (1.0, 0.0)
            old = self.lam[:, sl]
            d_lam = (-ms * (self.K_inv[:, sl] * rhs.unsqueeze(-2)).sum(-1) - imp_s * old) * self.lock[sl]
            self.lam[:, sl] = old + d_lam
            self.apply(sl, mu + d_lam)

    def reset(self, env_mask):
        # in place, so tensors captured by a CUDA graph keep their addresses
        m = env_mask.view(-1, 1)
        self.lam.masked_fill_(m.unsqueeze(-1), 0.0)
        self.lam_lower.masked_fill_(m, 0.0)
        self.lam_upper.masked_fill_(m, 0.0)
        self.lam_drive.masked_fill_(m, 0.0)
        self.target.copy_(torch.where(m, self.init_target, self.target))

    def set_targets(self, targets: dict):
        # {joint_id: value} -> same target in every env
        for jid, v in targets.items():
            assert jid in self.index, f'unknown joint {jid}'
            self.target[:, self.index[jid]] = v

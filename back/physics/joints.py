# Joint constraints: revolute, prismatic, fixed, batched over envs.
#
# every joint has a frame on body A and on body B (anchor point + orientation whose local +x is the joint axis).
# it has 6 constraint rows, all expressed along body A's joint-frame axes u_x, u_y, u_z:
#   rows 0..2 linear  : (pB - pA) · u_i       anchors stay together
#   rows 3..5 angular : rotation of B's frame relative to A's frame about u_i
# each joint type locks some rows and leaves one free row that the drive and limits act on:
#   revolute  locks [x y z | . y z], free = row 3 (rotation about the axis)
#   prismatic locks [. y z | x y z], free = row 0 (sliding along the axis)
#   fixed     locks everything
# all locked rows and all motor rows of a mechanism are solved together in one exact linear solve (Baraff 1996),
# so it behaves as one rigid whole: a light link holding up a heavy arm doesn't bend. solving joints one at a time
# (Gauss-Seidel), each joint only feels its two neighbouring bodies and soft joints sag under load.
# the system is block diagonal over articulations (bodies connected through joints; static bodies don't connect):
# each articulation is solved as its own dense system, batched over envs x articulations (padded to the largest).
import math
import torch
from physics.math3d import quat_mul, quat_conj, quat_rotate, quat_rotate_inv, quat_to_matrix, quat_from_two_axes, wrap_angle, const, spd_inverse

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

        # ["lekiwi/shoulder_lift", ...] - scene joint order, the order of every (.., J) tensor
        self.ids = [d['id'] for d in defs]
        assert len(set(self.ids)) == self.J, 'duplicate joint ids'
        # {"lekiwi/shoulder_lift": 0, ...}
        self.index = {jid: i for i, jid in enumerate(self.ids)}
        # (J,) body indices
        self.body_a = torch.tensor([world.body_index[d['body_a']] for d in defs], device=dev, dtype=torch.long)
        self.body_b = torch.tensor([world.body_index[d['body_b']] for d in defs], device=dev, dtype=torch.long)
        # {(a, b)} body index pairs (a < b) joined by a joint, they never collide with each other
        self.connected_pairs = {tuple(sorted((world.body_index[d['body_a']], world.body_index[d['body_b']]))) for d in defs}

        # articulations: union-find over the dynamic bodies each joint connects
        static = world.scene_static
        root = list(range(world.N))  # [int] union-find parent

        def find(i):
            while root[i] != i:
                root[i] = root[root[i]]
                i = root[i]
            return i

        for d in defs:
            a, b = world.body_index[d['body_a']], world.body_index[d['body_b']]
            assert not (static[a] and static[b]), f"joint {d['id']} connects two static bodies"
            if not static[a] and not static[b]:
                root[find(a)] = find(b)
        joint_art = [find(world.body_index[d['body_b' if static[world.body_index[d['body_a']]] else 'body_a']]) for d in defs]  # [root body]
        roots = sorted(set(joint_art))  # [root body] one per articulation
        # [[joint index]], [[body index]] per articulation, in scene order
        art_joints = [[j for j in range(self.J) if joint_art[j] == r] for r in roots]
        art_bodies = [[i for i in range(world.N) if not static[i] and find(i) == r] for r in roots]
        self.A_count = len(roots)
        # largest articulation: joints, bodies (smaller ones are padded with rows / bodies that do nothing)
        self.Jm = max((len(js) for js in art_joints), default=0)
        self.Nm = max((len(bs) for bs in art_bodies), default=0)
        # (J,) each joint's articulation and its slot in it; (J,) slot of its bodies within the articulation, -1 static
        self.art = torch.tensor([roots.index(r) for r in joint_art], device=dev, dtype=torch.long)
        self.art_slot = torch.tensor([art_joints[roots.index(joint_art[j])].index(j) for j in range(self.J)], device=dev, dtype=torch.long)
        slot_of = lambda j, side: -1 if static[world.body_index[defs[j][side]]] else art_bodies[roots.index(joint_art[j])].index(world.body_index[defs[j][side]])
        self.body_slot_a = torch.tensor([slot_of(j, 'body_a') for j in range(self.J)], device=dev, dtype=torch.long)
        self.body_slot_b = torch.tensor([slot_of(j, 'body_b') for j in range(self.J)], device=dev, dtype=torch.long)
        # (C, Jm) joint index of each articulation slot and whether it's real, (C, Nm) the same for bodies
        self.art_joint = torch.tensor([js + [0] * (self.Jm - len(js)) for js in art_joints], device=dev, dtype=torch.long).view(self.A_count, self.Jm)
        self.art_joint_valid = torch.tensor([[1.0] * len(js) + [0.0] * (self.Jm - len(js)) for js in art_joints], device=dev).view(self.A_count, self.Jm)
        self.art_body = torch.tensor([bs + [0] * (self.Nm - len(bs)) for bs in art_bodies], device=dev, dtype=torch.long).view(self.A_count, self.Nm)
        self.art_body_valid = torch.tensor([[1.0] * len(bs) + [0.0] * (self.Nm - len(bs)) for bs in art_bodies], device=dev).view(self.A_count, self.Nm)
        # joints whose A / B side is a dynamic body (static sides have no column in G)
        self.dynamic_a = (self.body_slot_a >= 0).nonzero().view(-1)
        self.dynamic_b = (self.body_slot_b >= 0).nonzero().view(-1)

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
        # (J,) joint value in the scene's pose (optional "initial", default 0): lets a robot be built in a rest pose while
        # its joint zeros stay where the real hardware's calibration puts them
        self.initial = torch.tensor([float(d.get('initial', 0.0)) for d in defs], device=dev)
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
        # (J,) optional servo speed limit (rad/s or m/s): a position drive never commands more than this
        self.max_velocity = torch.tensor([d['drive'].get('max_velocity', float('inf')) if d['drive'] else float('inf') for d in defs], device=dev)
        # (J, 6) one-hot of the free row of driven joints: the motor's row in the global solve
        self.drive_row = self.free * self.has_drive.unsqueeze(-1)
        # (J,) implicit drive softness γ = 1 / (h (c + h k)) (only used on drive rows)
        self.gamma = 1.0 / (world.h * (self.damping + world.h * self.stiffness))
        # (J,) initial targets, (E, J) live targets written in place by controllers
        self.init_target = torch.tensor([d['drive']['target'] if d['drive'] else 0.0 for d in defs], device=dev)
        self.target = self.init_target.repeat(world.E, 1)

        # accumulated impulses for warm starting
        # (E, J, 6) locked rows, (E, J) lower limit, upper limit, drive
        self.lam = torch.zeros(world.E, self.J, 6, device=dev)
        self.lam_lower = torch.zeros(world.E, self.J, device=dev)
        self.lam_upper = torch.zeros(world.E, self.J, device=dev)
        self.lam_drive = torch.zeros(world.E, self.J, device=dev)

    def prepare(self, errors_only=False):
        # everything that depends only on positions. errors_only: just C and value (the substeps, after positions moved);
        # the jacobians, masses and factorized systems stay from the start of the step (Box2D v3 soft step: effective
        # masses are computed once per step, substeps only refresh the constraint errors). the full set:
        #   J_a, J_b (E, J, 6, 6) jacobian rows per side (row velocity = J_a v_a + J_b v_b), W_a, W_b (E, J, 6, 6) =
        #   M⁻¹ Jᵀ per side (velocity change per unit row impulse), K_art (E, C, 6Jm, 6Jm) = G M⁻¹ Gᵀ per articulation
        #   (G: its dense jacobian, only built here), A_inv / L_inv its factorized systems
        #   C (E, J, 6) position error per row, value (E, J) joint angle / slide
        #   K (E, J, 6, 6) each joint's own block of K_art, proj / k_free for limits on the free row
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
        self.value = torch.where(self.is_revolute, wrap_angle(angle + self.initial), torch.where(self.is_prismatic, c_lin[..., 0] + self.initial, 0.0))
        self.C = torch.cat([c_lin, c_ang], dim=-1)
        if errors_only:
            return
        zeros = torch.zeros_like(axes)
        # (E, J, 6, 6) jacobian rows of each side: row velocity = J_a V_a + J_b V_b, V = [v, ω]
        self.J_a = torch.cat([torch.cat([-axes, -torch.cross(ra.unsqueeze(-2).expand_as(axes), axes, dim=-1)], dim=-1),
                              torch.cat([zeros, -axes], dim=-1)], dim=-2)
        self.J_b = torch.cat([torch.cat([axes, torch.cross(rb.unsqueeze(-2).expand_as(axes), axes, dim=-1)], dim=-1),
                              torch.cat([zeros, axes], dim=-1)], dim=-2)
        # exact system over every joint's 6 rows (Baraff 1996), one dense block per articulation: solving all locked rows
        # at once is exact for the whole mechanism, so a light link holding up a heavy arm doesn't bend (one joint at a
        # time, a soft hinge only feels its two neighbours and sags by load / (m_eff ω²))
        E, N, C, Jm, Nm = w.E, w.N, self.A_count, self.Jm, self.Nm
        # (E, C, Jm, Nm, 6, 6) row jacobians placed at each joint's two bodies (static sides left out: they don't move)
        # -> (E, C, 6Jm, 6Nm) dense G per articulation
        G = torch.zeros(E, C, Jm, Nm, 6, 6, device=w.device)
        da, db = self.dynamic_a, self.dynamic_b
        G[:, self.art[da], self.art_slot[da], self.body_slot_a[da]] = self.J_a[:, da]
        G[:, self.art[db], self.art_slot[db], self.body_slot_b[db]] = self.J_b[:, db]
        G = G.permute(0, 1, 2, 4, 3, 5).reshape(E, C, 6 * Jm, 6 * Nm)
        # (E, N, 6, 6) block-diagonal inverse mass [m⁻¹ I, 0; 0, I⁻¹_world]
        minv = torch.zeros(E, N, 6, 6, device=w.device)
        minv[..., :3, :3] = torch.diag_embed(w.inv_mass.expand(E, N).unsqueeze(-1).expand(E, N, 3))
        minv[..., 3:, 3:] = w.inv_inertia_world
        # (E, J, 6, 6) per side M⁻¹ Jᵀ: column i = velocity change of that body per unit impulse on row i
        self.W_a = minv[:, a] @ self.J_a.transpose(-1, -2)
        self.W_b = minv[:, b] @ self.J_b.transpose(-1, -2)
        # (E, C, Nm, 6, 6) each articulation's bodies (padding: zero), (E, C, 6Nm, 6Jm) M⁻¹ Gᵀ -> (E, C, 6Jm, 6Jm) K = G M⁻¹ Gᵀ
        minv_art = minv[:, self.art_body] * self.art_body_valid[..., None, None]
        GT = G.transpose(-1, -2).reshape(E, C, Nm, 6, 6 * Jm)
        self.K_art = G @ (minv_art @ GT).reshape(E, C, 6 * Nm, 6 * Jm)
        # (E, J, 6, 6) each joint's own block of K (its diagonal block)
        self.K = self.K_art.reshape(E, C, Jm, 6, Jm, 6).diagonal(dim1=2, dim2=4).permute(0, 1, 4, 2, 3)[:, self.art, self.art_slot]
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

        # (C, 6Jm) rows in the solve (locked + drive; padding rows are neither); (E, C, 6Jm, 6Jm) inverse of
        # A = K + diag(γ on drive rows), rows outside the solve masked to identity
        valid = self.art_joint_valid[..., None]
        active = ((self.lock + self.drive_row)[self.art_joint] * valid).reshape(C, -1)
        gamma = ((self.drive_row * self.gamma.unsqueeze(-1))[self.art_joint] * valid).reshape(C, -1)
        mask = active.unsqueeze(-1) * active.unsqueeze(-2)
        self.A_inv = spd_inverse(self.K_art * mask + torch.diag_embed(gamma + 1.0 - active)) * mask
        # (E, C, 6Jm, 6Jm) inverse over the locked rows only: re-solves them after motors saturate at their max force
        locked = (self.lock[self.art_joint] * valid).reshape(C, -1)
        lmask = locked.unsqueeze(-1) * locked.unsqueeze(-2)
        self.L_inv = spd_inverse(self.K_art * lmask + torch.diag_embed(1.0 - locked)) * lmask

    def art_mul(self, M, x):
        # per-articulation matrices M (E, C, 6Jm, 6Jm) times per-joint row vectors x (E, J, 6) -> (E, J, 6)
        E = x.shape[0]
        # (E, C, 6Jm, 1) x gathered into articulation slots (padding slots read joint 0, masked to zero)
        xa = (x[:, self.art_joint] * self.art_joint_valid[..., None]).reshape(E, self.A_count, -1, 1)
        return (M @ xa).reshape(E, self.A_count, self.Jm, 6)[:, self.art, self.art_slot]

    def row_velocity(self):
        # (E, J, 6) velocity of every joint row: J_a v_a + J_b v_b (each joint touches two bodies; multiply + sum, as in
        # contacts: the dense G v would read a (6J, 6N) matrix per env)
        V = self.world.velocity
        return (self.J_a * V[:, self.body_a].unsqueeze(-2)).sum(-1) + (self.J_b * V[:, self.body_b].unsqueeze(-2)).sum(-1)

    def speeds(self):
        # (E, J) velocity of every joint along its free row (rad/s or m/s), 0 for fixed joints
        return (self.row_velocity() * self.free).sum(-1)

    def apply(self, mu):
        # apply row impulses mu (E, J, 6) to both bodies of each joint: v += M⁻¹ Jᵀ mu (accumulate: a body in several joints)
        V = self.world.velocity
        env = torch.arange(V.shape[0], device=V.device).unsqueeze(-1).expand(-1, self.J)
        V.index_put_((env, self.body_a.expand_as(env)), (self.W_a * mu.unsqueeze(-2)).sum(-1), accumulate=True)
        V.index_put_((env, self.body_b.expand_as(env)), (self.W_b * mu.unsqueeze(-2)).sum(-1), accumulate=True)

    def warm_start(self):
        if self.J == 0:
            return
        # limits act along the projected free direction: lower pushes +, upper pushes -
        self.apply(self.lam + (self.lam_lower - self.lam_upper).unsqueeze(-1) * self.proj)
        # drive impulses restart from zero every substep (their target moves with the position)
        self.lam_drive.zero_()

    def solve(self, use_bias):
        if self.J == 0:
            return
        w = self.world
        h = w.h
        bias_rate, mass_scale, impulse_scale = w.joint_soft
        # 1. every joint at once, one exact solve (A Δλ = -r) over:
        #    locked rows: rigid, r = G v + bias C. the solve is exact, so no soft leak is needed for stability, and a
        #      leak would leave a light link bent under a strong motor's torque
        #    drive rows (free row of driven joints): implicit spring-damper F = k (x_t - x) + c (v_t - v) (PhysX drive),
        #      r = v - v_eff + γ λ_acc with γ = 1 / (h (c + h k)) on A's diagonal, v_eff = (k (x_t - x) + c v_t) / (c + h k)
        E = w.E
        value = self.value
        pos_err = torch.where(self.is_revolute, wrap_angle(self.target - value), self.target - value)
        x_err = torch.where(self.drive_is_position, pos_err, 0.0)
        v_target = torch.where(self.drive_is_position, 0.0, self.target)
        k, c = self.stiffness, self.damping
        v_eff = (k * x_err + c * v_target) / (c + h * k)
        v_eff = torch.maximum(torch.minimum(v_eff, self.max_velocity), -self.max_velocity)
        # (E, J, 6) row velocities
        rv = self.row_velocity()
        r_locked = rv + (bias_rate * self.C if use_bias else 0.0)
        r_drive = rv - v_eff.unsqueeze(-1) + (self.gamma * self.lam_drive).unsqueeze(-1)
        r = torch.where(self.lock > 0, r_locked, torch.where(self.drive_row > 0, r_drive, 0.0))
        d_lam = -self.art_mul(self.A_inv, r)
        # motors can't exceed their max force: clamp the drive row's accumulated impulse, then re-solve the locked rows
        # for the motor impulses actually applied (K_LL Δλ_L = -(r_L + K_LD Δλ_D)), so they stay consistent
        max_imp = self.max_force * h
        d_free = (d_lam * self.free).sum(-1)
        new_drive = torch.where(self.has_drive, torch.maximum(torch.minimum(self.lam_drive + d_free, max_imp), -max_imp), 0.0)
        d_drive = (new_drive - self.lam_drive).unsqueeze(-1) * self.drive_row
        r_l = (r + self.art_mul(self.K_art, d_drive)) * self.lock
        d_lam = -self.art_mul(self.L_inv, r_l) + d_drive
        self.lam_drive = new_drive
        self.lam = self.lam + d_lam * self.lock
        self.apply(d_lam)

        # 2. limits (after the global solve, so a motor pushing into a limit can't overrule it): one-sided rows on the free axis, all joints in one batch (rarely active, and only along the free
        # direction projected off the locked rows, so they barely couple). lower: x - lo >= 0 pushes +, upper: hi - x >= 0 pushes -
        rv = self.row_velocity()
        mu = torch.zeros_like(rv)
        for sign, gap, lam_buf in ((1.0, self.value - self.lower, self.lam_lower), (-1.0, self.upper - self.value, self.lam_upper)):
            # speculative when not yet touching (only stop the approach), soft push-out when violated
            spec = gap > 0
            bias = torch.where(spec, gap / h, bias_rate * gap if use_bias else 0.0)
            ms = torch.where(spec, 1.0, mass_scale if use_bias else 1.0)
            imp_s = torch.where(spec, 0.0, impulse_scale if use_bias else 0.0)
            old = lam_buf
            new = torch.clamp_min(old - ms * (sign * (rv * self.proj).sum(-1) + bias) / self.k_free - imp_s * old, 0.0)
            new = torch.where(self.has_limit, new, 0.0)
            d = sign * (new - old)
            lam_buf.copy_(new)
            mu = mu + d.unsqueeze(-1) * self.proj
            rv = rv + d.unsqueeze(-1) * self.K_proj
        self.apply(mu)

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

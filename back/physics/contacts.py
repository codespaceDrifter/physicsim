# Contact constraints: detection (via collision.py) + non-penetration, Coulomb friction, restitution.
#
# all contact slots of all pairs in all envs are solved at once, Jacobi style, with mass splitting
# (Tonge, Benevolenski, Voroshilov 2012): a body touched by k contacts acts as k copies of mass m/k, one per
# contact, and the copies' velocities are averaged afterwards. that makes the parallel update converge like
# Gauss-Seidel without any ordering, which is how GPU solvers handle contacts.
#
# detection fills a fixed table of S "slots" per env (every candidate pair x its max contact points). only a few are
# touching at any time, so each step the MAX_CONTACTS deepest valid slots per env are copied into a fixed-capacity
# contact buffer (E, K) and only those are solved (the contact buffer of MJX / Isaac Gym). shapes stay static.
import torch
from physics.math3d import quat_rotate, quat_rotate_inv, const
from physics import collision

PLANE, SPHERE, BOX, CYLINDER = 0, 1, 2, 3
# penetration allowed before pushing out (avoids jitter from contacts flickering on/off)
LINEAR_SLOP = 0.001
# contacts are created up to this far apart (speculative contacts prevent tunneling)
SPECULATIVE_DISTANCE = 0.02
# max speed used to push overlapping bodies apart
MAX_PUSH_VELOCITY = 3.0
# closing speed below which bounces are ignored (resting contacts don't jitter)
RESTITUTION_THRESHOLD = 0.5
# contact buffer capacity per env. if more slots touch, the shallowest ones are dropped
MAX_CONTACTS = 64
# {(type_a, type_b): contact slots per pair}
SLOTS = {(PLANE, SPHERE): 1, (PLANE, BOX): 8, (PLANE, CYLINDER): 8, (SPHERE, SPHERE): 1, (SPHERE, BOX): 1,
         (SPHERE, CYLINDER): 1, (BOX, BOX): 25, (BOX, CYLINDER): 24}


class Contacts:
    def __init__(self, world, bodies, connected_pairs):
        self.world = world
        dev = world.device
        types = world.shape_type.tolist()
        static = world.static.tolist()
        # candidate pairs: everything except static-static, non-colliding bodies, same nonzero group, joined by a joint
        pairs = {}  # {(type_a, type_b): [(a, b)]}
        for i in range(world.N):
            for j in range(i + 1, world.N):
                bi, bj = bodies[i], bodies[j]
                if static[i] and static[j]:
                    continue
                if not (bi['collide'] and bj['collide']):
                    continue
                if bi['group'] != 0 and bi['group'] == bj['group']:
                    continue
                if (i, j) in connected_pairs:
                    continue
                a, b = (i, j) if types[i] <= types[j] else (j, i)
                key = (types[a], types[b])
                assert key in SLOTS, f"no collision routine for {bi['id']} vs {bj['id']} (shape types {key})"
                pairs.setdefault(key, []).append((a, b))
        # [((type_a, type_b), a_idx (P,), b_idx (P,))] one batched narrowphase call per pair type
        self.groups = [(key, torch.tensor([p[0] for p in ps], device=dev), torch.tensor([p[1] for p in ps], device=dev))
                       for key, ps in pairs.items()]
        # (S,) body index of every contact slot, in group order
        self.slot_a = torch.cat([a.repeat_interleave(SLOTS[key]) for key, a, b in self.groups]) if self.groups else torch.zeros(0, dtype=torch.long, device=dev)
        self.slot_b = torch.cat([b.repeat_interleave(SLOTS[key]) for key, a, b in self.groups]) if self.groups else torch.zeros(0, dtype=torch.long, device=dev)
        self.S = len(self.slot_a)
        fr = torch.tensor([b['friction'] for b in bodies], device=dev)
        rest = torch.tensor([b['restitution'] for b in bodies], device=dev)
        # (S,) mixing rules (Box2D): geometric mean friction, max restitution
        self.slot_friction = torch.sqrt(fr[self.slot_a] * fr[self.slot_b])
        self.slot_restitution = torch.maximum(rest[self.slot_a], rest[self.slot_b])

        E = world.E
        # contact buffer capacity
        self.C = min(self.S, MAX_CONTACTS)
        # (E, 1) env index, broadcasts against (E, C) body indices
        self.env = torch.arange(E, device=dev).unsqueeze(-1)
        # (E, C) slot held by each buffer entry, (E, C, 3) accumulated impulse [normal, t1, t2], (E, C) entry in use.
        # the impulses are kept between steps and handed to the same slot next step (warm starting)
        self.slot = torch.zeros(E, self.C, dtype=torch.long, device=dev)
        self.lam = torch.zeros(E, self.C, 3, device=dev)
        self.valid = torch.zeros(E, self.C, dtype=torch.bool, device=dev)

    def detect(self):
        if self.C == 0:
            return
        w = self.world
        points, normals, depths = [], [], []  # [(E, P*K, 3)], [(E, P*K, 3)], [(E, P*K)]
        for key, a, b in self.groups:
            pa, qa, pb, qb = w.pos[:, a], w.quat[:, a], w.pos[:, b], w.quat[:, b]
            if key == (PLANE, SPHERE):
                out = collision.plane_sphere(pa, qa, pb, w.radius[b])
            elif key == (PLANE, BOX):
                out = collision.plane_box(pa, qa, pb, qb, w.half[b])
            elif key == (PLANE, CYLINDER):
                out = collision.plane_cylinder(pa, qa, pb, qb, w.radius[b], w.half_height[b])
            elif key == (SPHERE, SPHERE):
                out = collision.sphere_sphere(pa, w.radius[a], pb, w.radius[b])
            elif key == (SPHERE, BOX):
                out = collision.sphere_box(pa, w.radius[a], pb, qb, w.half[b])
            elif key == (SPHERE, CYLINDER):
                out = collision.sphere_cylinder(pa, w.radius[a], pb, qb, w.radius[b], w.half_height[b])
            elif key == (BOX, BOX):
                out = collision.box_box(pa, qa, w.half[a], pb, qb, w.half[b])
            else:
                out = collision.box_cylinder(pa, qa, w.half[a], pb, qb, w.radius[b], w.half_height[b])
            points.append(out[0].flatten(1, 2))
            normals.append(out[1].flatten(1, 2))
            depths.append(out[2].flatten(1, 2))
        # (E, S) every slot's depth, -inf for slots out of range
        depth = torch.cat(depths, 1)
        depth = torch.where(depth > -SPECULATIVE_DISTANCE, depth, -float('inf'))
        # fill the contact buffer with the deepest slots. (E, C)
        depth, slot = depth.topk(self.C, dim=1)
        valid = depth > -float('inf')
        # warm start: hand last step's impulses to the same slot (same slot = same geometric feature), via a full slot table
        table = torch.zeros(w.E, self.S, 3, device=w.device)
        table.scatter_(1, self.slot.unsqueeze(-1).expand(-1, -1, 3), torch.where(self.valid.unsqueeze(-1), self.lam, 0.0))
        self.lam = torch.where(valid.unsqueeze(-1), table.gather(1, slot.unsqueeze(-1).expand(-1, -1, 3)), 0.0)
        self.slot = slot
        self.valid = valid
        # (E, C) bodies, friction, restitution of each buffer entry
        self.idx_a = self.slot_a[slot]
        self.idx_b = self.slot_b[slot]
        self.friction = self.slot_friction[slot]
        self.restitution = self.slot_restitution[slot]
        point = torch.cat(points, 1).gather(1, slot.unsqueeze(-1).expand(-1, -1, 3))
        # (E, C, 3) A -> B
        self.normal = torch.cat(normals, 1).gather(1, slot.unsqueeze(-1).expand(-1, -1, 3))
        # separation at detection time; during substeps separation = (anchor_B - anchor_A) · n + s0
        self.s0 = torch.where(valid, -depth, 0.0)
        pos_a, pos_b = w.pos[self.env, self.idx_a], w.pos[self.env, self.idx_b]
        quat_a, quat_b = w.quat[self.env, self.idx_a], w.quat[self.env, self.idx_b]
        # (E, C, 3) lever arms from body centers to the contact point, fixed in world orientation for the whole step
        # (Box2D v3: rotating them with the body would drag a rolling ball's contact point up its side)
        self.ra = torch.where(valid.unsqueeze(-1), point - pos_a, 0.0)
        self.rb = torch.where(valid.unsqueeze(-1), point - pos_b, 0.0)
        # (E, C, 3) the same anchors in body frames, only used to measure separation as bodies move
        self.local_ra = quat_rotate_inv(quat_a, self.ra)
        self.local_rb = quat_rotate_inv(quat_b, self.rb)
        # (E, C, 3, 3) contact basis rows: normal, then friction directions t1, t2 orthogonal to it
        n = self.normal
        ref = torch.where((n[..., :1].abs() < 0.9), const([1., 0., 0.], n.device), const([0., 1., 0.], n.device))
        t1 = torch.cross(n, ref, dim=-1)
        t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True)
        self.basis = torch.stack([n, t1, torch.cross(n, t1, dim=-1)], dim=-2)
        # mass splitting counts: (E, N) number of valid contacts touching each body
        count = torch.zeros(w.E, w.N, device=w.device)
        count.scatter_add_(1, self.idx_a, valid.float())
        count.scatter_add_(1, self.idx_b, valid.float())
        # (E, C, 1) split factor of each side (>= 1)
        self.k_a = count.gather(1, self.idx_a).clamp_min(1.0).unsqueeze(-1)
        self.k_b = count.gather(1, self.idx_b).clamp_min(1.0).unsqueeze(-1)
        # once per step, like Box2D v3 (orientations barely change within one step):
        #   J_a, J_b (E, C, 3, 6) rows [n, t1, t2] of each side: row velocity = J_a V_a + J_b V_b, V = [v, ω]
        #   W_a, W_b (E, C, 6, 3) = M⁻¹ Jᵀ, mass (E, C, 3) split effective mass per row
        # point velocity v + ω × r projected on d = d·v + (r × d)·ω
        B = self.basis
        self.J_a = -torch.cat([B, torch.cross(self.ra.unsqueeze(-2).expand_as(B), B, dim=-1)], dim=-1)
        self.J_b = torch.cat([B, torch.cross(self.rb.unsqueeze(-2).expand_as(B), B, dim=-1)], dim=-1)
        self.W_a = w.inv_mass_jt(self.idx_a, self.J_a)
        self.W_b = w.inv_mass_jt(self.idx_b, self.J_b)
        # mass splitting: each side acts as if its inverse mass were k times larger (k = its contact count)
        inv = self.k_a * (self.J_a * self.W_a.transpose(-1, -2)).sum(-1) + self.k_b * (self.J_b * self.W_b.transpose(-1, -2)).sum(-1)
        self.mass = 1.0 / inv.clamp_min(1e-12)
        self.update_separation()
        # (E, C) closing speed before solving, used for restitution
        self.vn0 = self.row_velocity()[..., 0]

    def update_separation(self):
        # (E, C) separation, refreshed after every position update
        if self.C == 0:
            return
        w = self.world
        # current separation from how far the two anchor points (moving with their bodies) drifted since detection
        pa = w.pos[self.env, self.idx_a] + quat_rotate(w.quat[self.env, self.idx_a], self.local_ra)
        pb = w.pos[self.env, self.idx_b] + quat_rotate(w.quat[self.env, self.idx_b], self.local_rb)
        self.s = ((pb - pa) * self.normal).sum(-1) + self.s0

    def row_velocity(self):
        # (E, C, 3) relative velocity of the contact points along n, t1, t2
        V = self.world.velocity
        # small matrix-vector products as multiply + sum: batched cuBLAS gemv is very slow on tiny matrices
        return (self.J_a * V[self.env, self.idx_a].unsqueeze(-2)).sum(-1) + (self.J_b * V[self.env, self.idx_b].unsqueeze(-2)).sum(-1)

    def apply(self, lam):
        # apply row impulses lam (E, C, 3) to both bodies
        # accumulate=True sums several contacts on the same body
        V = self.world.velocity
        env = self.env.expand_as(self.idx_a)
        V.index_put_((env, self.idx_a), (self.W_a * lam.unsqueeze(-2)).sum(-1), accumulate=True)
        V.index_put_((env, self.idx_b), (self.W_b * lam.unsqueeze(-2)).sum(-1), accumulate=True)

    def warm_start(self):
        if self.C == 0:
            return
        self.apply(self.lam)

    def solve(self, use_bias):
        if self.C == 0:
            return
        w = self.world
        bias_rate, mass_scale, impulse_scale = w.contact_soft
        rv = self.row_velocity()
        s = self.s
        # speculative (s > 0): allow closing exactly the gap. overlapping: soft push-out, capped speed.
        spec = s > 0
        if use_bias:
            push = torch.clamp_min(bias_rate * torch.clamp_max(s + LINEAR_SLOP, 0.0), -MAX_PUSH_VELOCITY)
            bias = torch.where(spec, s / w.h, push)
            ms = torch.where(spec, 1.0, mass_scale)
            imp_s = torch.where(spec, 0.0, impulse_scale)
        else:
            bias = torch.where(spec, s / w.h, 0.0)
            ms, imp_s = 1.0, 0.0
        old_n = self.lam[..., 0]
        new_n = torch.clamp_min(old_n - ms * (rv[..., 0] + bias) * self.mass[..., 0] - imp_s * old_n, 0.0)
        new_n = torch.where(self.valid, new_n, 0.0)
        # Coulomb friction from the same velocity snapshot, bounded by the disk |λt| <= μ λn
        old_t = self.lam[..., 1:]
        new_t = old_t - rv[..., 1:] * self.mass[..., 1:]
        limit = (self.friction * new_n).unsqueeze(-1)
        t_len = torch.linalg.norm(new_t, dim=-1, keepdim=True)
        new_t = torch.where(t_len > limit, new_t * limit / t_len.clamp_min(1e-12), new_t)
        new_t = torch.where(self.valid.unsqueeze(-1), new_t, 0.0)
        new = torch.cat([new_n.unsqueeze(-1), new_t], dim=-1)
        self.apply(new - self.lam)
        self.lam = new

    def apply_restitution(self):
        # bounce: drive the normal velocity to -e * (closing speed before the step), Box2D b2ApplyRestitution
        if self.C == 0:
            return
        old = self.lam[..., 0]
        bounce = self.valid & (self.restitution > 0) & (self.vn0 < -RESTITUTION_THRESHOLD) & (old > 0)
        vn = self.row_velocity()[..., 0]
        new = torch.where(bounce, torch.clamp_min(old - (vn + self.restitution * self.vn0) * self.mass[..., 0], 0.0), old)
        self.apply(torch.stack([new - old, torch.zeros_like(old), torch.zeros_like(old)], dim=-1))
        self.lam = torch.cat([new.unsqueeze(-1), self.lam[..., 1:]], dim=-1)

    def reset(self, env_mask):
        self.lam.masked_fill_(env_mask.view(-1, 1, 1), 0.0)
        self.valid.masked_fill_(env_mask.view(-1, 1), False)

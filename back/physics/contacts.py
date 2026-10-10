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
# default contact buffer capacity per env (scene settings "max_contacts" overrides). if more slots touch, the
# shallowest ones are dropped
MAX_CONTACTS = 64
# omni wheels (cylinder shape with "omni": true): friction across the wheel is only the small rolling friction of its
# passive rollers, so the wheel grips along its rolling direction and slides freely along its axle
ROLLER_FRICTION = 0.02
# sticky friction anchors are dropped once they're this far from the current contact point (m): a rolling wheel's
# anchor rotates away from the contact (rolling isn't slipping), a held block's anchors stay put
ANCHOR_RANGE = 0.005
# {(type_a, type_b): contact slots per pair}
SLOTS = {(PLANE, SPHERE): 1, (PLANE, BOX): 8, (PLANE, CYLINDER): 8, (SPHERE, SPHERE): 1, (SPHERE, BOX): 1,
         (SPHERE, CYLINDER): 1, (BOX, BOX): 25, (BOX, CYLINDER): 24, (CYLINDER, CYLINDER): 32}


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
        for b in bodies:
            assert not b['shape'].get('omni', False) or b['shape']['type'] == 'cylinder', f"{b['id']}: only cylinders can be omni wheels"
        omni = torch.tensor([b['shape'].get('omni', False) for b in bodies], device=dev, dtype=torch.bool)
        # (S,) slot touches an omni wheel as body B (cylinders are body B of every pair except cylinder-cylinder,
        # where only B's rolling direction is used: fine for a wheel bumping another robot's base)
        self.slot_omni = omni[self.slot_b]
        self.slot_restitution = torch.maximum(rest[self.slot_a], rest[self.slot_b])
        # any pair that can bounce at all
        self.bouncy = bool((self.slot_restitution > 0).any())

        E = world.E
        # contact buffer capacity
        self.C = min(self.S, world.scene['settings'].get('max_contacts', MAX_CONTACTS))
        # (E, 1) env index, broadcasts against (E, C) body indices
        self.env = torch.arange(E, device=dev).unsqueeze(-1)
        # (E, C) slot held by each buffer entry, (E, C, 3) accumulated impulse [normal, t1, t2], (E, C) entry in use.
        # the impulses are kept between steps and handed to the same slot next step (warm starting)
        self.slot = torch.zeros(E, self.C, dtype=torch.long, device=dev)
        self.lam = torch.zeros(E, self.C, 3, device=dev)
        self.valid = torch.zeros(E, self.C, dtype=torch.bool, device=dev)
        # sticky friction (Catto's Solver2D "sticky", like PhysX friction anchors): (E, C, 3) where the contact first
        # touched, in A's and B's body frames, kept while the slot persists. friction also pulls these two points back
        # together, so a held block can't slowly creep out. (E, C) sliding: friction hit the Coulomb limit -> re-anchor
        self.anchor_a = torch.zeros(E, self.C, 3, device=dev)
        self.anchor_b = torch.zeros(E, self.C, 3, device=dev)
        self.sliding = torch.zeros(E, self.C, dtype=torch.bool, device=dev)

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
            elif key == (BOX, CYLINDER):
                out = collision.box_cylinder(pa, qa, w.half[a], pb, qb, w.radius[b], w.half_height[b])
            else:
                out = collision.cylinder_cylinder(pa, qa, w.radius[a], w.half_height[a], pb, qb, w.radius[b], w.half_height[b])
            points.append(out[0].flatten(1, 2))
            normals.append(out[1].flatten(1, 2))
            depths.append(out[2].flatten(1, 2))
        # (E, S) every slot's depth, -inf for slots out of range
        depth = torch.cat(depths, 1)
        depth = torch.where(depth > -SPECULATIVE_DISTANCE, depth, -float('inf'))
        # fill the contact buffer with the deepest slots. (E, C)
        depth, slot = depth.topk(self.C, dim=1)
        valid = depth > -float('inf')
        # hand last step's per-slot state to the same slot (same slot = same geometric feature), via a full slot table:
        # [impulse (3), anchor_a (3), anchor_b (3), sliding, was touching]
        state = torch.cat([self.lam, self.anchor_a, self.anchor_b, self.sliding.unsqueeze(-1).float(), torch.ones_like(self.lam[..., :1])], dim=-1)
        table = torch.zeros(w.E, self.S, 11, device=w.device)
        table.scatter_(1, self.slot.unsqueeze(-1).expand(-1, -1, 11), torch.where(self.valid.unsqueeze(-1), state, 0.0))
        prev = table.gather(1, slot.unsqueeze(-1).expand(-1, -1, 11))
        # touching this step and last step: warm start, keep the anchors unless it was sliding
        keep = valid & (prev[..., 10] > 0.5)
        self.lam = torch.where(keep.unsqueeze(-1), prev[..., :3], 0.0)
        stick = (keep & (prev[..., 9] < 0.5)).unsqueeze(-1)
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
        near = ((torch.linalg.norm(quat_rotate(quat_a, prev[..., 3:6]) - self.ra, dim=-1) < ANCHOR_RANGE)
                & (torch.linalg.norm(quat_rotate(quat_b, prev[..., 6:9]) - self.rb, dim=-1) < ANCHOR_RANGE)).unsqueeze(-1)
        self.anchor_a = torch.where(stick & near, prev[..., 3:6], self.local_ra)
        self.anchor_b = torch.where(stick & near, prev[..., 6:9], self.local_rb)
        self.sliding = torch.zeros_like(valid)
        # (E, C, 3, 3) contact basis rows: normal, then friction directions t1, t2 orthogonal to it.
        # omni wheels: t1 = rolling direction (axle × n), t2 = along the axle
        n = self.normal
        ref = torch.where((n[..., :1].abs() < 0.9), const([1., 0., 0.], n.device), const([0., 1., 0.], n.device))
        axle = quat_rotate(quat_b, const([0., 1., 0.], n.device).expand_as(n))
        rolling = torch.cross(axle, n, dim=-1)
        omni = self.slot_omni[slot] & (torch.linalg.norm(rolling, dim=-1) > 0.1)
        t1 = torch.where(omni.unsqueeze(-1), rolling, torch.cross(n, ref, dim=-1))
        t1 = t1 / torch.linalg.norm(t1, dim=-1, keepdim=True)
        self.basis = torch.stack([n, t1, torch.cross(n, t1, dim=-1)], dim=-2)
        # (E, C, 2) friction coefficient along t1, t2, and (E, C) omni contact: its two directions are independent
        # (the rollers decouple them), so they're clamped separately instead of by one circular cone
        self.mu = torch.stack([self.friction, torch.where(omni, ROLLER_FRICTION, self.friction)], dim=-1)
        self.omni = omni
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
        pos_a, quat_a = w.pos[self.env, self.idx_a], w.quat[self.env, self.idx_a]
        pos_b, quat_b = w.pos[self.env, self.idx_b], w.quat[self.env, self.idx_b]
        pa = pos_a + quat_rotate(quat_a, self.local_ra)
        pb = pos_b + quat_rotate(quat_b, self.local_rb)
        self.s = ((pb - pa) * self.normal).sum(-1) + self.s0
        # (E, C, 2) tangential drift between the sticky anchors along t1, t2
        drift = (pos_b + quat_rotate(quat_b, self.anchor_b)) - (pos_a + quat_rotate(quat_a, self.anchor_a))
        self.drift = (self.basis[..., 1:, :] * drift.unsqueeze(-2)).sum(-1)

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
        # with bias, static friction also removes the drift since the contact started (soft, like the normal row)
        old_t = self.lam[..., 1:]
        if use_bias:
            bias_t = (bias_rate * self.drift).clamp(-MAX_PUSH_VELOCITY, MAX_PUSH_VELOCITY)
            new_t = old_t - mass_scale * (rv[..., 1:] + bias_t) * self.mass[..., 1:] - impulse_scale * old_t
        else:
            new_t = old_t - rv[..., 1:] * self.mass[..., 1:]
        # Coulomb: circular cone |λt| <= μ λn, or per direction |λi| <= μi λn for omni wheels
        limit = self.mu * new_n.unsqueeze(-1)
        t_len = torch.linalg.norm(new_t, dim=-1, keepdim=True)
        circle = torch.where(t_len > limit[..., :1], new_t * limit[..., :1] / t_len.clamp_min(1e-12), new_t)
        boxed = torch.maximum(torch.minimum(new_t, limit), -limit)
        clamped = torch.where(self.omni.unsqueeze(-1), boxed, circle)
        self.sliding = ((clamped - new_t).abs().amax(-1) > 1e-9) & self.valid
        new_t = clamped
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
        self.sliding.masked_fill_(env_mask.view(-1, 1), False)

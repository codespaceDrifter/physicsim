# Narrowphase collision between primitive shapes, batched over (envs E, pairs P).
#
# every function returns fixed-size contact "slots" so tensor shapes never change between steps:
#   point  (E, P, K, 3)  world contact point (midway between the two surfaces)
#   normal (E, P, K, 3)  unit normal pointing from body A to body B
#   depth  (E, P, K)     penetration depth (> 0 overlapping, < 0 separated). invalid slots get -inf.
# K is the max number of contact points for that pair type. a slot's index always means the same
# geometric feature (e.g. "box corner 3"), which keeps warm starting stable between steps.
#
# shape params: box half extents h (P, 3), sphere radius (P,), cylinder radius (P,) + half height (P,).
# cylinder axis = local +y. plane normal = local +y, plane is always body A.
import math
import torch
from physics.math3d import quat_rotate, quat_rotate_inv, quat_to_matrix, const, one_hot

# [[sx, sy, sz]] the 8 box corner sign patterns
CORNER_SIGNS = [[sx, sy, sz] for sx in (-1., 1.) for sy in (-1., 1.) for sz in (-1., 1.)]


def plane_sphere(pA, qA, pB, rB):
    # (E, P, 3) plane normal
    n = quat_rotate(qA, const([0., 1., 0.], pA.device).expand_as(pA))
    # (E, P) height of sphere center above plane
    dist = (n * (pB - pA)).sum(-1)
    depth = rB - dist
    # midway between sphere bottom (pB - n r) and its projection on the plane (pB - n dist)
    point = pB - n * ((rB + dist) / 2).unsqueeze(-1)
    return point.unsqueeze(2), n.unsqueeze(2), depth.unsqueeze(2)


def plane_box(pA, qA, pB, qB, hB):
    n = quat_rotate(qA, const([0., 1., 0.], pA.device).expand_as(pA))
    # (E, P, 8, 3) world corners
    corners = pB.unsqueeze(2) + quat_rotate(qB.unsqueeze(2), const(CORNER_SIGNS, pB.device) * hB.unsqueeze(-2))
    # (E, P, 8)
    dist = (n.unsqueeze(2) * (corners - pA.unsqueeze(2))).sum(-1)
    point = corners - n.unsqueeze(2) * (dist / 2).unsqueeze(-1)
    return point, n.unsqueeze(2).expand_as(point), -dist


def plane_cylinder(pA, qA, pB, qB, rB, hhB):
    n = quat_rotate(qA, const([0., 1., 0.], pA.device).expand_as(pA))
    # (E, P, 3) cylinder axis
    a = quat_rotate(qB, const([0., 1., 0.], pA.device).expand_as(pA))
    # deepest rim direction: -n projected onto the cap plane
    d = -(n - (n * a).sum(-1, keepdim=True) * a)
    d_len = torch.linalg.norm(d, dim=-1, keepdim=True)
    # cylinder standing upright on a cap: every rim direction is equally deep, use local +x
    fallback = quat_rotate(qB, const([1., 0., 0.], pA.device).expand_as(pA))
    d = torch.where(d_len < 1e-6, fallback, d / d_len.clamp_min(1e-12))
    e = torch.cross(a, d, dim=-1)
    # (E, P, 4, 3) four rim directions 90° apart, deepest first
    dirs = torch.stack([d, e, -d, -e], dim=2)
    # (E, P, 2, 3) cap centers
    caps = pB.unsqueeze(2) + torch.stack([a, -a], dim=2) * hhB[:, None, None]
    # (E, P, 8, 3) rim points: 2 caps x 4 directions
    rim = (caps.unsqueeze(3) + dirs.unsqueeze(2) * rB[:, None, None, None]).flatten(2, 3)
    dist = (n.unsqueeze(2) * (rim - pA.unsqueeze(2))).sum(-1)
    point = rim - n.unsqueeze(2) * (dist / 2).unsqueeze(-1)
    return point, n.unsqueeze(2).expand_as(point), -dist


def sphere_sphere(pA, rA, pB, rB):
    delta = pB - pA
    dist = torch.linalg.norm(delta, dim=-1)
    # coincident centers: any direction works, use +x
    ex = const([1., 0., 0.], pA.device).expand_as(delta)
    n = torch.where((dist < 1e-9).unsqueeze(-1), ex, delta / dist.clamp_min(1e-9).unsqueeze(-1))
    depth = rA + rB - dist
    point = pA + n * (rA - depth / 2).unsqueeze(-1)
    return point.unsqueeze(2), n.unsqueeze(2), depth.unsqueeze(2)


def box_closest(c, h):
    # closest surface feature of a box (local frame, half extents h) to local points c.
    # c (..., 3), h broadcastable (..., 3) -> surface point (..., 3), outward normal (..., 3), signed distance (...)
    q = torch.maximum(torch.minimum(c, h), -h)
    diff = c - q
    out_dist = torch.linalg.norm(diff, dim=-1)
    outside = out_dist > 1e-9
    # inside: exit through the nearest face
    face_gap = h - c.abs()
    axis = face_gap.argmin(-1)
    onehot = one_hot(axis, 3)
    sign = torch.where(c >= 0, 1.0, -1.0)
    in_normal = onehot * sign
    in_point = c * (1 - onehot) + onehot * sign * h
    in_dist = -(face_gap * onehot).sum(-1)
    point = torch.where(outside.unsqueeze(-1), q, in_point)
    normal = torch.where(outside.unsqueeze(-1), diff / out_dist.clamp_min(1e-9).unsqueeze(-1), in_normal)
    return point, normal, torch.where(outside, out_dist, in_dist)


def cylinder_closest(c, r, hh):
    # closest surface feature of a cylinder (local frame, axis y) to local points c.
    # c (..., 3), r and hh broadcastable (...) -> surface point (..., 3), outward normal (..., 3), signed distance (...)
    radial = c * const([1., 0., 1.], c.device)
    rho = torch.linalg.norm(radial, dim=-1)
    ex = const([1., 0., 0.], c.device).expand_as(c)
    rdir = torch.where((rho < 1e-9).unsqueeze(-1), ex, radial / rho.clamp_min(1e-9).unsqueeze(-1))
    ey = const([0., 1., 0.], c.device).expand_as(c)
    y = c[..., 1]
    q = rdir * torch.minimum(rho, r).unsqueeze(-1) + ey * torch.maximum(torch.minimum(y, hh), -hh).unsqueeze(-1)
    diff = c - q
    out_dist = torch.linalg.norm(diff, dim=-1)
    outside = out_dist > 1e-9
    # inside: exit through the side or through the nearer cap, whichever is closer
    side_gap = r - rho
    cap_gap = hh - y.abs()
    use_side = (side_gap < cap_gap).unsqueeze(-1)
    ysign = torch.where(y >= 0, 1.0, -1.0).unsqueeze(-1)
    in_normal = torch.where(use_side, rdir, ey * ysign)
    in_point = torch.where(use_side, rdir * r.unsqueeze(-1) + ey * y.unsqueeze(-1), radial + ey * ysign * hh.unsqueeze(-1))
    in_dist = -torch.minimum(side_gap, cap_gap)
    point = torch.where(outside.unsqueeze(-1), q, in_point)
    normal = torch.where(outside.unsqueeze(-1), diff / out_dist.clamp_min(1e-9).unsqueeze(-1), in_normal)
    return point, normal, torch.where(outside, out_dist, in_dist)


def sphere_box(pA, rA, pB, qB, hB):
    # sphere center in box frame
    c = quat_rotate_inv(qB, pA - pB)
    surf, n_out, sd = box_closest(c, hB)
    # outward box normal points toward the sphere, so A->B is the opposite
    n_out = quat_rotate(qB, n_out)
    surf = pB + quat_rotate(qB, surf)
    depth = rA - sd
    point = (surf + pA - n_out * rA.unsqueeze(-1)) / 2
    return point.unsqueeze(2), (-n_out).unsqueeze(2), depth.unsqueeze(2)


def sphere_cylinder(pA, rA, pB, qB, rB, hhB):
    c = quat_rotate_inv(qB, pA - pB)
    surf, n_out, sd = cylinder_closest(c, rB.expand_as(c[..., 0]), hhB.expand_as(c[..., 0]))
    n_out = quat_rotate(qB, n_out)
    surf = pB + quat_rotate(qB, surf)
    depth = rA - sd
    point = (surf + pA - n_out * rA.unsqueeze(-1)) / 2
    return point.unsqueeze(2), (-n_out).unsqueeze(2), depth.unsqueeze(2)


def box_cylinder(pA, qA, hA, pB, qB, rB, hhB):
    # approximate: 16 cylinder rim points tested against the box + 8 box corners tested against the cylinder.
    # misses box-edge-vs-cylinder-side contacts with no point inside; fine for wheels bumping blocks.
    E, P = pA.shape[:2]
    # (8, 3) unit ring of 8 directions in the cap plane
    ring = const([[math.cos(i * math.pi / 4), 0.0, math.sin(i * math.pi / 4)] for i in range(8)], pA.device)
    # (P, 16, 3) rim points in cylinder frame
    rim_local = torch.cat([ring * rB[:, None, None] + const([0., 1., 0.], pA.device) * hhB[:, None, None],
                           ring * rB[:, None, None] - const([0., 1., 0.], pA.device) * hhB[:, None, None]], dim=1)
    # (E, P, 16, 3)
    rim = pB.unsqueeze(2) + quat_rotate(qB.unsqueeze(2), rim_local.expand(E, P, 16, 3))
    c = quat_rotate_inv(qA.unsqueeze(2), rim - pA.unsqueeze(2))
    surf, n_out, sd = box_closest(c, hA.unsqueeze(-2))
    # cylinder point inside box A must move out along the box's outward normal
    n_rim = quat_rotate(qA.unsqueeze(2), n_out)
    p_rim = (pA.unsqueeze(2) + quat_rotate(qA.unsqueeze(2), surf) + rim) / 2
    # (E, P, 8, 3) box corners
    corners = pA.unsqueeze(2) + quat_rotate(qA.unsqueeze(2), const(CORNER_SIGNS, pA.device) * hA.unsqueeze(-2))
    c = quat_rotate_inv(qB.unsqueeze(2), corners - pB.unsqueeze(2))
    surf, n_out, sd2 = cylinder_closest(c, rB[:, None].expand_as(c[..., 0]), hhB[:, None].expand_as(c[..., 0]))
    # box corner inside cylinder B: B moves away along -outward normal of the cylinder
    n_corner = -quat_rotate(qB.unsqueeze(2), n_out)
    p_corner = (pB.unsqueeze(2) + quat_rotate(qB.unsqueeze(2), surf) + corners) / 2
    return torch.cat([p_rim, p_corner], 2), torch.cat([n_rim, n_corner], 2), torch.cat([-sd, -sd2], 2)


def cylinder_cylinder(pA, qA, rA, hhA, pB, qB, rB, hhB):
    # approximate like box_cylinder: 16 points around the middle of each cylinder's side tested against the other solid
    # (covers two robot bases bumping side to side, and wheels touching)
    E, P = pA.shape[:2]
    # (16, 3) ring of 16 directions around the axis
    ring = const([[math.cos(i * math.pi / 8), 0.0, math.sin(i * math.pi / 8)] for i in range(16)], pA.device)
    points, normals, depths = [], [], []  # [(E, P, 16, 3)], [(E, P, 16, 3)], [(E, P, 16)]
    for (p1, q1, r1, h1), (p2, q2, r2, h2), sign in (((pA, qA, rA, hhA), (pB, qB, rB, hhB), -1.0), ((pB, qB, rB, hhB), (pA, qA, rA, hhA), 1.0)):
        # (P, 16, 3) points around the middle of cylinder 1's side (at the caps' edges a side-by-side overlap of thin
        # cylinders reads as touching a cap), then world
        rim_local = ring * r1[:, None, None]
        rim = p1.unsqueeze(2) + quat_rotate(q1.unsqueeze(2), rim_local.expand(E, P, 16, 3))
        c = quat_rotate_inv(q2.unsqueeze(2), rim - p2.unsqueeze(2))
        surf, n_out, sd = cylinder_closest(c, r2[:, None].expand_as(c[..., 0]), h2[:, None].expand_as(c[..., 0]))
        # A's rim inside B: A moves out along B's outward normal, so A -> B is minus it. B's rim inside A: plus A's.
        normals.append(sign * quat_rotate(q2.unsqueeze(2), n_out))
        points.append((p2.unsqueeze(2) + quat_rotate(q2.unsqueeze(2), surf) + rim) / 2)
        depths.append(-sd)
    return torch.cat(points, 2), torch.cat(normals, 2), torch.cat(depths, 2)


def box_box(pA, qA, hA, pB, qB, hB):
    # separating axis test over 15 axes (3 face A, 3 face B, 9 edge×edge), then:
    #   face axis: contact polygon = intersection of the reference face rectangle and the incident face
    #     projected onto it. its vertices are always among 24 candidate features (fixed slots 0..23):
    #     4 incident corners inside the ref rect, 4 ref corners inside the incident quad, 16 edge×edge crossings.
    #   edge axis: one point, closest points between the two edges (slot 24).
    E, P = pA.shape[:2]
    hA = hA.expand(E, P, 3)
    hB = hB.expand(E, P, 3)
    # (E, P, 3, 3) columns = box axes in world
    RA = quat_to_matrix(qA)
    RB = quat_to_matrix(qB)
    d = pB - pA
    # (E, P, 3, 3) C[i, j] = a_i · b_j
    C = (RA.unsqueeze(-1) * RB.unsqueeze(-2)).sum(-3)
    absC = C.abs() + 1e-6
    # (E, P, 3) d in A frame / B frame
    dA = (RA * d.unsqueeze(-1)).sum(-2)
    dB = (RB * d.unsqueeze(-1)).sum(-2)
    # (E, P, 3) separation along each face axis (> 0 separated)
    sepA = dA.abs() - (hA + (absC * hB.unsqueeze(-2)).sum(-1))
    sepB = dB.abs() - (hB + (absC * hA.unsqueeze(-1)).sum(-2))
    # (E, P, 3, 3, 3) edge axes L_ij = a_i × b_j
    L = torch.cross(RA.transpose(-1, -2).unsqueeze(-2).expand(E, P, 3, 3, 3),
                    RB.transpose(-1, -2).unsqueeze(-3).expand(E, P, 3, 3, 3), dim=-1)
    L_len = torch.linalg.norm(L, dim=-1)
    L = L / L_len.clamp_min(1e-9).unsqueeze(-1)
    # (E, P, 3, 3) projected radii of both boxes on each edge axis
    # |L · a_k| for each box axis k: (E, P, 3, 3, 3)
    rA_L = (hA[:, :, None, None, :] * (L.unsqueeze(-1) * RA[:, :, None, None]).sum(-2).abs()).sum(-1)
    rB_L = (hB[:, :, None, None, :] * (L.unsqueeze(-1) * RB[:, :, None, None]).sum(-2).abs()).sum(-1)
    dL = (L * d[:, :, None, None, :]).sum(-1)
    # near-parallel edges give a degenerate axis that the face axes already cover
    sepE = torch.where(L_len > 1e-5, dL.abs() - rA_L - rB_L, torch.full_like(dL, -float('inf'))).flatten(2)

    sepA_max, iA = sepA.max(-1)
    sepB_max, iB = sepB.max(-1)
    sepE_max, iE = sepE.max(-1)
    # prefer face axes (stable manifolds): pick B only if clearly better than A, edge only if clearly better than faces.
    # with parallel faces (a gripper finger flat on a block) both face axes are nearly tied; a tight tolerance makes the
    # reference box flip every step, renaming the contact slots and wiping the warm-started friction (Box2D's
    # b2CollidePolygons uses the same relative + absolute tolerance)
    ref_is_B = sepB_max > 0.95 * sepA_max + 0.0005
    face_sep = torch.maximum(sepA_max, sepB_max)
    use_edge = sepE_max * 1.05 - 1e-4 > face_sep

    # ---- face case ----
    # reference box (owns the face), incident box (other one). ref normal points from ref toward incident.
    ref_R = torch.where(ref_is_B[..., None, None], RB, RA)
    inc_R = torch.where(ref_is_B[..., None, None], RA, RB)
    ref_h = torch.where(ref_is_B[..., None], hB, hA)
    inc_h = torch.where(ref_is_B[..., None], hA, hB)
    ref_c = torch.where(ref_is_B[..., None], pB, pA)
    inc_c = torch.where(ref_is_B[..., None], pA, pB)
    i_ref = torch.where(ref_is_B, iB, iA)
    # (E, P, 3) one-hot of the reference face axis and its two tangent axes
    oh0 = one_hot(i_ref, 3)
    oh1 = one_hot((i_ref + 1) % 3, 3)
    oh2 = one_hot((i_ref + 2) % 3, 3)
    axis_r = (ref_R * oh0.unsqueeze(-2)).sum(-1)
    nr = axis_r * torch.where(((inc_c - ref_c) * axis_r).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
    u1 = (ref_R * oh1.unsqueeze(-2)).sum(-1)
    u2 = (ref_R * oh2.unsqueeze(-2)).sum(-1)
    e1 = (ref_h * oh1).sum(-1)
    e2 = (ref_h * oh2).sum(-1)
    face_c = ref_c + nr * (ref_h * oh0).sum(-1, keepdim=True)
    # incident face = face of the incident box most anti-parallel to nr
    inc_dots = (inc_R * nr.unsqueeze(-1)).sum(-2)
    j_inc = inc_dots.abs().argmax(-1)
    ohj0 = one_hot(j_inc, 3)
    ohj1 = one_hot((j_inc + 1) % 3, 3)
    ohj2 = one_hot((j_inc + 2) % 3, 3)
    bj = (inc_R * ohj0.unsqueeze(-2)).sum(-1)
    nI = -bj * torch.where((inc_dots * ohj0).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
    cI = inc_c + nI * (inc_h * ohj0).sum(-1, keepdim=True)
    w1 = (inc_R * ohj1.unsqueeze(-2)).sum(-1) * (inc_h * ohj1).sum(-1, keepdim=True)
    w2 = (inc_R * ohj2.unsqueeze(-2)).sum(-1) * (inc_h * ohj2).sum(-1, keepdim=True)
    # (E, P, 4, 3) incident face corners in cyclic order
    inc_v = torch.stack([cI + w1 + w2, cI - w1 + w2, cI - w1 - w2, cI + w1 - w2], dim=2)
    # (E, P, 4, 2) incident corners in reference-face 2D coords
    rel = inc_v - face_c.unsqueeze(2)
    inc2 = torch.stack([(rel * u1.unsqueeze(2)).sum(-1), (rel * u2.unsqueeze(2)).sum(-1)], dim=-1)
    # (E, P, 4, 2) reference rectangle corners, cyclic
    ref2 = torch.stack([torch.stack([e1, e2], -1), torch.stack([-e1, e2], -1),
                        torch.stack([-e1, -e2], -1), torch.stack([e1, -e2], -1)], dim=2)
    tol = 1e-6
    # (a) incident corners inside the reference rectangle
    ok_a = (inc2[..., 0].abs() <= e1.unsqueeze(-1) + tol) & (inc2[..., 1].abs() <= e2.unsqueeze(-1) + tol)
    # (b) reference corners inside the incident quad: same side of all 4 quad edges.
    # skip ones sitting on an incident corner (already in (a)), so stacked equal boxes don't double count
    # (E, P, 4 edges, 2)
    inc_edge = inc2.roll(-1, dims=2) - inc2
    # (E, P, 4 ref corners, 4 edges)
    to_pt = ref2.unsqueeze(3) - inc2.unsqueeze(2)
    side = inc_edge[..., 0].unsqueeze(2) * to_pt[..., 1] - inc_edge[..., 1].unsqueeze(2) * to_pt[..., 0]
    on_inc_corner = (torch.linalg.norm(to_pt, dim=-1) < 1e-5).any(-1)
    ok_b = ((side >= -tol).all(-1) | (side <= tol).all(-1)) & ~on_inc_corner
    # (c) reference edge i × incident edge k crossings, strictly inside both edges (endpoints are corners, covered by (a)/(b)). (E, P, 4, 4)
    ref_edge = ref2.roll(-1, dims=2) - ref2
    dr = ref_edge.unsqueeze(3)
    di = inc_edge.unsqueeze(2)
    w = inc2.unsqueeze(2) - ref2.unsqueeze(3)
    denom = dr[..., 0] * di[..., 1] - dr[..., 1] * di[..., 0]
    safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    t = (w[..., 0] * di[..., 1] - w[..., 1] * di[..., 0]) / safe
    s = (w[..., 0] * dr[..., 1] - w[..., 1] * dr[..., 0]) / safe
    ok_c = (denom.abs() > 1e-12) & (t > 1e-5) & (t < 1 - 1e-5) & (s > 1e-5) & (s < 1 - 1e-5)
    cross2 = (ref2.unsqueeze(3) + t.unsqueeze(-1) * dr).flatten(2, 3)
    # (E, P, 24, 2) all candidate polygon vertices
    cand2 = torch.cat([inc2, ref2, cross2], dim=2)
    ok = torch.cat([ok_a, ok_b, ok_c.flatten(2)], dim=2)
    # lift onto the reference face, then walk along nr to the incident face plane
    Pw = face_c.unsqueeze(2) + cand2[..., :1] * u1.unsqueeze(2) + cand2[..., 1:] * u2.unsqueeze(2)
    # (E, P, 24) distance from ref face to incident plane along nr (> 0 = gap)
    gap = ((cI.unsqueeze(2) - Pw) * nI.unsqueeze(2)).sum(-1) / (nI * nr).sum(-1, keepdim=True)
    face_point = Pw + nr.unsqueeze(2) * (gap / 2).unsqueeze(-1)
    face_depth = torch.where(ok & ~use_edge.unsqueeze(-1), -gap, torch.full_like(gap, -float('inf')))
    # contact normal is always A -> B
    n_face = torch.where(ref_is_B.unsqueeze(-1), -nr, nr)

    # ---- edge case ----
    i_e = iE // 3
    j_e = iE % 3
    ohA = one_hot(i_e, 3)
    ohB = one_hot(j_e, 3)
    nE = (L.flatten(2, 3) * one_hot(iE, 9).unsqueeze(-1)).sum(-2)
    nE = nE * torch.where((nE * d).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
    # edge of A parallel to a_i that is furthest toward B, edge of B parallel to b_j furthest toward A
    signA = torch.where((RA * nE.unsqueeze(-1)).sum(-2) >= 0, 1.0, -1.0)
    signB = torch.where((RB * nE.unsqueeze(-1)).sum(-2) >= 0, 1.0, -1.0)
    mA = pA + (RA * (signA * hA * (1 - ohA)).unsqueeze(-2)).sum(-1)
    mB = pB - (RB * (signB * hB * (1 - ohB)).unsqueeze(-2)).sum(-1)
    ea = (RA * ohA.unsqueeze(-2)).sum(-1)
    eb = (RB * ohB.unsqueeze(-2)).sum(-1)
    # closest points mA + s ea, mB + t eb between the two edge lines, clamped to the edges
    r = mA - mB
    ab = (ea * eb).sum(-1)
    s_e = (ab * (eb * r).sum(-1) - (ea * r).sum(-1)) / (1 - ab * ab).clamp_min(1e-9)
    t_e = (eb * r).sum(-1) + s_e * ab
    s_e = torch.maximum(torch.minimum(s_e, (hA * ohA).sum(-1)), -(hA * ohA).sum(-1))
    t_e = torch.maximum(torch.minimum(t_e, (hB * ohB).sum(-1)), -(hB * ohB).sum(-1))
    edge_point = ((mA + s_e.unsqueeze(-1) * ea) + (mB + t_e.unsqueeze(-1) * eb)) / 2
    edge_depth = torch.where(use_edge, -sepE_max, torch.full_like(sepE_max, -float('inf')))

    point = torch.cat([face_point, edge_point.unsqueeze(2)], dim=2)
    normal = torch.cat([n_face.unsqueeze(2).expand(E, P, 24, 3), nE.unsqueeze(2)], dim=2)
    depth = torch.cat([face_depth, edge_depth.unsqueeze(2)], dim=2)
    return point, normal, depth


# capsules: a segment along local y from -hh to +hh (hh = half its length) swept by a sphere of radius r


def closest_on_segment(p, c, a, hh):
    # closest point to p (..., 3) on the segment c + a t, t in [-hh, hh] (a unit). -> (..., 3)
    t = torch.maximum(torch.minimum(((p - c) * a).sum(-1), hh), -hh)
    return c + a * t.unsqueeze(-1)


def plane_capsule(pA, qA, pB, qB, rB, hhB):
    # the two cap spheres against the plane
    n = quat_rotate(qA, const([0., 1., 0.], pA.device).expand_as(pA))
    a = quat_rotate(qB, const([0., 1., 0.], pA.device).expand_as(pA))
    # (E, P, 2, 3) cap centers
    ends = pB.unsqueeze(2) + torch.stack([a, -a], dim=2) * hhB[:, None, None]
    dist = (n.unsqueeze(2) * (ends - pA.unsqueeze(2))).sum(-1)
    point = ends - n.unsqueeze(2) * ((rB[:, None] + dist) / 2).unsqueeze(-1)
    return point, n.unsqueeze(2).expand_as(point), rB[:, None] - dist


def sphere_capsule(pA, rA, pB, qB, rB, hhB):
    # sphere against the nearest point of the capsule's segment
    a = quat_rotate(qB, const([0., 1., 0.], pA.device).expand_as(pA))
    return sphere_sphere(pA, rA, closest_on_segment(pA, pB, a, hhB), rB)


def capsule_capsule(pA, qA, rA, hhA, pB, qB, rB, hhB):
    # 5 slots: the closest points of the two segments (crossing capsules), then each segment end against the other
    # segment (parallel capsules lying side by side need two points to rest without rocking)
    a = quat_rotate(qA, const([0., 1., 0.], pA.device).expand_as(pA))
    b = quat_rotate(qB, const([0., 1., 0.], pA.device).expand_as(pA))
    # segment-segment closest points (Ericson, Real-Time Collision Detection 5.1.9) with unit directions
    r = pA - pB
    ab = (a * b).sum(-1)
    c = (a * r).sum(-1)
    f = (b * r).sum(-1)
    denom = 1 - ab * ab
    # parallel segments: any s works, the end slots carry the contact
    s = torch.where(denom > 1e-6, (ab * f - c) / denom.clamp_min(1e-6), torch.zeros_like(c))
    s = torch.maximum(torch.minimum(s, hhA), -hhA)
    t = torch.maximum(torch.minimum(ab * s + f, hhB), -hhB)
    s = torch.maximum(torch.minimum(ab * t - c, hhA), -hhA)
    endsA = pA.unsqueeze(2) + torch.stack([a, -a], dim=2) * hhA[:, None, None]
    endsB = pB.unsqueeze(2) + torch.stack([b, -b], dim=2) * hhB[:, None, None]
    # (E, P, 5, 3) matching points on segment A and segment B
    onA = torch.cat([(pA + a * s.unsqueeze(-1)).unsqueeze(2), endsA,
                     closest_on_segment(endsB, pA.unsqueeze(2), a.unsqueeze(2), hhA[:, None])], dim=2)
    onB = torch.cat([(pB + b * t.unsqueeze(-1)).unsqueeze(2),
                     closest_on_segment(endsA, pB.unsqueeze(2), b.unsqueeze(2), hhB[:, None]), endsB], dim=2)
    delta = onB - onA
    dist = torch.linalg.norm(delta, dim=-1)
    # segments touching: push apart perpendicular to both (or to A alone if they're parallel)
    perp = torch.cross(a, b, dim=-1)
    ref = torch.where(a[..., :1].abs() < 0.9, const([1., 0., 0.], pA.device), const([0., 1., 0.], pA.device))
    perp = torch.where((torch.linalg.norm(perp, dim=-1) < 1e-6).unsqueeze(-1), torch.cross(a, ref, dim=-1), perp)
    perp = perp / torch.linalg.norm(perp, dim=-1, keepdim=True).clamp_min(1e-9)
    n = torch.where((dist < 1e-9).unsqueeze(-1), perp.unsqueeze(2).expand_as(delta), delta / dist.clamp_min(1e-9).unsqueeze(-1))
    depth = rA[:, None] + rB[:, None] - dist
    point = onA + n * (rA[:, None] - depth / 2).unsqueeze(-1)
    return point, n, depth


def box_capsule(pA, qA, hA, pB, qB, rB, hhB):
    # 3 points of the capsule's segment, each tested as a sphere against the box: the two ends and the point of the
    # segment deepest in (or closest to) the box. the box's signed distance is convex along the segment, so a ternary
    # search finds that point
    c = quat_rotate_inv(qA, pB - pA)
    ax = quat_rotate_inv(qA, quat_rotate(qB, const([0., 1., 0.], pA.device).expand_as(pA)))
    hh = hhB.expand_as(c[..., 0])
    lo, hi = -hh, hh
    for _ in range(16):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        closer = box_closest(c + ax * m1.unsqueeze(-1), hA)[2] < box_closest(c + ax * m2.unsqueeze(-1), hA)[2]
        lo, hi = torch.where(closer, lo, m1), torch.where(closer, m2, hi)
    t = (lo + hi) / 2
    # (E, P, 3, 3) segment points in the box frame
    pts = c.unsqueeze(2) + ax.unsqueeze(2) * torch.stack([hh, -hh, t], dim=-1).unsqueeze(-1)
    surf, n_out, sd = box_closest(pts, hA.unsqueeze(-2))
    point = pA.unsqueeze(2) + quat_rotate(qA.unsqueeze(2), (surf + pts - n_out * rB[:, None, None]) / 2)
    # the box's outward normal points toward the capsule: that's A -> B
    return point, quat_rotate(qA.unsqueeze(2), n_out), rB[:, None] - sd


def segment_distance(p0, p1, q0, q1):
    # distance between segments p0-p1 and q0-q1 (..., 3) (Ericson, Real-Time Collision Detection 5.1.9; a zero-length
    # segment is a point) -> (...)
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    b, c = (d1 * d2).sum(-1), (d1 * r).sum(-1)
    p_point, q_point = a <= 1e-9, e <= 1e-9
    denom = a * e - b * b
    # general case: s on the line closest to segment 2 (any s if parallel), then t for that s, clamped
    s = torch.where(denom > 1e-9, ((b * f - c * e) / denom.clamp_min(1e-9)).clamp(0, 1), torch.zeros_like(a))
    t = (b * s + f) / e.clamp_min(1e-9)
    t_c = t.clamp(0, 1)
    # t clamped: recompute s for it
    s = torch.where(t != t_c, ((b * t_c - c) / a.clamp_min(1e-9)).clamp(0, 1), s)
    # segment 2 is a point: t = 0, s = its projection; segment 1 is a point: s = 0, t = its projection
    s = torch.where(q_point, (-c / a.clamp_min(1e-9)).clamp(0, 1), s)
    t_c = torch.where(q_point, 0.0, t_c)
    s = torch.where(p_point, 0.0, s)
    t_c = torch.where(p_point, torch.where(q_point, 0.0, (f / e.clamp_min(1e-9)).clamp(0, 1)), t_c)
    return torch.linalg.norm((p0 + d1 * s[..., None]) - (q0 + d2 * t_c[..., None]), dim=-1)

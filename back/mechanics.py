# Physics mechanics with batched tensor operations (linear only).
import torch
import config

G = 6.674e-11


def apply_gravity_batched(objects):
    # Pairwise gravity across all objects.
    if len(objects) <= 1:
        return

    N = len(objects)
    # (N, 3)
    positions = torch.stack([obj.position for obj in objects])
    # (N,)
    masses = torch.tensor([obj.mass for obj in objects], device=config.device)

    # (N, N, 3) delta[i,j] = pos[j] - pos[i]
    delta = positions.unsqueeze(0) - positions.unsqueeze(1)

    # (N, N)
    dist_sq = (delta ** 2).sum(dim=2)
    dist_sq = dist_sq + torch.eye(N, device=config.device) * 1e10

    # (N, N)
    dist = torch.sqrt(dist_sq)
    # (N, N, 3)
    direction = delta / dist.unsqueeze(2)

    # (N, N)
    mass_products = masses.unsqueeze(0) * masses.unsqueeze(1)
    # (N, N)
    force_mags = G * mass_products / dist_sq
    # (N, N, 3)
    force_vectors = direction * force_mags.unsqueeze(2)
    # (N, 3)
    total_forces = force_vectors.sum(dim=1)

    for i, obj in enumerate(objects):
        obj.force = obj.force + total_forces[i]


def apply_collision_response(obj_a, obj_b):
    # Linear-only impulse collision along the line between centers.
    # (3,)
    n = obj_b.position - obj_a.position
    n_norm = torch.norm(n)
    if n_norm < 1e-6:
        n = torch.tensor([1.0, 0.0, 0.0], device=config.device)
    else:
        n = n / n_norm

    # n points from A to B. v_rel_n > 0 means A approaching B.
    # (3,)
    v_rel = obj_a.velocity - obj_b.velocity
    # scalar
    v_rel_n = torch.dot(v_rel, n)

    # already separating
    if v_rel_n < 0:
        return

    e = torch.sqrt(torch.tensor(obj_a.restitution * obj_b.restitution, device=config.device))
    m_a_inv = 1.0 / obj_a.mass
    m_b_inv = 1.0 / obj_b.mass

    # scalar impulse magnitude
    j = -(1.0 + e) * v_rel_n / (m_a_inv + m_b_inv)
    # (3,)
    impulse = j * n

    # (3,)
    obj_a.velocity = obj_a.velocity + impulse * m_a_inv
    # (3,)
    obj_b.velocity = obj_b.velocity - impulse * m_b_inv

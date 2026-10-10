# Batched quaternion / vector math. quaternions are [w, x, y, z] on the last dim.
import torch

# {(values, device): tensor} cached device constants. building a tensor from a python list copies host -> device,
# which also isn't allowed while a CUDA graph is being captured, so every constant is built once and reused.
_CONSTS = {}


def const(values, device):
    # cached float32 tensor of `values` on `device`
    key = (str(values), str(device))
    if key not in _CONSTS:
        _CONSTS[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return _CONSTS[key]


def one_hot(idx, n):
    # float one-hot without F.one_hot (which syncs with the GPU to validate indices). (...) -> (..., n)
    return (idx.unsqueeze(-1) == torch.arange(n, device=idx.device)).float()


def quat_mul(a, b):
    # hamilton product a ⊗ b. (..., 4), (..., 4) -> (..., 4)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conj(q):
    # inverse of a unit quaternion. (..., 4) -> (..., 4)
    return q * const([1.0, -1.0, -1.0, -1.0], q.device)


def quat_rotate(q, v):
    # rotate vector v by unit quaternion q: v + 2w(u×v) + 2u×(u×v), u = vector part.
    # (..., 4), (..., 3) -> (..., 3)
    w = q[..., :1]
    u, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(u, v, dim=-1)
    return v + 2.0 * (w * uv + torch.cross(u, uv, dim=-1))


def quat_rotate_inv(q, v):
    # rotate v by q⁻¹ (world -> body frame). (..., 4), (..., 3) -> (..., 3)
    return quat_rotate(quat_conj(q), v)


def quat_from_rotvec(r):
    # exponential map: rotation vector (axis * angle) -> unit quaternion. exact for any angle.
    # (..., 3) -> (..., 4)
    angle = torch.linalg.norm(r, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(half)/angle -> 0.5 as angle -> 0 (taylor: 0.5 - angle²/48)
    small = angle < 1e-6
    scale = torch.where(small, 0.5 - angle * angle / 48.0, torch.sin(half) / torch.where(small, torch.ones_like(angle), angle))
    return torch.cat([torch.cos(half), r * scale], dim=-1)


def quat_to_matrix(q):
    # unit quaternion -> rotation matrix whose columns are the body axes in world. (..., 4) -> (..., 3, 3)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def quat_from_two_axes(x_axis):
    # unit quaternion whose rotated local +x equals the given unit vector (minimal rotation from +x).
    # used to build joint frames from an axis. (..., 3) -> (..., 4)
    ex = torch.zeros_like(x_axis)
    ex[..., 0] = 1.0
    d = (ex * x_axis).sum(-1, keepdim=True)
    c = torch.cross(ex, x_axis, dim=-1)
    q = torch.cat([1.0 + d, c], dim=-1)
    # axis == -x: rotate 180° about y
    flipped = const([0.0, 0.0, 1.0, 0.0], x_axis.device).expand_as(q)
    q = torch.where(d < -1.0 + 1e-6, flipped, q)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def spd_inverse(A):
    # inverse of symmetric positive definite matrices (..., n, n) by Cholesky: A = L Lᵀ -> A⁻¹ = L⁻ᵀ L⁻¹.
    # unlike torch.linalg.inv this can be recorded in a CUDA graph
    L = torch.linalg.cholesky_ex(A)[0]
    L_inv = torch.linalg.solve_triangular(L, torch.eye(A.shape[-1], device=A.device).expand_as(A), upper=False)
    return L_inv.transpose(-1, -2) @ L_inv


def wrap_angle(a):
    # wrap to (-π, π]
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi

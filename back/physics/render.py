# Batched camera rendering by analytic ray casting: one ray per pixel against every shape, all envs at once.
# output is what a robot's camera sees: (E, H, W, 3) uint8 RGB, row 0 = top.
import math
import torch
from physics.math3d import quat_mul, quat_rotate, quat_rotate_inv, const

PLANE, SPHERE, BOX, CYLINDER = 0, 1, 2, 3
# light direction (toward the light), ambient and diffuse weights
LIGHT = [0.4, 1.0, 0.3]
AMBIENT = 0.4
DIFFUSE = 0.6
# ground checker square size (m) and how much darker the odd squares are
CHECKER = 0.25
CHECKER_DARK = 0.8
SKY_TOP = [0.55, 0.7, 0.9]
SKY_HORIZON = [0.85, 0.9, 0.95]
FAR = 1e6


class Renderer:
    def __init__(self, world, camera_def, env_chunk=32):
        self.world = world
        dev = world.device
        self.id = camera_def['id']
        self.body = world.body_index[camera_def['body']]
        # camera offset in its body's frame
        self.local_pos = torch.tensor(camera_def['position'], device=dev, dtype=torch.float32)
        self.local_quat = torch.tensor(camera_def['rotation'], device=dev, dtype=torch.float32)
        self.W, self.H = camera_def['width'], camera_def['height']
        # envs rendered per batch, bounds the (E, pixels, shapes, 3) temporaries
        self.env_chunk = env_chunk
        # (H*W, 3) unit ray directions in the camera frame (looks down -z, up +y), row 0 = top
        tan_half = math.tan(math.radians(camera_def['fov']) / 2)
        ys = (1 - 2 * (torch.arange(self.H, device=dev) + 0.5) / self.H) * tan_half
        xs = (2 * (torch.arange(self.W, device=dev) + 0.5) / self.W - 1) * tan_half * self.W / self.H
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dirs = torch.stack([xx, yy, -torch.ones_like(xx)], dim=-1).view(-1, 3)
        self.dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
        # (N, 3) body colors 0..1
        self.colors = torch.tensor([[int(b['color'][i:i + 2], 16) / 255 for i in (1, 3, 5)] for b in world.scene['bodies']], device=dev)
        # (Ns,) body indices of each shape type
        self.by_type = {t: (world.shape_type == t).nonzero().view(-1) for t in (PLANE, SPHERE, BOX, CYLINDER)}

    def enable_cuda_graph(self):
        # record render() as a CUDA graph (one launch instead of thousands of small kernels). call after the world's
        # enable_cuda_graph(), so the poses it reads live at fixed addresses
        self.world.sync_graph_state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.render_eager()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # (E, H, W, 3) fixed-address output
            self.graph_out = self.render_eager()
        self.graph = graph

    def render(self):
        # (E, H, W, 3) uint8. with a CUDA graph, the result is overwritten by the next render: copy it to keep it
        if getattr(self, 'graph', None) is None:
            return self.render_eager()
        self.world.sync_graph_state()
        self.graph.replay()
        return self.graph_out

    def render_eager(self):
        w = self.world
        out = [self.render_envs(slice(s, s + self.env_chunk)) for s in range(0, w.E, self.env_chunk)]
        return torch.cat(out, 0)

    def render_envs(self, envs):
        w = self.world
        pos, quat = w.pos[envs], w.quat[envs]
        E = pos.shape[0]
        # camera pose in world. (E, 3), (E, 4)
        cam_q = quat_mul(quat[:, self.body], self.local_quat)
        origin = pos[:, self.body] + quat_rotate(quat[:, self.body], self.local_pos)
        # (E, R, 3) world ray directions, R = H*W
        d = quat_rotate(cam_q.unsqueeze(1), self.dirs.unsqueeze(0))
        R = d.shape[1]
        # closest hit so far: distance (E, R), world normal (E, R, 3), body index (E, R), -1 = sky
        best_t = torch.full((E, R), FAR, device=w.device)
        best_n = torch.zeros(E, R, 3, device=w.device)
        best_i = torch.full((E, R), -1, device=w.device, dtype=torch.long)

        def keep(t, n, idx):
            # merge candidate hits t (E, R, S) with normals n (E, R, S, 3) for shapes idx (S,) into the running best
            nonlocal best_t, best_n, best_i
            t_min, s = t.min(-1)
            closer = t_min < best_t
            best_t = torch.where(closer, t_min, best_t)
            best_n = torch.where(closer.unsqueeze(-1), n.gather(2, s[..., None, None].expand(E, R, 1, 3)).squeeze(2), best_n)
            best_i = torch.where(closer, idx[s], best_i)

        idx = self.by_type[PLANE]
        if len(idx):
            # (E, S, 3) plane normals and points
            n = quat_rotate(quat[:, idx], const([0., 1., 0.], w.device).expand(E, len(idx), 3))
            denom = d @ n.transpose(1, 2)
            t = ((pos[:, idx] - origin.unsqueeze(1)) * n).sum(-1).unsqueeze(1) / denom.clamp_max(-1e-9)
            t = torch.where((denom < -1e-9) & (t > 0), t, FAR)
            keep(t, n.unsqueeze(1).expand(E, R, len(idx), 3), idx)

        idx = self.by_type[SPHERE]
        if len(idx):
            # (E, S, 3) origin relative to centers
            oc = origin.unsqueeze(1) - pos[:, idx]
            r = w.radius[idx]
            # (E, R, S) ray-sphere quadratic t² + 2bt + c = 0
            b = d @ oc.transpose(1, 2)
            c = (oc * oc).sum(-1).unsqueeze(1) - r * r
            disc = b * b - c
            t = -b - torch.sqrt(disc.clamp_min(0))
            t = torch.where((disc > 0) & (t > 0), t, FAR)
            hit = origin[:, None, None] + d.unsqueeze(2) * t.unsqueeze(-1)
            n = (hit - pos[:, None, idx]) / r[:, None]
            keep(t, n, idx)

        idx = self.by_type[BOX]
        if len(idx):
            q = quat[:, idx]
            # ray in each box's frame: (E, 1, S, 3) origin, (E, R, S, 3) direction
            o_l = quat_rotate_inv(q, origin.unsqueeze(1) - pos[:, idx]).unsqueeze(1)
            d_l = quat_rotate_inv(q.unsqueeze(1), d.unsqueeze(2))
            h = w.half[idx]
            inv = 1.0 / torch.where(d_l.abs() < 1e-9, 1e-9, d_l)
            t1 = (-h - o_l) * inv
            t2 = (h - o_l) * inv
            # slab test: entry = latest of the per-axis entries, exit = earliest exit
            t_near, axis = torch.minimum(t1, t2).max(-1)
            t_far = torch.maximum(t1, t2).min(-1).values
            t = torch.where((t_near <= t_far) & (t_near > 0), t_near, FAR)
            # normal: entry axis, pointing against the ray
            n_l = (axis.unsqueeze(-1) == torch.arange(3, device=w.device)).float() * -torch.sign(d_l)
            keep(t, quat_rotate(q.unsqueeze(1), n_l), idx)

        idx = self.by_type[CYLINDER]
        if len(idx):
            q = quat[:, idx]
            o_l = quat_rotate_inv(q, origin.unsqueeze(1) - pos[:, idx]).unsqueeze(1)
            d_l = quat_rotate_inv(q.unsqueeze(1), d.unsqueeze(2))
            r, hh = w.radius[idx], w.half_height[idx]
            # side: (ox + t dx)² + (oz + t dz)² = r², hit valid if |y| <= hh
            a = d_l[..., 0] ** 2 + d_l[..., 2] ** 2
            b = o_l[..., 0] * d_l[..., 0] + o_l[..., 2] * d_l[..., 2]
            c = o_l[..., 0] ** 2 + o_l[..., 2] ** 2 - r * r
            disc = b * b - a * c
            t_side = (-b - torch.sqrt(disc.clamp_min(0))) / a.clamp_min(1e-12)
            y_side = o_l[..., 1] + t_side * d_l[..., 1]
            t_side = torch.where((disc > 0) & (t_side > 0) & (y_side.abs() <= hh), t_side, FAR)
            # caps: the plane y = ±hh facing the ray, hit valid if inside the radius
            cap_y = -torch.sign(d_l[..., 1]) * hh
            t_cap = (cap_y - o_l[..., 1]) / torch.where(d_l[..., 1].abs() < 1e-9, 1e-9, d_l[..., 1])
            px = o_l[..., 0] + t_cap * d_l[..., 0]
            pz = o_l[..., 2] + t_cap * d_l[..., 2]
            t_cap = torch.where((t_cap > 0) & (px * px + pz * pz <= r * r), t_cap, FAR)
            side = t_side < t_cap
            hit_l = o_l + d_l * torch.minimum(t_side, t_cap).unsqueeze(-1)
            n_side = hit_l * const([1., 0., 1.], w.device) / r[:, None]
            n_cap = const([0., 1., 0.], w.device) * torch.sign(cap_y).unsqueeze(-1)
            n_l = torch.where(side.unsqueeze(-1), n_side, n_cap)
            keep(torch.minimum(t_side, t_cap), quat_rotate(q.unsqueeze(1), n_l), idx)

        # shading: lambert + ambient on the body color, checkerboard on the ground, gradient sky
        light = const(LIGHT, w.device) / math.sqrt(sum(v * v for v in LIGHT))
        lambert = (best_n * light).sum(-1).clamp_min(0)
        color = self.colors[best_i.clamp_min(0)] * (AMBIENT + DIFFUSE * lambert).unsqueeze(-1)
        hit = origin.unsqueeze(1) + d * best_t.clamp_max(1e3).unsqueeze(-1)
        is_plane = (w.shape_type[best_i.clamp_min(0)] == PLANE) & (best_i >= 0)
        odd = (torch.floor(hit[..., 0] / CHECKER) + torch.floor(hit[..., 2] / CHECKER)) % 2 == 1
        color = torch.where((is_plane & odd).unsqueeze(-1), color * CHECKER_DARK, color)
        up = d[..., 1:2].clamp(0, 1)
        sky = const(SKY_HORIZON, w.device) * (1 - up) + const(SKY_TOP, w.device) * up
        color = torch.where((best_i >= 0).unsqueeze(-1), color, sky)
        return (color.clamp(0, 1) * 255).to(torch.uint8).view(E, self.H, self.W, 3)

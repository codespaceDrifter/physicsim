# Physics object with linear dynamics only (no rotation).
import torch
from mesh import Mesh
import config


class Obj:
    def __init__(self, obj_id: str, color: str, mesh: Mesh, density: float,
                 position: torch.Tensor, velocity: torch.Tensor):
        self.id = obj_id
        self.color = color
        self.mesh = mesh

        # Linear dynamics - all (3,) tensors
        self.position = position
        self.velocity = velocity
        self.acceleration = torch.zeros(3, device=config.device)
        self.force = torch.zeros(3, device=config.device)

        self.density = density
        self.mass = mesh.volume * density
        self.restitution = 1.0

    def update(self, dt: float):
        # semi-implicit Euler
        self.force = torch.where(torch.abs(self.force) < 1e-6,
                                 torch.zeros_like(self.force), self.force)
        # (3,)
        self.acceleration = self.force / self.mass
        # (3,)
        self.velocity = self.velocity + self.acceleration * dt
        # (3,)
        self.position = self.position + self.velocity * dt

        self.force = torch.zeros_like(self.force)

    def step_back(self, dt: float):
        self.position = self.position - self.velocity * 1.3 * dt

    def get_world_vertices(self):
        # (N, 3) body-space vertices translated to world
        return self.mesh.vertices + self.position

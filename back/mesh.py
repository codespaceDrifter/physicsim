"""
Mesh with pure tensor operations.

auto centers the mass assuming uniform density
"""
import torch


class Mesh:
    def __init__(self, vertices: torch.Tensor, indices: torch.Tensor):
        """
        Args:
            vertices: (num_verts, 3)
            indices: (num_triangles, 3) -  indices of vertices
        """

        # Compute volume-weighted center of mass (correct for uniform density).
        # vertex mean is WRONG - biased toward regions with more vertices.
        # Instead, decompose into tetrahedra (each triangle + origin) and weight
        # each tetrahedron's centroid by its signed volume.
        tris = vertices[indices]  # (num_tris, 3, 3)
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        # signed volume of each tetrahedron (origin as 4th vertex)
        # cross product produces vector with magnitude the area of the base of tetrahedron |b x c| = |b| * |c| * sin(θ) and direction orthogonal to the base
        # (a * cross).sum(dim=1) is a dot product. which is equal to |cross| * |a| * cos(θ). so it multiples the magnitude of cross (the area of the base) with the a projected onto the orthogonal vector to the base (height component) to get a parallelepiped that contains the tetrahedron
        # since the pyramid is 1/3 the volume of the parallelepiped and the tetrahedron is 1/2 the volume or the pyramid, the tetrahedron is 1/6 the volume of the parallelepiped
        # (num_tris,)
        tet_volumes = (a * torch.cross(b, c, dim=1)).sum(dim=1) / 6.0
        # centroid of each tetrahedron = mean of 4 vertices (origin + a + b + c) / 4
        # (num_tris, 3)
        tet_centroids = (a + b + c) / 4.0
        # volume-weighted average
        total_volume = tet_volumes.sum()
        # (3,)
        centroid = (tet_volumes.unsqueeze(1) * tet_centroids).sum(dim=0) / total_volume
        vertices = vertices - centroid

        self.vertices = vertices
        self.indices = indices

        # Compute AABB bounding box for fast collision checking
        # min takes minimum of each coordinate across all examples. returns value, indices. [0] grabs values.
        self.min_aabb = vertices.min(dim=0)[0]  # (3,)
        self.max_aabb = vertices.max(dim=0)[0]  # (3,)

        # Precompute triangles
        # row selection. (num_triangles, 3 (vertices), 3 (xyz))
        self.triangles = vertices[indices]

        # Volume = |sum of signed tetrahedron volumes|. signed (not per-tet abs) so tets
        # whose origin lies outside the mesh (hollow / non-convex shapes) cancel correctly.
        # translating the mesh doesn't change the signed sum of a closed mesh, so reuse it.
        self.volume = torch.abs(total_volume)

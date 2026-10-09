"""
Physics world simulation.
"""
from typing import List
from obj import Obj
from collision import check_collision_pair
from mechanics import apply_gravity_batched, apply_collision_response


class World:
    def __init__(self, world_id: str, time_step: float):
        """
        Args:
            world_id: Identifier
            time_step: Simulation timestep (seconds)
        """
        self.id = world_id
        self.dt = time_step
        self.time = 0.0
        self.objects: List[Obj] = []

    def add_object(self, obj: Obj):
        self.objects.append(obj)

    def update(self):
        """Run one simulation step."""
        self.time += self.dt

        # 1. Find collisions (parallel collision detection)
        collisions = []
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                if check_collision_pair(self.objects[i], self.objects[j]):
                    collisions.append((i, j))

        # 2. Resolve collisions
        for i, j in collisions:
            self.objects[i].step_back(self.dt)
            self.objects[j].step_back(self.dt)
            apply_collision_response(self.objects[i], self.objects[j])

        # 3. Apply forces (BATCHED gravitational computation)
        apply_gravity_batched(self.objects)

        # 4. Update all objects
        for obj in self.objects:
            obj.update(self.dt)

    def get_frame_data(self):
        """
        Get frame data for serialization (compatible with Go backend format).

        Returns:
            dict with frame data
        """
        return {
            'cur_time': self.time,
            'id_array': [obj.id for obj in self.objects],
            'position_arrays': [obj.position.cpu().tolist() for obj in self.objects],
            'vertex_arrays': [obj.mesh.vertices.cpu().reshape(-1).tolist() for obj in self.objects],
            'index_arrays': [obj.mesh.indices.cpu().reshape(-1).tolist() for obj in self.objects],
            'color_array': [obj.color for obj in self.objects],
        }
# Example scenes as scene JSON (see documentation.md for the format).
import math
from robot.pick_place import scene as pick_place_scene


def body(id, shape, position, density, color, velocity=(0.0, 0.0, 0.0), static=False, friction=0.5, restitution=0.0, collide=True):
    return {'id': id, 'shape': shape, 'position': list(position), 'rotation': [1.0, 0.0, 0.0, 0.0], 'velocity': list(velocity),
            'angular_velocity': [0.0, 0.0, 0.0], 'density': density, 'static': static, 'collide': collide, 'group': 0,
            'friction': friction, 'restitution': restitution, 'color': color}


def three_body():
    # Alpha Centauri A, B, Proxima and the planet Trisolaris, pulling on each other with real G. no floor gravity.
    bodies = [
        body('aCenA', {'type': 'sphere', 'radius': 40.0}, (0, 0, 0), 5e8, '#ffe599', (0, 0, -5), restitution=0.8),
        body('aCenB', {'type': 'sphere', 'radius': 25.0}, (300, 0, 0), 3e8, '#ffa64d', (-3, 0, 0), restitution=0.8),
        body('proxima', {'type': 'sphere', 'radius': 30.0}, (-300, 0, 0), 6e8, '#ff4d4d', (0, 0, 2), restitution=0.8),
        body('trisolaris', {'type': 'sphere', 'radius': 5.0}, (350, 0, 0), 5500, '#3fa9c0', (1, 0, 0), restitution=0.8),
    ]
    return {'name': 'three_body', 'settings': {'dt': 0.01, 'substeps': 2, 'gravity': [0.0, 0.0, 0.0], 'G': 6.674e-11},
            'bodies': bodies, 'joints': [], 'cameras': []}


def billiards():
    # 9 ft pool table (2.54 x 1.27 m cloth) with cushions, 57 mm balls. rows of resting balls hit by moving ones.
    radius = 0.028575
    # 170 g ball
    density = 0.17 / (4 / 3 * math.pi * radius ** 3)
    length, width, rail, rail_h = 2.54, 1.27, 0.05, 0.04
    bodies = [body('cloth', {'type': 'plane'}, (0, 0, 0), 1.0, '#1f6b3a', static=True, friction=0.2)]
    for name, size, pos in [('rail_left', (rail, rail_h, width + 2 * rail), (-(length + rail) / 2, rail_h / 2, 0)),
                            ('rail_right', (rail, rail_h, width + 2 * rail), ((length + rail) / 2, rail_h / 2, 0)),
                            ('rail_top', (length, rail_h, rail), (0, rail_h / 2, -(width + rail) / 2)),
                            ('rail_bottom', (length, rail_h, rail), (0, rail_h / 2, (width + rail) / 2))]:
        bodies.append(body(name, {'type': 'box', 'size': list(size)}, pos, 1.0, '#5a3a1e', static=True, friction=0.2, restitution=0.8))
    for row in range(3):
        moving = row + 1
        z = -0.25 + row * 0.25
        for i in range(5):
            x = -0.8 + i * 2 * radius
            is_moving = i >= 5 - moving
            bodies.append(body(f'row{row}_{"mov" if is_moving else "stat"}{i}', {'type': 'sphere', 'radius': radius}, (x, radius, z), density,
                               '#ffff00' if is_moving else '#ffffff', (-2.0, 0, 0) if is_moving else (0, 0, 0), friction=0.05, restitution=0.95))
    return {'name': 'billiards', 'settings': {'dt': 1 / 120, 'substeps': 4, 'gravity': [0.0, -9.81, 0.0], 'G': 0.0},
            'bodies': bodies, 'joints': [], 'cameras': []}


def tofubot_pick():
    return pick_place_scene(3)

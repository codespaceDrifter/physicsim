"""
Billard ball simulation (matches Go version exactly).
"""
import torch
import config

from world import World
from obj import Obj
from shapes import sphere, hollow_rect_frame
from recording import Recording


def simulate_billard_ball():
    print(f"Device: {config.device}")

    # Create world
    world = World('billardBall', time_step=0.01)

    # Table frame
    table = Obj(
        'table', '#654321',
        hollow_rect_frame(280.0, 140.0, 5.0, 10.0),
        1000,
        torch.tensor([0., 10.0 / 2, 0.], device=config.device),
        torch.tensor([0., 0., 0.], device=config.device),
        static=True
    )
    world.add_object(table)

    # Ball parameters
    radius = 5.7
    spacing = 0.0
    ball_density = 170
    balls_per_row = 5
    row_count = 3
    row_separation = 20.0
    start_x = -70.0
    start_z = -20.0
    move_speed = 200.0

    for row in range(row_count):
        moving_balls = row + 1
        stationary_balls = balls_per_row - moving_balls
        z = start_z + row * row_separation

        # Place stationary balls
        for i in range(stationary_balls):
            x = start_x + i * (radius * 2 + spacing)
            ball = Obj(
                f'row{row}_stat{i}', '#ffffff',
                sphere(radius, 0),
                ball_density,
                torch.tensor([x, radius, z], device=config.device),
                torch.tensor([0., 0., 0.], device=config.device)
            )
            world.add_object(ball)

        # Place moving balls
        for i in range(moving_balls):
            x = start_x + (stationary_balls + i) * (radius * 2 + spacing)
            ball = Obj(
                f'row{row}_mov{i}', '#ffff00',
                sphere(radius, 0),
                ball_density,
                torch.tensor([x, radius, z], device=config.device),
                torch.tensor([-move_speed, 0., 0.], device=config.device)
            )
            world.add_object(ball)

    print(f"World has {len(world.objects)} objects")

    # Create and run recording (matches Go: 0.01 seconds per frame, 10 seconds total)
    recording = Recording('billardBall', world, seconds_per_frame=0.01, end_seconds=10)
    recording.simulate()


if __name__ == '__main__':
    simulate_billard_ball()

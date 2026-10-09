# Write the example scenes to scenes/ (editable in the frontend) and record the passive ones into logs/.
# run from back/:  python -m simulate
import json
import shutil
from pathlib import Path
import torch
from example_scenes import three_body, billiards, tofubot_pick
from physics.world import World
from recording import LOGS_DIR, record_world

SCENES_DIR = Path(__file__).parent / 'scenes'
# {name: (scene, seconds per recorded frame, recorded seconds)}, seconds None = scene file only
EXAMPLES = {'three_body': (three_body(), 0.5, 3000), 'billiards': (billiards(), 1 / 60, 10), 'tofubot_pick': (tofubot_pick(), None, None)}


def main():
    SCENES_DIR.mkdir(exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for name, (scene, seconds_per_frame, end_seconds) in EXAMPLES.items():
        with open(SCENES_DIR / f'{name}.json', 'w') as f:
            json.dump(scene, f, indent=1)
        print(f'wrote scenes/{name}.json')
        if seconds_per_frame is None:
            continue
        if (LOGS_DIR / name).exists():
            shutil.rmtree(LOGS_DIR / name)
        world = World(scene, 1, device)
        if device == 'cuda':
            world.enable_cuda_graph()
            world.reset(torch.ones(1, dtype=torch.bool, device=device))
            world.time = 0.0
        record_world(name, world, seconds_per_frame, end_seconds)


if __name__ == '__main__':
    main()

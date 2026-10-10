# Write the example scenes to scenes/ for the editor. run from back/:  python -m write_scenes
import json
from pathlib import Path
from robot.pick_place import scene as pick_scene
from robot.block_race import scene as race_scene

SCENES_DIR = Path(__file__).parent / 'scenes'

if __name__ == '__main__':
    SCENES_DIR.mkdir(exist_ok=True)
    for name, sc in (('lekiwi_pick', pick_scene(3)), ('block_race', race_scene())):
        with open(SCENES_DIR / f'{name}.json', 'w') as f:
            json.dump(sc, f, indent=1)
        print(f'wrote scenes/{name}.json')

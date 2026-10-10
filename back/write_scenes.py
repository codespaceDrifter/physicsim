# Write the example scene(s) to scenes/ for the editor. run from back/:  python -m write_scenes
import json
from pathlib import Path
from robot.pick_place import scene

SCENES_DIR = Path(__file__).parent / 'scenes'

if __name__ == '__main__':
    SCENES_DIR.mkdir(exist_ok=True)
    with open(SCENES_DIR / 'lekiwi_pick.json', 'w') as f:
        json.dump(scene(3), f, indent=1)
    print('wrote scenes/lekiwi_pick.json')

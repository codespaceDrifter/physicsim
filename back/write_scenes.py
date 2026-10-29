# Write the course scenes to scenes/ for the editor: the LeKiwi course (one per stage, numbered in course order:
# course_1_drive ...) and the humanoid warehouse course (one scene for all its stages). run from back/:  python -m write_scenes
import json
from pathlib import Path
from robot.course import scene, STAGES
from robot import warehouse

SCENES_DIR = Path(__file__).parent / 'scenes'

if __name__ == '__main__':
    SCENES_DIR.mkdir(exist_ok=True)
    for stage in STAGES:
        sc = scene(stage)
        with open(SCENES_DIR / f'{sc["name"]}.json', 'w') as f:
            json.dump(sc, f, indent=1)
        print(f'wrote scenes/{sc["name"]}.json')
    sc = warehouse.scene()
    with open(SCENES_DIR / f'{sc["name"]}.json', 'w') as f:
        json.dump(sc, f, indent=1)
    print(f'wrote scenes/{sc["name"]}.json')

# Write the course scenes (one per stage, numbered in course order: course_1_drive ...) to scenes/ for the editor. run from back/:  python -m write_scenes
import json
from pathlib import Path
from robot.course import scene, STAGES

SCENES_DIR = Path(__file__).parent / 'scenes'

if __name__ == '__main__':
    SCENES_DIR.mkdir(exist_ok=True)
    for stage in STAGES:
        sc = scene(stage)
        with open(SCENES_DIR / f'{sc["name"]}.json', 'w') as f:
            json.dump(sc, f, indent=1)
        print(f'wrote scenes/{sc["name"]}.json')

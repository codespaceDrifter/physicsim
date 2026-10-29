# Write HumanoidBench's tasks (github.com/carlosferrazza/humanoid-bench, its G1) as scenes the frontend opens like ours:
# scenes/hb_{task}.json, the task's scene at its start state (robot/mjcf.py geom_scene: every visible geom a static
# body), the meshes as binary STLs under robot/assets/mujoco/ (served by the API's /meshes).
#
# run from back/ with the HumanoidBench environment (~/hbenv):  python -m write_hb_scenes --tasks walk stair
import argparse
import json
from pathlib import Path
import gymnasium as gym
import humanoid_bench
from humanoid_bench.env import TASKS
from recording import ASSETS_DIR, save_stl
from robot.mjcf import geom_scene

SCENES_DIR = Path(__file__).parent / 'scenes'
# tasks with a G1 scene (humanoid_bench/assets/envs/g1_torque_{task}.xml)
ENVS_DIR = Path(humanoid_bench.__file__).parent / 'assets' / 'envs'
G1_TASKS = [t for t in TASKS if (ENVS_DIR / f'g1_torque_{t}.xml').exists()]

parser = argparse.ArgumentParser()
parser.add_argument('--tasks', nargs='+', default=G1_TASKS, choices=G1_TASKS)
args = parser.parse_args()

for task in args.tasks:
    env = gym.make(f'g1-{task}-v0', render_mode=None)
    env.reset(seed=0)
    scene, meshes, _ = geom_scene(env.unwrapped.model, env.unwrapped.data, f'hb_{task}')
    for name, (vertices, triangles) in meshes.items():
        if not (ASSETS_DIR / name).exists():
            save_stl(ASSETS_DIR / name, vertices, triangles)
    (SCENES_DIR / f'hb_{task}.json').write_text(json.dumps(scene, indent=1))
    print(f'hb_{task}: {len(scene["bodies"])} bodies, {len(meshes)} meshes')
    env.close()

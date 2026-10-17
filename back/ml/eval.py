# The results table: every model on every stage. rows: the scripted expert, the teacher, then each VLA checkpoint.
# success = fraction of robot-episodes that succeeded (race: won against the teacher driving robot 1).
#
# run from back/:  python -m ml.eval --vla checkpoints/vla_sft.pt checkpoints/vla_grpo.pt
# writes logs/eval.json {model name: {stage: {"success", "score", "opponent_score"}}}
import argparse
import json
import os
import torch
from robot.course import STAGES
from ml.teacher import Teacher
from ml.vla import VLA
from ml.rollout import TeacherDriver, VLADriver, ExpertDriver, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--teacher', default='checkpoints/teacher.pt')
parser.add_argument('--vla', nargs='*', default=[], help='VLA checkpoints, one row each')
parser.add_argument('--envs', type=int, default=256, help='episodes per stage')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--out', default='logs/eval.json')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = Teacher().to(device)
teacher.load_state_dict(torch.load(args.teacher, map_location=device)['model'])
teacher.eval()
drivers = {'expert': ExpertDriver(), 'teacher': TeacherDriver(teacher)}  # {name: driver}
for path in args.vla:
    vla = VLA().to(device)
    vla.load_state_dict(torch.load(path, map_location=device)['model'])
    vla.eval()
    drivers[os.path.basename(path).removesuffix('.pt')] = VLADriver(vla)
results = {}  # {name: {stage: metrics}}
for name, driver in drivers.items():
    results[name] = {s: evaluate(driver, TeacherDriver(teacher), s, args.envs, device, args.seed) for s in STAGES}
    print(f'{name}: ' + '  '.join(f'{s} {m["success"]:.2f}' for s, m in results[name].items()))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(results, open(args.out, 'w'), indent=1)
print(f'\n| model | {" | ".join(STAGES)} |\n|---|{"---|" * len(STAGES)}')
for name, res in results.items():
    print(f'| {name} | ' + ' | '.join(f'{res[s]["success"]:.2f}' for s in STAGES) + ' |')

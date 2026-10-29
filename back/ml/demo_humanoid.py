# The two-system demo: a typed request -> System 2 (ml/planner.py: Qwen3.5-4B writes skill commands) -> System 1 (the
# trained body policy, ml/train_humanoid.py) runs the commands one after another in one continuous scene (boxes and
# pads laid out like the sort stage, robot/warehouse.py), each until its skill's pass test succeeds (a walk line: for
# its seconds) or MAX_SECONDS go by; a fall ends the demo. recorded for the frontend's playback mode.
#
# run from back/:  python -m ml.demo_humanoid --ckpt checkpoints/warehouse_v7.pt --request "pick up the red box and put it on the blue pad"
#                  (--plan "line" "line" ... skips System 2 and runs the given lines)
# writes logs/{name}.simrec (recording.py format), info {"request", "plan": [line], "captions": [[start s, line]],
#                                                        "outcome": [str] per line}
import argparse
import re
import torch
from recording import Recorder
from robot.warehouse import WarehouseCourse, STAGES, PROPRIO_DIM, ACTION_DIM, AUX_DIM, VOCAB, BOXES
from robot.humanoid import DT, TOKENS
from ml.humanoid_policy import Policy, HISTORY, load_matching
from ml.planner import Planner

# longest a skill line may run (s)
MAX_SECONDS = 60.0
COLORS = list(BOXES)
# plan line -> (stage, box color, pad color) for the box / track skills (walk lines are parsed separately)
PATTERNS = [(re.compile(r'pick up the (\w+) box'), 'pick'), (re.compile(r'put the (\w+) box on the (\w+) pad'), 'deliver'),
            (re.compile(r'run a lap with the (\w+) box'), 'carry'), (re.compile(r'run a lap'), 'track'),
            (re.compile(r'sort the boxes by color'), 'sort')]
WALK = re.compile(r'walk forward (\S+) left (\S+) turn (\S+) for (\d+) seconds')

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/warehouse_v7.pt')
parser.add_argument('--request', default='')
parser.add_argument('--plan', nargs='*', default=None, help='plan lines (skips System 2)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', default='demo')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plan = args.plan if args.plan is not None else Planner(torch.device(device)).plan(args.request)
print('plan:', plan)
ck = torch.load(args.ckpt, map_location=device)
assert ck['args']['backend'] == 'physicsim'
policy = Policy(PROPRIO_DIM, ACTION_DIM, AUX_DIM, TOKENS, len(VOCAB), True, 0, ck['args']['width'], ck['args']['layers'], ck['args']['group']).to(device)
with torch.no_grad():
    fresh = load_matching(policy, ck['policy'])
assert not fresh, f'checkpoint lacks {fresh}'
policy.eval()
env = WarehouseCourse(1, device, [float(s == 'sort') for s in STAGES], vision=True, seed=args.seed)
env.pushes = False
env.auto_reset = False
env.test[:] = True
env.ep_steps[:] = 10 ** 9
rec = Recorder(args.name, env.world.scene, DT)
rec.add(0.0, env.world.poses()[0].cpu().numpy())
obs = env.observe(noise=False)
# (R, HISTORY, P) the robot's proprio history, continuous across the plan's lines
history = obs['proprio'][0][:, None].repeat(1, HISTORY, 1)
step = 0
captions, outcome = [], []  # [[start s, line]], [str]
with torch.no_grad():
    for line in plan:
        walk = WALK.fullmatch(line)
        if walk:
            env.set_task('walk')
            command = torch.tensor([float(v) for v in walk.groups()[:3]], device=device)
            steps = int(int(walk.group(4)) / DT)
        else:
            stage, colors = next((st, m.groups()) for p, st in PATTERNS if (m := p.fullmatch(line)))
            env.set_task(stage, *[COLORS.index(c) for c in colors] + ([COLORS.index(colors[0])] if len(colors) == 1 else []))
            steps = int(MAX_SECONDS / DT)
        captions.append([step * DT, line])
        result = 'out of time' if not walk else 'done'
        for _ in range(steps):
            if walk:
                env.command[:] = command
            mean, _ = policy(history, obs['command'][0], obs['image'][0])
            _, done, _, info = env.step(mean.view(1, env.R, -1))
            step += 1
            rec.add(step * DT, env.world.poses()[0].cpu().numpy())
            obs = env.observe(noise=False)
            history = torch.cat([history[:, 1:], obs['proprio'][0][:, None]], 1)
            if env.fallen().any():
                result = 'fell'
                break
            if not walk and info['metrics'][f'passed_{stage}'] > 0:
                result = f'done in {step * DT - captions[-1][0]:.1f} s'
                break
        outcome.append(result)
        print(f'{line}: {result}')
        if result == 'fell':
            break
rec.save({'request': args.request, 'plan': plan, 'captions': captions, 'outcome': outcome})

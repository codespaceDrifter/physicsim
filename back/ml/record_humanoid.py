# Record the humanoid policy (ml/train_humanoid.py checkpoint) for the frontend's playback mode: one env, mean
# actions, no pushes.
#   walk         a fixed command sequence (SCRIPT); prints how well it tracked each command. a fall stops it
#   other stages --seconds of the stage, episode after episode (every one judged by the stage's pass test, see
#                robot/warehouse.py; a finished or failed one is followed by a fresh layout); prints passes / episodes
#
# run from back/:  python -m ml.record_humanoid --ckpt checkpoints/warehouse_v7.pt --stage pick --name warehouse_pick
# writes logs/{name}.simrec (recording.py format), info {"stage", "checkpoint", "summary": the printed result lines}
import argparse
import torch
from recording import Recorder
from robot.warehouse import WarehouseCourse, STAGES, PROPRIO_DIM, ACTION_DIM, AUX_DIM, VOCAB
from robot.humanoid import DT, TOKENS, SCRIPT
from ml.humanoid_policy import Policy, HISTORY, load_matching


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/humanoid.pt')
parser.add_argument('--stage', default='walk', choices=STAGES)
parser.add_argument('--seconds', type=float, default=30.0, help='length of a box stage recording')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', default='humanoid_walk')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(args.ckpt, map_location=device)
assert ck['args']['backend'] == 'physicsim'
policy = Policy(PROPRIO_DIM, ACTION_DIM, AUX_DIM, TOKENS, len(VOCAB), True, 0, ck['args']['width'], ck['args']['layers'], ck['args']['group']).to(device)
with torch.no_grad():
    fresh = load_matching(policy, ck['policy'])
assert not fresh, f'checkpoint lacks {fresh}'
policy.eval()
walk = args.stage == 'walk'
env = WarehouseCourse(1, device, [float(s == args.stage) for s in STAGES], vision=not walk, seed=args.seed)
env.test[:] = True
env.pushes = False
total = int((sum(s for s, c in SCRIPT) if walk else args.seconds) / DT)
# no time limit inside a walk recording; the other stages keep their episode length (an episode that runs out counts as
# a failed try, and the next one starts)
if walk:
    env.stage_steps[:] = total + 1
    env.ep_steps[:] = total + 1
rec = Recorder(args.name, env.world.scene, DT)
rec.add(0.0, env.world.poses()[0].cpu().numpy())
R = env.R
obs = env.observe(noise=False)
# (R, HISTORY, P) each robot's proprio history
history = obs['proprio'][0][:, None].repeat(1, HISTORY, 1)
# walk: [(steps, command)]; the box stages: one stretch without a command override
plan = [(int(s / DT), c) for s, c in SCRIPT] if walk else [(total, None)]
step = 0
# {metric: sum over the recording} the course's per-step counts (successes, boxes delivered / lost), episodes ended
counts = {}  # {str: float}
episodes = 0
summary = []  # [str] result lines, printed and saved in the recording's info
with torch.no_grad():
    for steps, command in plan:
        err = torch.zeros(3, device=device)
        for _ in range(steps):
            if command is not None:
                env.command[:] = torch.tensor(command, device=device)
            # every episode is a test (its end counts a try of the stage's pass test)
            env.test[:] = True
            mean, _ = policy(history, obs['command'][0], None if walk else obs['image'][0])
            _, done, _, info = env.step(mean.view(1, R, -1))
            assert not (walk and done.any()), f'fell at {step * DT:.2f} s'
            step += 1
            episodes += int(done.sum())
            for k, v in info['metrics'].items():
                counts[k] = counts.get(k, 0.0) + float(v)
            if walk:
                err += (env.base_velocity()[0, 0] - env.command[0, 0]).abs() / steps
            rec.add(step * DT, env.world.poses()[0].cpu().numpy())
            obs = env.observe(noise=False)
            history = torch.where(done[0], obs['proprio'][0][:, None], torch.cat([history[:, 1:], obs['proprio'][0][:, None]], 1))
        if walk:
            summary.append(f'command {command}: mean |error| forward {err[0]:.2f} m/s, left {err[1]:.2f} m/s, turn {err[2]:.2f} rad/s')
if not walk:
    summary.append(f'{args.seconds:.0f} s of {args.stage}: {counts[f"passed_{args.stage}"]:.0f} passed of {counts[f"tried_{args.stage}"]:.0f} '
                   f'tries ({episodes} episodes)')
print('\n'.join(summary))
rec.save({'stage': args.stage, 'checkpoint': args.ckpt, 'summary': summary})

# Bootstrap the teacher from the scripted expert (demo-augmented RL): collect (state, expert action) pairs on the
# block stages, behavior-clone the teacher's actor on them, then PPO (ml/train_teacher.py --init) fine-tunes on the
# whole course. random exploration almost never finds a grasp, the expert already succeeds ~97% of the time; PPO
# learns the drive stage (the expert only fetches blocks) and improves on the expert everywhere. --drive-teacher: a
# teacher checkpoint that already drives (e.g. from a PPO run from scratch): its mean actions on drive-stage states
# join the demos, so the clone starts out driving too. --dagger-rounds: then DAgger with the expert: the clone drives,
# the expert labels every state it reaches (it has no memory, so the label is what it would do there), the clone
# retrains on everything. cloning alone compounds its small errors.
#
# run from back/:  python -m ml.bc_teacher
# writes checkpoints/teacher_bc.pt {"model": Teacher state_dict, "loss": float, "round": DAgger rounds done} (after the
#                                 plain clone and after every DAgger round), and a copy per round:
#                                 checkpoints/teacher_bc_r{round}.pt (rounds vary; keep the best)
#        data/expert_demos.pt     {"state": (M, STATE_DIM) float32, "action": (M, ACTION_DIM) float32} every active
#                                 robot's state and the expert's action, PPO keeps cloning it at a decaying weight
#                                 (--reuse loads it instead of collecting again)
#        data/expert_demos_drive.pt the same plus the --drive-teacher's drive rows (only with --drive-teacher)
#        data/expert_demos_dagger.pt everything after the DAgger rounds (only with --dagger-rounds)
import argparse
import os
import time
import torch
from robot.course import CourseEnv, STAGES
from ml.teacher import Teacher
from ml.rollout import ExpertDriver, TeacherDriver

parser = argparse.ArgumentParser()
parser.add_argument('--envs', type=int, default=512)
parser.add_argument('--steps', type=int, default=2400, help='control steps of expert play to record')
parser.add_argument('--train-steps', type=int, default=80_000)
parser.add_argument('--batch', type=int, default=4096)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--reuse', action='store_true', help='load the expert demos from --demos instead of collecting')
parser.add_argument('--drive-teacher', default='', help='teacher checkpoint that labels drive-stage states')
parser.add_argument('--drive-envs', type=int, default=2048)
parser.add_argument('--drive-steps', type=int, default=300)
parser.add_argument('--dagger-rounds', type=int, default=0)
parser.add_argument('--dagger-steps', type=int, default=400, help='control steps per DAgger round')
parser.add_argument('--dagger-train-steps', type=int, default=80_000)
parser.add_argument('--dagger-envs', type=int, default=512)
parser.add_argument('--dagger-probs', default='0,1,1,1,1', help='stage sampling weights in DAgger rounds (STAGES order); '
                    'weight the speedrun and run rounds as long as its episodes so late-episode states get labeled')
parser.add_argument('--init', default='', help='fine-tune this teacher checkpoint (its input normalization kept) instead of '
                    'training a fresh clone on the demos and in every DAgger round')
parser.add_argument('--model', default='', help='start the DAgger rounds from this clone instead of training one (with --reuse of its demos)')
parser.add_argument('--demos', default='data/expert_demos.pt')
parser.add_argument('--out', default='checkpoints/teacher_bc.pt')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
for d in (os.path.dirname(args.out), os.path.dirname(args.demos)):
    os.makedirs(d, exist_ok=True)


@torch.no_grad()
def record(env, driver, steps, labeler=None):
    # run driver for steps control steps: (n, STATE_DIM) states of active robots, (n, ACTION_DIM) labels (the
    # labeler's actions, default the driver's own)
    states, actions = [], []  # [(n, STATE_DIM)], [(n, ACTION_DIM)]
    fresh = torch.ones(env.E, dtype=torch.bool, device=device)
    t0 = time.time()
    for t in range(steps):
        state = env.state()
        action = driver(env, fresh)
        label = action if labeler is None else labeler(env, fresh)
        active = env.active()
        states.append(state[active])
        actions.append(label[active])
        _, done, info = env.step(action)
        fresh = done
        if (t + 1) % 200 == 0:
            print(f'step {t + 1}/{steps} ({time.time() - t0:.0f}s)', flush=True)
    return torch.cat(states), torch.cat(actions)


if args.reuse:
    demos = torch.load(args.demos)
    state, action = demos['state'].to(device), demos['action'].to(device)
else:
    # every stage the expert can do, equally often
    env = CourseEnv(args.envs, device, stage_probs=[float(s != 'drive') for s in STAGES], seed=args.seed)
    if device == 'cuda':
        env.enable_cuda_graph()
    state, action = record(env, ExpertDriver(), args.steps)
    del env
    torch.save({'state': state.cpu(), 'action': action.cpu()}, args.demos)
print(f'{len(state)} expert demo rows')
if args.drive_teacher:
    driver_model = Teacher().to(device)
    driver_model.load_state_dict(torch.load(args.drive_teacher, map_location=device)['model'])
    driver_model.eval()
    env = CourseEnv(args.drive_envs, device, stage_probs=[float(s == 'drive') for s in STAGES], seed=args.seed + 1)
    if device == 'cuda':
        env.enable_cuda_graph()
    d_state, d_action = record(env, TeacherDriver(driver_model), args.drive_steps)
    del env
    state, action = torch.cat([state, d_state]), torch.cat([action, d_action])
    print(f'+ {len(d_state)} drive rows from {args.drive_teacher}')
    torch.save({'state': state.cpu(), 'action': action.cpu()}, args.demos.replace('.pt', '_drive.pt'))
def train(steps):
    # behavior cloning on all (state, action) rows so far, linear lr decay. every DAgger round trains a fresh teacher:
    # briefly fine-tuning the last clone on the grown data (lr restarted) left it fitting even its old rows twice as
    # badly, and it drove worse every round. --init: fine-tune the current model instead (a changed expert's labels
    # taught to an already trained teacher)
    global model, opt
    if not args.init:
        model = Teacher().to(device)
        model.update_norm(state)
    model.train()
    opt = torch.optim.Adam(model.actor.parameters(), lr=args.lr)
    M = len(state)
    for step in range(steps):
        for g in opt.param_groups:
            g['lr'] = args.lr * (1 - step / steps)
        idx = torch.randint(0, M, (args.batch,), device=device)
        mean, _ = model(state[idx])
        loss = ((mean - action[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f'bc step {step + 1} loss {loss.item():.4f}', flush=True)
    return loss.item()


if args.model:
    model = Teacher().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    loss = float('nan')
else:
    if args.init:
        model = Teacher().to(device)
        model.load_state_dict(torch.load(args.init, map_location=device)['model'])
    loss = train(args.train_steps)
torch.save({'model': model.state_dict(), 'loss': loss, 'round': 0}, args.out)
torch.save({'model': model.state_dict(), 'loss': loss, 'round': 0}, args.out.replace('.pt', '_r0.pt'))
for round_ in range(args.dagger_rounds):
    model.eval()
    env = CourseEnv(args.dagger_envs, device, stage_probs=[float(p) for p in args.dagger_probs.split(',')], seed=args.seed + 10 + round_)
    if device == 'cuda':
        env.enable_cuda_graph()
    r_state, r_action = record(env, TeacherDriver(model), args.dagger_steps, ExpertDriver())
    del env
    state, action = torch.cat([state, r_state]), torch.cat([action, r_action])
    print(f'dagger round {round_ + 1}: + {len(r_state)} rows, {len(state)} total', flush=True)
    loss = train(args.dagger_train_steps)
    torch.save({'model': model.state_dict(), 'loss': loss, 'round': round_ + 1}, args.out)
    torch.save({'model': model.state_dict(), 'loss': loss, 'round': round_ + 1}, args.out.replace('.pt', f'_r{round_ + 1}.pt'))
    torch.save({'state': state.cpu(), 'action': action.cpu()}, args.demos.replace('.pt', '_dagger.pt'))
torch.save({'model': model.state_dict(), 'loss': loss}, args.out)
print(f'saved {args.out}')

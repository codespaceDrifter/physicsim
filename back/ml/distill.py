# Distill the teacher (state -> actions) into the VLA (cameras + command + proprio -> action chunks).
#   round 0 (behavior cloning): the teacher drives (its mean action + a little noise, so the data also covers small
#           mistakes), every frame is labeled with the teacher's mean action
#   rounds 1..R (DAgger): the VLA drives, the teacher labels every state the VLA reached; the data accumulates
# after each round the VLA trains on everything collected so far (L1 on the next CHUNK teacher actions), then is
# evaluated per stage. in the race robot 1 is always the teacher (the opponent). one collection env persists across
# rounds, so long episodes (race, speedrun) are covered past their start.
#
# --resume DIR: continue a finished run: the VLA from --out, then --rounds more DAgger rounds (no behavior cloning
# round). DIR's rows (and its parents') are read in place; the new rows go to --data, which records DIR as its parent.
# --fresh-steps N (with --resume): start from a new VLA instead and train it N steps on the resumed rows first.
#
# run from back/:  python -m ml.distill --teacher checkpoints/teacher.pt
# writes checkpoints/vla_sft.pt {"model": state_dict, "round": int, "metrics": {stage: {...}}}
#        logs/distill.jsonl     {"round", "rows", "train_loss", "metrics": {stage: evaluate() dict}} per round, and
#                               {"round", "step", "loss", "lr"} every --log-every steps
#        data/distill/          one row per active robot per control step, preallocated memmaps (rows of this run
#                               only; global row r lives in the directory whose [first, rows) holds it):
#          rgb.npy (CAP, 6, 96, 96) uint8      front RGB then wrist RGB, what the robot saw before acting
#          words.npy (CAP, 16) int16           command word ids (ml/vla.py VOCAB), 0 = padding
#          proprio.npy (CAP, 9) float32        joint readings
#          action.npy (CAP, 9) float32         the teacher's mean action in this state (the label)
#          next.npy (rows,) int64              global row of the same robot's next step in this episode, -1 at its
#                                              end, for all rows incl. the parents' (rows of different robots
#                                              interleave; chunk targets follow this)
#          meta.json {"first": int, "rows": int, "parent": str | null}  first global row stored here, global rows
#                                              filled so far, the resumed run's directory
import argparse
import json
import math
import os
import time
import numpy as np
import torch
from robot.course import CourseEnv, STAGES
from ml.teacher import Teacher
from ml.vla import VLA, CHUNK, MAX_WORDS, tokenize
from ml.rollout import VLADriver, TeacherDriver, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--teacher', default='checkpoints/teacher.pt')
parser.add_argument('--envs', type=int, default=256)
parser.add_argument('--stage-probs', default='1,1,1,1,1', help='stage sampling weights of the collection env (STAGES order)')
parser.add_argument('--rounds', type=int, default=5, help='DAgger rounds after the behavior cloning round')
parser.add_argument('--bc-rows', type=int, default=800_000)
parser.add_argument('--dagger-rows', type=int, default=200_000, help='rows added per DAgger round')
parser.add_argument('--bc-steps', type=int, default=30_000, help='training steps after round 0')
parser.add_argument('--dagger-steps', type=int, default=10_000, help='training steps after each DAgger round')
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup', type=int, default=1000)
parser.add_argument('--weight-decay', type=float, default=0.05)
parser.add_argument('--noise', type=float, default=0.1, help='std of the noise on the teacher\'s actions in round 0')
parser.add_argument('--eval-envs', type=int, default=128)
parser.add_argument('--log-every', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data', default='data/distill')
parser.add_argument('--resume', default='', help='data dir of a finished run to continue from (its VLA: --out)')
parser.add_argument('--fresh-steps', type=int, default=0, help='with --resume: train a new VLA this many steps on the resumed rows first')
parser.add_argument('--out', default='checkpoints/vla_sft.pt')
parser.add_argument('--log', default='logs/distill.jsonl')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
teacher = Teacher().to(device)
teacher.load_state_dict(torch.load(args.teacher, map_location=device)['model'])
teacher.eval()
vla = VLA().to(device)
# round numbers continue after a resumed run's last
first_round = 0
assert not args.fresh_steps or args.resume
if args.resume:
    ck = torch.load(args.out, map_location=device)
    if not args.fresh_steps:
        vla.load_state_dict(ck['model'])
    first_round = ck['round'] + 1
# the GRPO-only log_std is not trained here
params = [p for n, p in vla.named_parameters() if n != 'log_std']  # [nn.Parameter]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
for d in (args.data, os.path.dirname(args.out), os.path.dirname(args.log)):
    os.makedirs(d, exist_ok=True)
log = open(args.log, 'a')

env = CourseEnv(args.envs, device, stage_probs=[float(p) for p in args.stage_probs.split(',')], seed=args.seed)
if device == 'cuda':
    env.enable_cuda_graph()
E, R = env.E, env.R
FIELDS = ('rgb', 'words', 'proprio', 'action')
# [(first global row, end row, {field: memmap})] oldest first: the resumed run's directories (read-only), then this run's
shards = []
d = args.resume
while d:
    meta = json.load(open(os.path.join(d, 'meta.json')))
    shards.insert(0, (meta['first'], meta['rows'], {f: np.load(os.path.join(d, f + '.npy'), mmap_mode='r') for f in FIELDS}))
    d = meta['parent']
# rows already collected by a resumed run
old_rows = shards[-1][1] if shards else 0
# capacity of this run's directory: its rounds' rows + one step of overshoot per round
CAP = (0 if args.resume else args.bc_rows) + args.rounds * args.dagger_rows + (args.rounds + 1) * E * R
obs0 = env.observe()
rgb_shape, P, A = obs0['rgb'].shape[2:], obs0['proprio'].shape[-1], teacher.log_std.shape[0]
mm = lambda name, dtype, shape: np.lib.format.open_memmap(os.path.join(args.data, name), mode='w+', dtype=dtype, shape=(CAP, *shape))
d_rgb, d_words, d_proprio, d_action = mm('rgb.npy', np.uint8, rgb_shape), mm('words.npy', np.int16, (MAX_WORDS,)), \
    mm('proprio.npy', np.float32, (P,)), mm('action.npy', np.float32, (A,))
shards.append((old_rows, old_rows + CAP, {'rgb': d_rgb, 'words': d_words, 'proprio': d_proprio, 'action': d_action}))
# (old_rows + CAP,) next global row of the same robot-episode, kept in RAM and saved after every round
d_next = np.full(old_rows + CAP, -1, dtype=np.int64)
rows = old_rows
if args.resume:
    d_next[:old_rows] = np.load(os.path.join(args.resume, 'next.npy'))[:old_rows]
    print(f'resumed: {old_rows} rows from {args.resume} and its parents, '
          f'VLA {"new" if args.fresh_steps else f"from {args.out}"} (round {first_round - 1})', flush=True)
# (E, R) row written for this robot at the previous step (-1: none), (E, R, MAX_WORDS) command words this episode
last_row = np.full((E, R), -1, dtype=np.int64)
words = torch.zeros(E, R, MAX_WORDS, dtype=torch.long, device=device)
# (E,) episode just started (no link to the previous row)
new_episode = torch.ones(E, dtype=torch.bool, device=device)
teacher_driver = TeacherDriver(teacher)


@torch.no_grad()
def collect(target_rows, driver, noise):
    # run the env until target_rows more rows are written. driver drives (robot 1 in the race: the teacher)
    global rows, new_episode
    end = rows + target_rows
    fresh = torch.ones(E, dtype=torch.bool, device=device)
    t0, start_rows = time.time(), rows
    while rows < end:
        if new_episode.any():
            envs = new_episode.nonzero().view(-1)
            words[envs] = tokenize([env.command(e, r) for e in envs.tolist() for r in range(R)], device).view(len(envs), R, -1)
        obs = env.observe()
        # (E, R, A) labels
        label = teacher(env.state())[0].clamp(-1, 1)
        action = label + noise * torch.randn_like(label) if driver is None else driver(env, fresh, obs)
        race = env.is_stage('race')
        action[:, 1] = torch.where(race[:, None], label[:, 1], action[:, 1])
        # write the active robots' rows
        active = env.active()
        n = int(active.sum())
        # this run's directory starts at global row old_rows
        at = rows - old_rows
        d_rgb[at:at + n] = obs['rgb'][active].cpu().numpy()
        d_words[at:at + n] = words[active].cpu().numpy()
        d_proprio[at:at + n] = obs['proprio'][active].cpu().numpy()
        d_action[at:at + n] = label[active].cpu().numpy()
        # link each robot's previous row to this one, unless a new episode started
        act_np = active.cpu().numpy()
        link = act_np & ~new_episode.cpu().numpy()[:, None] & (last_row >= 0)
        new_rows = np.full((E, R), -1, dtype=np.int64)
        new_rows[act_np] = np.arange(rows, rows + n)
        d_next[last_row[link]] = new_rows[link]
        last_row[:] = new_rows
        rows += n
        reward, done, info = env.step(action)
        new_episode = done
        fresh = done
    print(f'collected {rows - start_rows} rows in {time.time() - t0:.0f}s ({(rows - start_rows) / (time.time() - t0):.0f} rows/s)')


def gather(field, idx):
    # global rows idx (B,) numpy -> (B, ...) values of field, each row read from the directory that holds it
    out = np.empty((len(idx), *shards[-1][2][field].shape[1:]), dtype=shards[-1][2][field].dtype)
    for first, end, arrs in shards:
        sel = (idx >= first) & (idx < end)
        out[sel] = arrs[field][idx[sel] - first]
    return out


def batch(idx):
    # sorted global rows idx (B,) numpy -> rgb (B, 6, 96, 96), words (B, MAX_WORDS), proprio (B, P),
    # target (B, CHUNK, A), mask (B, CHUNK) (0 past the episode's end, where the target repeats its last action)
    targets, mask = [], []  # [np.ndarray (B, A)], [np.ndarray (B,)]
    cur, valid = idx, np.ones(len(idx), dtype=bool)
    for _ in range(CHUNK):
        targets.append(gather('action', cur))
        mask.append(valid.copy())
        nxt = d_next[cur]
        valid &= nxt >= 0
        cur = np.where(nxt >= 0, nxt, cur)
    to = lambda x, dtype=None: torch.from_numpy(np.ascontiguousarray(x)).to(device, dtype=dtype, non_blocking=True)
    return (to(gather('rgb', idx)), to(gather('words', idx), torch.long), to(gather('proprio', idx)), to(np.stack(targets, 1)),
            to(np.stack(mask, 1), torch.float32))


def train(steps, round_):
    # steps of AdamW on random rows of everything collected; linear warmup then cosine to 10% (per round)
    vla.train()
    total = 0.0
    for step in range(steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        rgb, w, proprio, target, mask = batch(np.sort(np.random.randint(0, rows, args.batch)))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            pred = vla(rgb, w, proprio)
        # masked L1 over (B, CHUNK, A)
        loss = ((pred.float() - target).abs().mean(-1) * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        total += loss.item()
        if (step + 1) % args.log_every == 0:
            print(f'round {round_} step {step + 1}/{steps} loss {total / (step + 1):.4f} lr {lr:.1e}')
            log.write(json.dumps({'round': round_, 'step': step + 1, 'loss': round(loss.item(), 5), 'lr': lr}) + '\n')
            log.flush()
    vla.eval()
    return total / steps


start = time.time()
# --fresh-steps: one extra round first that only trains the new VLA on the resumed rows
fresh = bool(args.fresh_steps)
for round_ in range(first_round, first_round + args.rounds + (0 if args.resume else 1) + fresh):
    if round_ == 0:
        collect(args.bc_rows, None, args.noise)
    elif not (fresh and round_ == first_round):
        collect(args.dagger_rows, VLADriver(vla), 0.0)
    np.save(os.path.join(args.data, 'next.npy'), d_next[:rows])
    json.dump({'first': old_rows, 'rows': rows, 'parent': args.resume or None}, open(os.path.join(args.data, 'meta.json'), 'w'))
    for m in (d_rgb, d_words, d_proprio, d_action):
        m.flush()
    loss = train(args.fresh_steps if fresh and round_ == first_round else args.bc_steps if round_ == 0 else args.dagger_steps, round_)
    metrics = {s: evaluate(VLADriver(vla), teacher_driver, s, args.eval_envs, device) for s in STAGES}
    print(f'round {round_} rows {rows} loss {loss:.4f} ({time.time() - start:.0f}s)')
    for s, m in metrics.items():
        print(f'  {s}: {m}')
    log.write(json.dumps({'round': round_, 'rows': rows, 'train_loss': round(loss, 5), 'metrics': metrics}) + '\n')
    log.flush()
    torch.save({'model': vla.state_dict(), 'round': round_, 'metrics': metrics}, args.out)

# PPO for the humanoid model (ml/humanoid_policy.py) on a course (--backend):
#   physicsim     the warehouse course on our engine (robot/warehouse.py): word commands, head camera for the box stages
#   mujoco        the warehouse course on MuJoCo Warp (robot/mj_warehouse.py): the pure motor baseline, a task vector
#   humanoidbench HumanoidBench's G1 stand / walk / run on MuJoCo Warp (robot/hb_course.py), a task vector
# asymmetric: the policy acts from its sensor history + command (+ head camera), the critic values the privileged
# state. smooth control: a temporal smoothness loss on the policy's mean action (CAPS, Mysore et al. 2021): each step's
# mean is pulled toward the average of its neighbors' rollout means in the same episode (--smooth-coef). a reward
# penalty on action changes would also charge the exploration noise, and the policy stops exploring. the policy's state decoder is trained alongside by regression on the true world state (--aux-coef).
# hyperparameters follow legged_gym (rsl_rl PPO): adaptive learning rate from the KL, clipped value loss, values
# bootstrapped through time-outs. the camera is rendered only when some stage needs it (any but walk).
# curriculum: the first --unlocked stages of the course are trained; when the newest one's pass test (robot/
# warehouse.py PASS) passes over the last PASS_WINDOW iterations, the next stage unlocks. walk commands follow a
# curriculum (legged_gym): the top forward speed starts at FORWARD_START and rises a grid step (robot/humanoid.py
# COMMAND_STEP) whenever the walk speed error (EMA over iterations) gets under FORWARD_ERR, up to COMMAND_RANGES'
# (from a 1.5 m/s top, standing still scores nearly as well as trying, and a fresh policy learns only to stand). the newest stage gets half
# the episodes, the earlier ones share the rest.
# --init: start from another run's checkpoint (parameters whose shape changed start fresh; inputs that run never
# varied, e.g. the opponent before the duel, restart their normalizer statistics; the image path restarts if that run
# was blind; the action std restarts at --init-std). the critic trains alone for --critic-warmup iterations first, so
# its first advantages on the new stages don't wreck the policy.
# anchor (continual learning): when a stage unlocks, a frozen snapshot of the policy is taken; on samples of the stages
# before the newest the loss adds --anchor-coef times the KL from the snapshot's mean action to the policy's (at the
# policy's std), so training the new stage doesn't erode the passed ones (with three stages walk got a quarter of the
# episodes and its dash pass fell 0.9 -> 0.05).
#
# run from back/:  python -m ml.train_humanoid --out checkpoints/warehouse.pt --log logs/warehouse.jsonl
# writes checkpoints/humanoid.pt {"policy", "critic", "opt": state_dicts, "iter": int, "unlocked": int, "args": {...}}
#        logs/humanoid.jsonl     one line per iteration {"iter", "steps", "fps", "reward" (mean per step), "episode_len"
#                                (mean length of episodes that ended), "fall_rate" (ended episodes that ended early:
#                                fell, finished or failed the task; metrics "falls" counts falls alone),
#                                "terms": {reward term: mean per step, weighted, before DT}, "metrics": {name: mean
#                                per step} (robot/humanoid.py step()), "policy_loss", "value_loss", "aux_loss", "smooth_loss",
#                                "anchor_loss", "kl",
#                                "lr", "std", "rollout_s", "update_s" (wall time of the two phases), "unlocked",
#                                "pass": {stage: pass rate over the window, for stages with enough tries}}
import argparse
import copy
import json
import math
import os
import time
from collections import deque
import torch
from robot import warehouse, mj_warehouse, hb_course
from robot.humanoid import UPPER_JOINTS, COMMAND_STEP, TOKENS
from ml.humanoid_policy import Policy, Critic, HISTORY, WIDTH, LAYERS, GROUP, load_matching

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['physicsim', 'mujoco', 'humanoidbench'], default='mujoco')
parser.add_argument('--envs', type=int, default=4096)
parser.add_argument('--horizon', type=int, default=24, help='control steps per env per iteration')
parser.add_argument('--iters', type=int, default=3000)
parser.add_argument('--hours', type=float, default=24.0, help='stop after this much wall time')
parser.add_argument('--lr', type=float, default=3e-4, help='starting learning rate, then adapted to --kl-target')
parser.add_argument('--kl-target', type=float, default=0.02)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--minibatches', type=int, default=8, help='optimizer steps per epoch (rsl_rl: 4)')
parser.add_argument('--accum', type=int, default=4, help='micro-batches per minibatch (gradient accumulation, for memory)')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--clip', type=float, default=0.2)
parser.add_argument('--value-coef', type=float, default=1.0)
parser.add_argument('--entropy-coef', type=float, default=0.002)
parser.add_argument('--aux-coef', type=float, default=1.0, help='weight of the state decoder regression')
parser.add_argument('--init-std', type=float, default=0.8)
parser.add_argument('--max-grad-norm', type=float, default=1.0)
parser.add_argument('--save-every', type=int, default=50)
parser.add_argument('--std-to', type=float, default=0.3, help='action std the cap anneals to (noise annealing)')
parser.add_argument('--std-iters', type=int, default=500, help='iterations over which the std cap anneals, from this launch')
parser.add_argument('--smooth-coef', type=float, default=1.0, help='weight of the temporal smoothness loss (CAPS)')
parser.add_argument('--unlocked', type=int, default=1, help='stages trained at the start (the first ones in STAGES)')
parser.add_argument('--resume', action='store_true', help='continue from --out')
parser.add_argument('--init', default='', help='start the policy from this checkpoint (shape-matching parameters)')
parser.add_argument('--anchor-coef', type=float, default=0.01)
parser.add_argument('--critic-warmup', type=int, default=20, help='critic-only iterations after --init')
parser.add_argument('--reset-upper', action='store_true', help='with --init: zero the action head rows of the waist and '
                    'arms, so they start holding the course\'s rest pose (the legs keep their walking)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--width', type=int, default=WIDTH, help='policy transformer width (heads of 64)')
parser.add_argument('--layers', type=int, default=LAYERS)
parser.add_argument('--group', type=int, default=GROUP, help='proprio readings per history token')
parser.add_argument('--out', default='checkpoints/humanoid.pt')
parser.add_argument('--log', default='logs/humanoid.jsonl')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
# the course module, its env class, the command's size and vocabulary (0: a task vector), whether it has a camera
course = {'physicsim': warehouse, 'mujoco': mj_warehouse, 'humanoidbench': hb_course}[args.backend]
if args.backend == 'physicsim':
    Course, command_dim, vocab, has_camera = warehouse.WarehouseCourse, TOKENS, len(warehouse.VOCAB), True
else:
    Course, command_dim, vocab, has_camera = {'mujoco': mj_warehouse.MjWarehouseCourse, 'humanoidbench': hb_course.HbCourse}[args.backend], course.TASK_DIM, 0, False
STAGES, ACTUATORS, PASS = course.STAGES, course.ACTUATORS, course.PASS
# a task vector starts with the stage one-hot: per-stage action noise (see the policy)
policy = Policy(course.PROPRIO_DIM, course.ACTION_DIM, course.AUX_DIM, command_dim, vocab, has_camera, 0 if vocab else len(STAGES),
                args.width, args.layers, args.group).to(device)
critic = Critic(course.PRIV_DIM).to(device)
# the policy's input normalizers (the task vector's too, without words)
norms = [policy.norm, policy.aux_norm] + ([] if vocab else [policy.task_norm])
with torch.no_grad():
    policy.log_std.fill_(torch.log(torch.tensor(args.init_std)).item())
opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=args.lr)
start_iter = 1
unlocked = args.unlocked
# command curriculum (see the header): top forward command (m/s), its start, the error that raises it
FORWARD_START = 0.5
FORWARD_ERR = 0.25
forward_max = FORWARD_START
if args.resume:
    ck = torch.load(args.out, map_location=device)
    policy.load_state_dict(ck['policy'])
    critic.load_state_dict(ck['critic'])
    opt.load_state_dict(ck['opt'])
    start_iter = ck['iter'] + 1
    unlocked = max(unlocked, ck['unlocked'])
    forward_max = ck['forward_max']
    print(f'resumed {args.out} at iter {ck["iter"]}, {unlocked} stages unlocked')
elif args.init:
    ck = torch.load(args.init, map_location=device)
    # a run without the camera never trained the image path: it starts fresh too (its zero stem output, see the policy)
    blind = ck['unlocked'] == 1
    saved = {k: v for k, v in ck['policy'].items() if not (blind and k.startswith('image_'))}
    with torch.no_grad():
        print(f'init policy from {args.init}, fresh: {load_matching(policy, saved)}')
        print(f'init critic from {args.init}, fresh: {load_matching(critic, ck["critic"])}')
        # the action noise restarts too: a policy fine-tuned on new stages explores from --init-std
        policy.log_std.fill_(torch.log(torch.tensor(args.init_std)).item())
        if args.reset_upper:
            # (U,) action indices of the waist and arm actuators
            upper_actions = torch.tensor([ACTUATORS.index(n) for n in UPPER_JOINTS], device=device)
            policy.action_head.weight[upper_actions].zero_()
            policy.action_head.bias[upper_actions].zero_()
    for norm in norms + [critic.norm]:
        norm.forget_constant()
# iterations that train only the critic (after --init)
warmup_until = start_iter + args.critic_warmup if args.init and not args.resume else 0
lr = opt.param_groups[0]['lr']
# compiled forward (fused kernels: the update is mostly many small transformer ops); parameters are shared with policy
policy_fn = torch.compile(policy)
# the anchor (see the header): frozen, its own normalizers; none before the second stage unlocks
anchor = copy.deepcopy(policy).eval().requires_grad_(False) if unlocked > 1 else None
if anchor is not None and args.resume:
    anchor.load_state_dict(ck['anchor'])

# pass tests need this many tries in the window to count
PASS_WINDOW = 20
MIN_TRIES = 200


def stage_probs(k):
    # [float] per stage in STAGES: the first k unlocked, the newest one half the episodes (all of them alone)
    if k == 1:
        return [1.0] + [0.0] * (len(STAGES) - 1)
    return [0.5 / (k - 1)] * (k - 1) + [0.5] + [0.0] * (len(STAGES) - k)


# the camera is rendered once any stage but walk is trained (the policy's batch has one input layout)
vision = has_camera and unlocked > 1
env = Course(args.envs, device, stage_probs(unlocked), vision, seed=args.seed)
# (3,) the env's top command per axis in grid steps: forward from the curriculum (courses with walk commands)
if course.COMMANDED:
    top_forward = int(env.cmd_hi[0])
    env.cmd_hi[0] = round(forward_max / COMMAND_STEP)
# noise annealing: the action std is capped, the cap going linearly from this launch's std to --std-to over --std-iters
# (a policy trained under large noise adapts to it: its noise-free action overshot 1.0 m/s to 1.42 and fell, while
# the noisy one walked; the learned std alone shrank ~0.01 per 100 iterations)
std_start = policy.log_std.exp().max().item()
# EMA of the walk speed error (m/s), iterations since the last raise (the EMA restarts then, and needs RAISE_EVERY
# iterations of the new commands before it counts)
walk_err = None
RAISE_EVERY = 20
since_raise = 0
if device == 'cuda':
    env.enable_cuda_graph()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
os.makedirs(os.path.dirname(args.log), exist_ok=True)
log = open(args.log, 'a')

T, R = args.horizon, env.R
# agents = every robot in every env
N = env.E * R


def observe():
    # {name: (N, ...)} every agent's observation
    return {k: v.reshape(N, *v.shape[2:]) for k, v in env.observe().items()}


def step(action):
    # action (N, A) -> reward (N,), done (N,), timeout (N,) per agent, info {"terms": {name: mean}, "metrics"}
    reward, done, timeout, info = env.step(action.view(env.E, R, -1))
    return reward.reshape(N), done.repeat_interleave(R), timeout.repeat_interleave(R), {'terms': {k: v.mean() for k, v in info['terms'].items()}, 'metrics': info['metrics']}


obs = observe()
P, A = obs['proprio'].shape[-1], env.A
# (N, HISTORY, P) each agent's last HISTORY proprio readings, oldest first (a new episode starts filled with its first)
history = obs['proprio'][:, None].repeat(1, HISTORY, 1)
# rollout buffers (T, N, ...)
buf = {'history': torch.zeros(T, N, HISTORY, P, device=device), 'command': torch.zeros(T, N, obs['command'].shape[-1], dtype=obs['command'].dtype, device=device),
       'priv': torch.zeros(T, N, obs['privileged'].shape[-1], device=device), 'aux': torch.zeros(T, N, obs['aux'].shape[-1], device=device),
       'action': torch.zeros(T, N, A, device=device), 'logp': torch.zeros(T, N, device=device), 'mean': torch.zeros(T, N, A, device=device),
       'value': torch.zeros(T, N, device=device), 'reward': torch.zeros(T, N, device=device), 'done': torch.zeros(T, N, device=device),
       'explore': torch.zeros(T, N, device=device), 'std': torch.zeros(T, N, A, device=device),
       'stage': torch.zeros(T, N, dtype=torch.long, device=device)}
if has_camera:
    buf['image'] = torch.zeros(T, N, *env.image.shape[2:], dtype=torch.uint8, device=device)
# (N,) current episode length per agent
ep_len = torch.zeros(N, device=device)
# [{stage: (tries, passes)}] per recent iteration
window = deque(maxlen=PASS_WINDOW)
start = time.time()

for it in range(start_iter, args.iters + 1):
    # ---- rollout ----
    t0 = time.time()

    ended_len, ended_fell = [], []  # [(k,) lengths of episodes that ended], [(k,) bool fell]
    term_sums = {}  # {reward term: running sum over the rollout of the per-step mean}
    metric_sums = {}  # {metric: running sum over the rollout}
    with torch.no_grad():
        for t in range(T):
            mean, _ = policy_fn(history, obs['command'], obs.get('image'))
            std = policy.std(obs['command'])
            # (N,) pass-test episodes act on the mean
            test = env.test.repeat_interleave(R)
            action = torch.where(test[:, None], mean, mean + std * torch.randn_like(mean))
            value = critic(obs['privileged'])
            buf['history'][t], buf['command'][t], buf['priv'][t], buf['aux'][t] = history, obs['command'], obs['privileged'], obs['aux']
            if vision:
                buf['image'][t] = obs['image']
            buf['action'][t], buf['mean'][t], buf['value'][t], buf['std'][t] = action, mean, value, std
            buf['logp'][t] = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
            buf['explore'][t] = (~test).float()
            buf['stage'][t] = env.stage.repeat_interleave(R)
            reward, done, timeout, info = step(action)
            # a time-out isn't a real end: credit the value the episode would have gone on to earn (legged_gym)
            reward = reward + args.gamma * value * timeout
            buf['reward'][t], buf['done'][t] = reward, done.float()
            for k, v in info['terms'].items():
                term_sums[k] = term_sums.get(k, 0.0) + v / T
            for k, v in info['metrics'].items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v / T
            ep_len += 1
            ended_len.append(ep_len[done])
            ended_fell.append((done & ~timeout)[done])
            ep_len = torch.where(done, 0.0, ep_len)
            obs = observe()
            history = torch.where(done[:, None, None], obs['proprio'][:, None], torch.cat([history[:, 1:], obs['proprio'][:, None]], 1))
        last_value = critic(obs['privileged'])
        # GAE
        adv = torch.zeros(T, N, device=device)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else buf['value'][t + 1]
            nonterminal = 1.0 - buf['done'][t]
            delta = buf['reward'][t] + args.gamma * next_value * nonterminal - buf['value'][t]
            gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + buf['value']

    # ---- update ----
    torch.cuda.synchronize()
    t1 = time.time()
    # CAPS targets: (T, N, A) the average of the neighbor steps' means, (T, N) 1 where both neighbors are in the episode
    smooth_target = torch.zeros_like(buf['mean'])
    smooth_target[1:-1] = (buf['mean'][:-2] + buf['mean'][2:]) / 2
    smooth_valid = torch.zeros(T, N, device=device)
    smooth_valid[1:-1] = (1 - buf['done'][:-2]) * (1 - buf['done'][1:-1])
    flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in buf.items()}
    flat['smooth_target'], flat['smooth_valid'] = smooth_target.reshape(T * N, -1), smooth_valid.reshape(-1)
    flat_adv, flat_ret = adv.reshape(-1), returns.reshape(-1)
    mb = T * N // args.minibatches
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'aux_loss': 0.0, 'smooth_loss': 0.0, 'anchor_loss': 0.0, 'kl': 0.0}  # {name: mean over minibatches}
    for epoch in range(args.epochs):
        perm = torch.randperm(T * N, device=device)
        for i in range(args.minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            # advantages normalized per stage over the minibatch's exploring samples (pass tests don't train the policy):
            # each stage's gradient at unit scale. normalized together, the box stages' task rewards (~100x walk's
            # per-step rewards) drowned walk's signal and it forgot the dash while pick trained
            keep_mb = flat['explore'][idx]
            adv_mb = flat_adv[idx]
            # (mb,) each sample's stage, (mb, stages) its one-hot over the exploring samples
            stage_mb = flat['stage'][idx]
            in_stage = (stage_mb[:, None] == torch.arange(len(STAGES), device=device)).float() * keep_mb[:, None]
            # (stages,) exploring samples, advantage mean and std per stage
            count = in_stage.sum(0).clamp_min(1)
            adv_mean = (adv_mb[:, None] * in_stage).sum(0) / count
            adv_std = (((adv_mb[:, None] - adv_mean) ** 2 * in_stage).sum(0) / count).sqrt()
            adv_mb = (adv_mb - adv_mean[stage_mb]) / (adv_std[stage_mb] + 1e-8)
            opt.zero_grad()
            # one optimizer step per minibatch, its gradient accumulated over --accum micro-batches (activation memory)
            kl = torch.zeros((), device=device)
            for micro, a, keep in zip(idx.chunk(args.accum), adv_mb.chunk(args.accum), keep_mb.chunk(args.accum)):
                mean, aux_pred = policy_fn(flat['history'][micro], flat['command'][micro], flat['image'][micro] if vision else None)
                std = policy.std(flat['command'][micro])
                # (b, A) the rollout's action std, for the KL
                old_std = flat['std'][micro]
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(flat['action'][micro]).sum(-1)
                ratio = torch.exp(logp - flat['logp'][micro])
                policy_loss = -(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a) * keep).sum() / keep.sum().clamp_min(1)
                value = critic(flat['priv'][micro])
                old_v = flat['value'][micro]
                v_clipped = old_v + (value - old_v).clamp(-args.clip, args.clip)
                value_loss = torch.max((value - flat_ret[micro]) ** 2, (v_clipped - flat_ret[micro]) ** 2).mean()
                aux_loss = ((aux_pred - policy.aux_norm(flat['aux'][micro])) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                valid = flat['smooth_valid'][micro]
                smooth_loss = (((mean - flat['smooth_target'][micro]) ** 2).mean(-1) * valid).sum() / valid.sum().clamp_min(1)
                loss = policy_loss + args.value_coef * value_loss + args.aux_coef * aux_loss - args.entropy_coef * entropy + args.smooth_coef * smooth_loss
                anchor_loss = torch.zeros((), device=device)
                if anchor is not None:
                    # (b,) 1 on samples of the stages before the newest
                    older = (flat['stage'][micro] < unlocked - 1).float()
                    with torch.no_grad():
                        # (b, A) the anchor's mean action
                        anchor_mean, _ = anchor(flat['history'][micro], flat['command'][micro], flat['image'][micro] if vision else None)
                    # KL(anchor || policy) at the policy's std, mean over those samples
                    anchor_loss = ((((mean - anchor_mean) / std) ** 2).sum(-1) / 2 * older).sum() / older.sum().clamp_min(1)
                    loss = loss + args.anchor_coef * anchor_loss
                if it < warmup_until:
                    loss = args.value_coef * value_loss
                (loss / args.accum).backward()
                with torch.no_grad():
                    # KL(old || new) between the diagonal gaussians, mean over the minibatch
                    kl += (torch.log(std / old_std) + (old_std ** 2 + (flat['mean'][micro] - mean) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean() / args.accum
                n = args.epochs * args.minibatches * args.accum
                stats['policy_loss'] += policy_loss.item() / n
                stats['value_loss'] += value_loss.item() / n
                stats['aux_loss'] += aux_loss.item() / n
                stats['smooth_loss'] += smooth_loss.item() / n
                stats['anchor_loss'] += anchor_loss.item() / n
            stats['kl'] += kl.item() / (args.epochs * args.minibatches)
            # adaptive learning rate (rsl_rl): shrink when the policy moved too far, grow when it barely moved (not while
            # only the critic trains: the policy doesn't move at all then)
            if it >= warmup_until and kl > 2 * args.kl_target:
                lr = max(1e-5, lr / 1.5)
            elif it >= warmup_until and kl < args.kl_target / 2:
                lr = min(1e-2, lr * 1.5)
            for g in opt.param_groups:
                g['lr'] = lr
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), args.max_grad_norm)
            opt.step()
            cap = max(args.std_to, std_start + (args.std_to - std_start) * (it - start_iter) / args.std_iters)
            with torch.no_grad():
                policy.log_std.clamp_(max=torch.log(torch.tensor(cap)).item())

    # normalizers learn from this rollout's inputs after the update, so the update saw the inputs the rollout acted on
    policy.norm.update(buf['history'][:, :, -1])
    if not vocab:
        policy.task_norm.update(buf['command'])
    policy.aux_norm.update(buf['aux'])
    critic.norm.update(buf['priv'])
    ended_len, ended_fell = torch.cat(ended_len), torch.cat(ended_fell)
    t2 = time.time()
    elapsed = time.time() - start
    row = {'iter': it, 'steps': it * T * N, 'fps': (it - start_iter + 1) * T * N / elapsed,
           'reward': buf['reward'].mean().item(), 'episode_len': ended_len.mean().item() if len(ended_len) else None,
           'fall_rate': ended_fell.float().mean().item() if len(ended_fell) else None,
           'terms': {k: round(float(v), 5) for k, v in term_sums.items()}, 'metrics': {k: round(float(v), 5) for k, v in metric_sums.items()},
           **stats, 'lr': lr, 'std': policy.log_std.exp().mean().item(), 'rollout_s': t1 - t0, 'update_s': t2 - t1}
    # pass tests: tries / passes this iteration (metrics are means per step)
    window.append({st: (metric_sums.get(f'tried_{st}', 0.0) * T, metric_sums.get(f'passed_{st}', 0.0) * T) for st in STAGES})
    tries = {st: sum(float(w[st][0]) for w in window) for st in STAGES}  # {stage: tries in the window}
    rates = {st: sum(float(w[st][1]) for w in window) / tries[st] for st in STAGES if tries[st] >= MIN_TRIES}  # {stage: pass rate}
    row['unlocked'], row['pass'] = unlocked, {k: round(v, 3) for k, v in rates.items()}
    err = row['metrics']['walk_lin_err'] if course.COMMANDED else math.inf
    walk_err = err if walk_err is None else 0.9 * walk_err + 0.1 * err
    since_raise += 1
    if course.COMMANDED and walk_err < FORWARD_ERR and env.cmd_hi[0] < top_forward and since_raise >= RAISE_EVERY:
        env.cmd_hi[0] += 1
        forward_max = float(env.cmd_hi[0]) * COMMAND_STEP
        walk_err = None
        since_raise = 0
        print(f'it {it}: walk commands up to {forward_max:.2f} m/s forward', flush=True)
    row['forward_max'] = forward_max
    newest = STAGES[unlocked - 1]
    if unlocked < len(STAGES) and rates.get(newest, 0.0) >= PASS[newest]:
        unlocked += 1
        env.stage_probs = torch.tensor(stage_probs(unlocked), device=device)
        env.vision = vision = has_camera
        window.clear()
        anchor = copy.deepcopy(policy).eval().requires_grad_(False)
        print(f'it {it}: {newest} passed ({rates[newest]:.2f}), unlocked {STAGES[unlocked - 1]}', flush=True)
    log.write(json.dumps(row) + '\n')
    log.flush()
    if it % 10 == 0 or it == start_iter:
        print(f"it {it} | {row['fps']:.0f} sps | reward {row['reward']:.4f} | ep len {row['episode_len']} | fall {row['fall_rate']} | "
              f"{' '.join(f'{k} {v:.3f}' for k, v in row['metrics'].items() if not k.startswith(('tried_', 'passed_')))} | pass {row['pass']} | aux {stats['aux_loss']:.3f} | anchor {stats['anchor_loss']:.3f} | kl {stats['kl']:.4f} | lr {lr:.1e} | std {row['std']:.3f}", flush=True)
    if it % args.save_every == 0 or it == args.iters or elapsed > args.hours * 3600:
        torch.save({'policy': policy.state_dict(), 'critic': critic.state_dict(), 'opt': opt.state_dict(), 'iter': it,
                    'unlocked': unlocked, 'forward_max': forward_max, 'anchor': anchor.state_dict() if anchor is not None else None,
                    'args': vars(args)}, args.out)
    if elapsed > args.hours * 3600:
        break

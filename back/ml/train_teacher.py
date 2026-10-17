# PPO for the teacher (ml/teacher.py) on the whole course: every env samples a stage at reset (stage mixing), one
# network drives every robot (self-play in the race). see DOC.md and MINDOLOGY writing/training/PPO.md.
#
# stage mixing: per stage a running success rate (EMA over iterations of the finished episodes' success). a stage's
# sampling weight is max(1 - success, FLOOR); stage k+1 unlocks when stage k reaches UNLOCK (drive -> pick -> bring
# -> race + speedrun together; race "success" is winning self-play, ~50% at best, so it never gates anything).
#
# --init (from ml/bc_teacher.py, the actor cloned from the scripted expert): every stage unlocked from the start, the
# first --critic-warmup iterations train only the critic (a random critic's advantages would wreck the cloned actor),
# and the update keeps cloning the expert demos with weight bc_coef * bc_decay^iter (DAPG, Rajeswaran et al. 2018).
# the input normalizer stays frozen at the clone's statistics.
#
# --home-curriculum: reverse curriculum over the speedrun's subcourses (episodes starting with k blocks already home,
# robot/course.py): only k = NUM_BLOCKS - 1 (one block left) at first; k's sampling weight is max(1 - its success,
# FLOOR), and k - 1 unlocks when k succeeds --unlock-home of the time, down to the full speedrun (k = 0).
#
# run from back/:  python -m ml.train_teacher --init checkpoints/teacher_bc.pt
# writes checkpoints/teacher.pt {"model": state_dict, "opt": state_dict, "iter": int, "success": [S], "unlocked": [S],
#                                "home_success": [NUM_BLOCKS], "home_unlocked": [NUM_BLOCKS],
#                                "args": {...}}
#        logs/teacher.jsonl     one line per iteration {"iter", "steps", "fps", "success": {stage: float},
#                               "probs": {stage: float}, "home_success" / "home_probs": [per k] (speedrun subcourses),
#                               "speedrun_blocks": mean blocks a finished speedrun brought home itself (score - k),
#                               "return": {stage: mean unscaled episode return of the
#                               robot-episodes that ended this iteration}, "policy_loss", "value_loss", "kl", "lr", "std", "explained_var"}
import argparse
import json
import math
import os
import time
import torch
from robot.course import CourseEnv, STAGES, NUM_BLOCKS
from robot.lekiwi import ACTUATORS
from ml.teacher import Teacher

parser = argparse.ArgumentParser()
parser.add_argument('--envs', type=int, default=4096)
parser.add_argument('--horizon', type=int, default=32, help='control steps per env per iteration')
parser.add_argument('--iters', type=int, default=1500)
parser.add_argument('--hours', type=float, default=12.0, help='stop after this much wall time')
parser.add_argument('--lr', type=float, default=3e-4, help='critic lr, and the actor\'s starting lr')
parser.add_argument('--actor-lr', type=float, default=0.0, help='actor starting lr if > 0 (with --init: small, a tiny std makes the KL very sensitive)')
parser.add_argument('--kl-target', type=float, default=0.01, help='adaptive lr: shrink when KL > 2x, grow when < 0.5x')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--minibatches', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--clip', type=float, default=0.2)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--entropy-coef', type=float, default=0.001)
parser.add_argument('--floor', type=float, default=0.1, help='min sampling weight of an unlocked stage')
parser.add_argument('--unlock', type=float, default=0.8, help='success rate of stage k that unlocks stage k+1')
parser.add_argument('--save-every', type=int, default=25)
parser.add_argument('--resume', action='store_true', help='continue from --out')
parser.add_argument('--init', default='', help='start from this behavior-cloned teacher (ml/bc_teacher.py)')
parser.add_argument('--init-std', type=float, default=0.2, help='action std at the start with --init')
parser.add_argument('--init-std-wheels', type=float, default=0.0, help='if > 0: the wheels\' start std (driving tolerates '
                    'far more noise than grasping)')
parser.add_argument('--critic-warmup', type=int, default=15, help='iterations of critic-only training with --init')
parser.add_argument('--demos', default='data/expert_demos.pt', help='expert demos for the BC term (with --init)')
parser.add_argument('--bc-coef', type=float, default=1.0)
parser.add_argument('--bc-decay', type=float, default=0.995, help='per iteration')
parser.add_argument('--bc-batch', type=int, default=8192)
parser.add_argument('--home-curriculum', action='store_true', help='reverse curriculum over the speedrun subcourses')
parser.add_argument('--unlock-home', type=float, default=0.5, help='success at k blocks home that unlocks k - 1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out', default='checkpoints/teacher.pt')
parser.add_argument('--log', default='logs/teacher.jsonl')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
S = len(STAGES)
model = Teacher().to(device)
# two groups: actor (+ log_std) with the KL-adaptive lr, critic with a fixed lr
actor_params = list(model.actor.parameters()) + [model.log_std]  # [nn.Parameter]
opt = torch.optim.Adam([{'params': actor_params}, {'params': model.critic.parameters()}], lr=args.lr, eps=1e-5)
# (S,) running success rate per stage, (S,) stage can be sampled
success = torch.zeros(S, device=device)
unlocked = torch.tensor([s == 'drive' for s in STAGES], device=device)
# (NUM_BLOCKS,) speedrun subcourses: running success rate per k blocks already home, k can be sampled. without
# --home-curriculum only k = 0 (the full speedrun)
home_success = torch.zeros(NUM_BLOCKS, device=device)
home_unlocked = torch.arange(NUM_BLOCKS, device=device) == (NUM_BLOCKS - 1 if args.home_curriculum else 0)
start_iter = 1
# expert demos for the BC term: (D, STATE_DIM), (D, ACTION_DIM), or None without --init
demo_state = demo_action = None
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=device)['model'])
    with torch.no_grad():
        model.log_std.fill_(math.log(args.init_std))
        if args.init_std_wheels > 0:
            model.log_std[[i for i, (n, kind) in enumerate(ACTUATORS) if kind == 'wheel']] = math.log(args.init_std_wheels)
    unlocked[:] = True
    demos = torch.load(args.demos)
    demo_state, demo_action = demos['state'].to(device), demos['action'].to(device)
    print(f'init from {args.init}, {len(demo_state)} expert demo rows')
if args.resume:
    ck = torch.load(args.out, map_location=device)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    success, unlocked = ck['success'].to(device), ck['unlocked'].to(device)
    if 'home_success' in ck and args.home_curriculum:
        home_success, home_unlocked = ck['home_success'].to(device), ck['home_unlocked'].to(device)
    start_iter = ck['iter'] + 1
    print(f'resumed {args.out} at iter {ck["iter"]}')


def stage_probs():
    # (S,) sampling probability of each stage from the running success rates
    w = torch.where(unlocked, (1 - success).clamp(min=args.floor), 0.0)
    return w / w.sum()


def home_probs():
    # (NUM_BLOCKS,) sampling probability of each speedrun subcourse (k blocks already home)
    w = torch.where(home_unlocked, (1 - home_success).clamp(min=args.floor), 0.0)
    return w / w.sum()


env = CourseEnv(args.envs, device, stage_probs=stage_probs().tolist(), seed=args.seed)
env.home_probs = home_probs()
if device == 'cuda':
    env.enable_cuda_graph()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
os.makedirs(os.path.dirname(args.log), exist_ok=True)
log = open(args.log, 'a')

T, E, R = args.horizon, args.envs, env.R
# agents = every robot in every env
N = E * R
A = model.log_std.shape[0]
# rollout buffers (T, N, ...): state, sampled action, its log prob, the old mean action (for the KL), value,
# reward, episode ended after this step, robot was active (the speedrun's parked robot is not trained on)
buf_state = torch.zeros(T, N, env.state().shape[-1], device=device)
buf_action = torch.zeros(T, N, A, device=device)
buf_logp = torch.zeros(T, N, device=device)
buf_mean = torch.zeros(T, N, A, device=device)
buf_value = torch.zeros(T, N, device=device)
buf_reward = torch.zeros(T, N, device=device)
buf_done = torch.zeros(T, N, device=device)
buf_active = torch.zeros(T, N, dtype=torch.bool, device=device)
# (N,) unscaled return of each agent's current episode (logged per stage)
ep_ret = torch.zeros(N, device=device)
# running discounted return per agent and the running variance of it, for reward scaling (Engstrom et al. 2020)
ret = torch.zeros(N, device=device)
ret_var, ret_count = 1.0, 1e-4
# actor lr, adapted to keep the KL near --kl-target
lr = args.actor_lr if args.actor_lr > 0 else args.lr
opt.param_groups[0]['lr'] = lr
start = time.time()
state = env.state().view(N, -1)
for it in range(start_iter, args.iters + 1):
    # (S,) finished robot-episodes and their successes this iteration
    ep_count = torch.zeros(S, device=device)
    ep_success = torch.zeros(S, device=device)
    ep_return = torch.zeros(S, device=device)
    # (NUM_BLOCKS,) finished speedrun episodes and successes per k blocks already home; blocks they brought home
    home_count = torch.zeros(NUM_BLOCKS, device=device)
    home_succ = torch.zeros(NUM_BLOCKS, device=device)
    run_blocks = torch.zeros(NUM_BLOCKS, device=device)
    t0 = time.time()
    with torch.no_grad():
        for t in range(T):
            mean, value = model(state)
            dist = torch.distributions.Normal(mean, model.log_std.exp())
            action = dist.sample()
            # (E, R) robots acting this step
            active = env.active()
            buf_state[t], buf_action[t], buf_mean[t], buf_value[t] = state, action, mean, value
            buf_logp[t] = dist.log_prob(action).sum(-1)
            buf_active[t] = active.flatten()
            reward, done, info = env.step(action.view(E, R, A))
            reward = reward.flatten()
            # (N,) per agent
            agent_done = done.repeat_interleave(R)
            ret = ret * args.gamma * (~agent_done) + reward
            batch_var = ret.var().item()
            count = ret_count + N
            ret_var = (ret_var * ret_count + batch_var * N) / count
            ret_count = count
            buf_reward[t] = reward / (ret_var ** 0.5 + 1e-8)
            buf_done[t] = agent_done.float()
            # per-stage success of the finished episodes' active robots
            ended = done[:, None] & active
            stage_oh = torch.nn.functional.one_hot(info['stage'], S).float()
            ep_count += (stage_oh * ended.sum(-1, keepdim=True)).sum(0)
            ep_success += (stage_oh * (ended & info['success']).sum(-1, keepdim=True)).sum(0)
            ep_ret += reward
            ep_return += (stage_oh * (ended.float() * ep_ret.view(E, R)).sum(-1, keepdim=True)).sum(0)
            run_end = done & (info['stage'] == STAGES.index('speedrun'))
            k_oh = torch.nn.functional.one_hot(info['start_home'], NUM_BLOCKS).float() * run_end[:, None]
            home_count += k_oh.sum(0)
            home_succ += (k_oh * info['success'][:, :1]).sum(0)
            run_blocks += (k_oh * (info['score'][:, :1] - info['start_home'][:, None]).float()).sum(0)
            ep_ret = ep_ret * (~agent_done)
            state = env.state().view(N, -1)
        _, next_value = model(state)
        # GAE, no bootstrap across episode ends (time left is in the state, so a timeout is a true end)
        adv = torch.zeros(T, N, device=device)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else buf_value[t + 1]
            delta = buf_reward[t] + args.gamma * nv * (1 - buf_done[t]) - buf_value[t]
            gae = delta + args.gamma * args.gae_lambda * (1 - buf_done[t]) * gae
            adv[t] = gae
        returns = adv + buf_value
    collect_time = time.time() - t0

    # PPO update over the active agents' samples, flattened to (M, ...)
    keep = buf_active.flatten()
    f_state, f_action, f_logp, f_mean, f_adv, f_ret, f_value = (x.flatten(0, 1)[keep] for x in (buf_state, buf_action, buf_logp, buf_mean, adv, returns, buf_value))
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    M = f_state.shape[0]
    old_std = model.log_std.exp().detach()
    # critic-only warmup after --init: the actor (and its std) gets no gradient, the lr stays put
    actor_on = not args.init or it > args.critic_warmup
    bc_w = args.bc_coef * args.bc_decay ** it if demo_state is not None else 0.0
    for _ in range(args.epochs):
        for mb in torch.randperm(M, device=device).chunk(args.minibatches):
            mean, value = model(f_state[mb])
            std = model.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            ratio = (dist.log_prob(f_action[mb]).sum(-1) - f_logp[mb]).exp()
            policy_loss = -torch.minimum(ratio * f_adv[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * f_adv[mb]).mean()
            value_loss = ((value - f_ret[mb]) ** 2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = args.value_coef * value_loss
            if actor_on:
                loss = loss + policy_loss - args.entropy_coef * entropy
            if actor_on and bc_w > 1e-3:
                d = torch.randint(0, len(demo_state), (args.bc_batch,), device=device)
                loss = loss + bc_w * ((model(demo_state[d])[0] - demo_action[d]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            opt.step()
            # adaptive lr from the exact KL(old || new) of the gaussians (as in legged_gym / rl_games)
            with torch.no_grad():
                kl = (torch.log(std / old_std) + (old_std ** 2 + (f_mean[mb] - mean) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean().item()
            if actor_on and kl > 2 * args.kl_target:
                lr = max(lr / 1.5, 1e-7)
            elif actor_on and kl < args.kl_target / 2:
                lr = min(lr * 1.5, 1e-2)
            opt.param_groups[0]['lr'] = lr
    # normalizer: fold in this rollout's states after the update, so collection and update saw the same normalization.
    # frozen with --init: the cloned actor only works on the input scaling it was cloned with
    if not args.init:
        model.update_norm(f_state)

    # stage mixing
    seen = ep_count > 0
    success = torch.where(seen, 0.9 * success + 0.1 * ep_success / ep_count.clamp(min=1), success)
    for k, nxt in ((0, [1]), (1, [2]), (2, [3, 4])):
        if success[k] >= args.unlock:
            unlocked[nxt] = True
    env.stage_probs = stage_probs()
    # speedrun subcourses: one fewer block home once this many succeed
    home_success = torch.where(home_count > 0, 0.9 * home_success + 0.1 * home_succ / home_count.clamp(min=1), home_success)
    for k in range(1, NUM_BLOCKS):
        if args.home_curriculum and home_unlocked[k] and home_success[k] >= args.unlock_home:
            home_unlocked[k - 1] = True
    env.home_probs = home_probs()
    if args.home_curriculum:
        # the speedrun stage counts as mastered by the full run only, not by its easier subcourses
        success[STAGES.index('speedrun')] = home_success[0]
        env.stage_probs = stage_probs()

    explained_var = 1 - (f_ret - f_value).var().item() / (f_ret.var().item() + 1e-8)
    steps = it * T * N
    rec = {'iter': it, 'steps': steps, 'fps': round(T * N / (time.time() - t0)), 'collect_s': round(collect_time, 2),
           'success': {s: round(success[i].item(), 3) for i, s in enumerate(STAGES)},
           'probs': {s: round(p, 3) for s, p in zip(STAGES, env.stage_probs.tolist())},
           'home_success': [round(x, 3) for x in home_success.tolist()], 'home_probs': [round(x, 3) for x in env.home_probs.tolist()],
           'speedrun_blocks': round((run_blocks.sum() / home_count.sum().clamp(min=1)).item(), 3),
           'return': {s: round(ep_return[i].item() / ep_count[i].item(), 3) for i, s in enumerate(STAGES) if ep_count[i] > 0},
           'policy_loss': round(policy_loss.item(), 4), 'value_loss': round(value_loss.item(), 4), 'kl': round(kl, 4),
           'lr': lr, 'bc_w': round(bc_w, 4), 'std': round(model.log_std.exp().mean().item(), 3), 'explained_var': round(explained_var, 3)}
    log.write(json.dumps(rec) + '\n')
    log.flush()
    print(f'iter {it} steps {steps / 1e6:.1f}M fps {rec["fps"]} success {rec["success"]} home {rec["home_success"]} / {rec["home_probs"]} '
          f'run blocks {rec["speedrun_blocks"]} return {rec["return"]} probs {rec["probs"]} '
          f'kl {rec["kl"]} lr {lr:.1e} std {rec["std"]} ev {rec["explained_var"]} ({time.time() - start:.0f}s)')
    out_of_time = time.time() - start > args.hours * 3600
    if it % args.save_every == 0 or it == args.iters or out_of_time:
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'iter': it, 'success': success.cpu(),
                    'unlocked': unlocked.cpu(), 'home_success': home_success.cpu(), 'home_unlocked': home_unlocked.cpu(),
                    'args': vars(args)}, args.out)
    if out_of_time:
        print(f'stopping: {args.hours} h limit')
        break

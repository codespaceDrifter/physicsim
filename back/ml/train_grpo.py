# GRPO fine-tune of the distilled VLA (ml/distill.py) on episode outcomes: the RLVR recipe for LLMs, with a robot.
# every iteration runs one stage (round robin). envs come in groups of G that start from the same copied state
# (same layout, same command, like G answers to one prompt); the VLA samples action chunks around its means
# (gaussian, learned std) and each robot's episode gets one outcome reward:
#   drive / pick / bring: 1 if it succeeded, else 0
#   race: 1 win, 0.5 tie, 0 loss against the frozen teacher (robot 1)
#   speedrun: blocks home / 6, + the fraction of time left if all 6 are home
# advantage = (reward - group mean) / group std, the same for every decision of that episode (no critic). loss: the
# clipped PPO objective per decision + beta * KL(policy || reference), reference = the frozen distilled VLA.
# groups whose rollouts all got the same reward carry no signal and are dropped before the update (as in DAPO): with
# zero advantage the gradient is pure rounding noise, which Adam would still turn into full-size steps.
# only the first episode of each env counts (envs auto-reset when done; later steps are ignored).
#
# run from back/:  python -m ml.train_grpo --init checkpoints/vla_sft.pt --teacher checkpoints/teacher.pt
# writes checkpoints/vla_grpo.pt {"model": state_dict, "iter": int, "metrics": {stage: evaluate() dict}}
#        logs/grpo.jsonl         {"iter", "stage", "reward", "signal_groups", "decisions", "kl", "clip_frac", "std", "collect_s"} per iteration,
#                                {"iter", "metrics"} per evaluation
import argparse
import copy
import json
import math
import os
import time
import torch
from robot.course import CourseEnv, STAGES, NUM_BLOCKS
from ml.teacher import Teacher
from ml.vla import VLA, CHUNK
from ml.rollout import VLADriver, TeacherDriver, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--init', default='checkpoints/vla_sft.pt')
parser.add_argument('--teacher', default='checkpoints/teacher.pt')
parser.add_argument('--envs', type=int, default=512)
parser.add_argument('--group', type=int, default=8)
parser.add_argument('--iters', type=int, default=300)
parser.add_argument('--hours', type=float, default=12.0, help='stop after this much wall time')
parser.add_argument('--lr', type=float, default=1e-6, help='small: the log prob of a 45-number chunk at std 0.1 is very sensitive to the means')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--minibatch', type=int, default=256, help='decisions per gradient step')
parser.add_argument('--clip', type=float, default=0.2)
parser.add_argument('--target-kl', type=float, default=0.05, help='stop the iteration\'s updates once KL(sampling || current) passes this (as in SB3)')
parser.add_argument('--beta', type=float, default=0.01, help='weight of the KL to the reference')
parser.add_argument('--init-std', type=float, default=0.1, help='std of the sampled actions at the start')
parser.add_argument('--eval-every', type=int, default=25)
parser.add_argument('--eval-envs', type=int, default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out', default='checkpoints/vla_grpo.pt')
parser.add_argument('--log', default='logs/grpo.jsonl')
args = parser.parse_args()
assert args.envs % args.group == 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
teacher = Teacher().to(device)
teacher.load_state_dict(torch.load(args.teacher, map_location=device)['model'])
teacher.eval()
policy = VLA().to(device)
policy.load_state_dict(torch.load(args.init, map_location=device)['model'])
with torch.no_grad():
    policy.log_std.fill_(math.log(args.init_std))
ref = copy.deepcopy(policy).eval().requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
os.makedirs(os.path.dirname(args.log), exist_ok=True)
log = open(args.log, 'a')

env = CourseEnv(args.envs, device, stage_probs=[1.0] * len(STAGES), seed=args.seed)
if device == 'cuda':
    env.enable_cuda_graph()
E, R, G = env.E, env.R, args.group
# (E,) env -> the group leader whose start it copies
leader = torch.arange(E, device=device) // G * G
followers = leader != torch.arange(E, device=device)
driver = VLADriver(policy, sample=True, execute=CHUNK)
teacher_driver = TeacherDriver(teacher)
all_envs = torch.ones(E, dtype=torch.bool, device=device)


def gaussian_logp_kl(model, rgb, words, proprio, chunk):
    # decisions (B, ...) -> log prob of the sampled chunk (B,) under model, KL(model || ref) per decision (B,)
    mean = model(rgb, words, proprio)
    std = model.log_std.exp()
    logp = torch.distributions.Normal(mean, std).log_prob(chunk).sum((-1, -2))
    with torch.no_grad():
        ref_mean = ref(rgb, words, proprio)
    ref_std = ref.log_std.exp()
    kl = (torch.log(ref_std / std) + (std ** 2 + (mean - ref_mean) ** 2) / (2 * ref_std ** 2) - 0.5).sum((-1, -2))
    return logp, kl


start = time.time()
for it in range(1, args.iters + 1):
    stage = STAGES[(it - 1) % len(STAGES)]
    race = stage == 'race'
    # new episodes everywhere in this stage, then every group copies its leader's start
    env.stage_probs = torch.tensor([float(s == stage) for s in STAGES], device=device)
    env.reset(all_envs)
    env.copy_envs(leader[followers], followers.nonzero().view(-1))
    # (E, R) robots the policy drives and trains on
    trained = env.active()
    trained[:, 1] &= not race
    # decisions: lists of per-plan tensors, concatenated after the rollout
    d_rgb, d_words, d_proprio, d_chunk, d_env, d_robot = [], [], [], [], [], []  # [Tensor]
    finished = torch.zeros(E, dtype=torch.bool, device=device)
    reward = torch.zeros(E, R, device=device)
    fresh = all_envs.clone()
    t0 = time.time()
    with torch.no_grad():
        for t in range(int(env.stage_steps.max())):
            action = driver(env, fresh)
            plan = driver.plan
            if plan is not None:
                # keep decisions of first episodes, for trained robots: (n, R) -> (m,)
                keep = (~finished[plan['envs']])[:, None] & trained[plan['envs']]
                e_idx, r_idx = keep.nonzero().unbind(-1)
                d_rgb.append(plan['rgb'][e_idx, r_idx])
                d_words.append(plan['words'][e_idx, r_idx])
                d_proprio.append(plan['proprio'][e_idx, r_idx])
                d_chunk.append(plan['chunk'][e_idx, r_idx])
                d_env.append(plan['envs'][e_idx])
                d_robot.append(r_idx)
            if race:
                action[:, 1] = teacher_driver(env, fresh)[:, 1]
            _, done, info = env.step(action)
            newly = done & ~finished
            score = info['score'].float()
            if race:
                outcome = (score > score.flip(-1)).float() + 0.5 * (score == score.flip(-1)).float()
            elif stage == 'speedrun':
                outcome = score / NUM_BLOCKS + info['success'].float() * (1 - (t + 1) / env.stage_steps[env.stage])[:, None]
            else:
                outcome = info['success'].float()
            reward = torch.where(newly[:, None], outcome, reward)
            finished |= done
            fresh = done
            if finished.all():
                break
    assert finished.all()
    collect_time = time.time() - t0

    # group-relative advantage per (env, robot): (E, R) -> (E / G, G, R)
    rg = reward.view(E // G, G, R)
    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(E, R)
    # (D,) advantage of every decision, then only decisions from groups with a signal
    d_adv = adv[torch.cat(d_env), torch.cat(d_robot)]
    signal = d_adv != 0
    rgb, words, proprio, chunk = (torch.cat(x)[signal] for x in (d_rgb, d_words, d_proprio, d_chunk))
    d_adv = d_adv[signal]
    D = len(d_adv)
    kls, clips = [], []  # [float]
    # no group with a signal (e.g. all failed): nothing to learn this iteration
    if D > 0:
        # old log probs (the policy that sampled), in minibatches
        with torch.no_grad():
            old_logp = torch.cat([gaussian_logp_kl(policy, rgb[mb], words[mb], proprio[mb], chunk[mb])[0]
                                  for mb in torch.arange(D, device=device).split(args.minibatch)])
        approx_kl = 0.0
        for _ in range(args.epochs):
            for mb in torch.randperm(D, device=device).split(args.minibatch):
                if approx_kl > args.target_kl:
                    break
                # fp32, like the sampling pass: with a 0.01 std, bf16 rounding of the means alone wrecks the ratios
                logp, kl = gaussian_logp_kl(policy, rgb[mb], words[mb], proprio[mb], chunk[mb])
                ratio = (logp - old_logp[mb]).exp()
                a = d_adv[mb]
                policy_loss = -torch.minimum(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                loss = policy_loss + args.beta * kl.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                kls.append(kl.mean().item())
                clips.append(((ratio - 1).abs() > args.clip).float().mean().item())
                # k3 estimator of KL(sampling policy || current) on this minibatch
                with torch.no_grad():
                    log_r = old_logp[mb] - logp
                    approx_kl = (log_r.exp() - 1 - log_r).mean().item()

    # fraction of (group, trained robot) whose rollouts disagreed, i.e. that gave a learning signal
    signal_groups = (rg.std(1) > 0)[:, trained.view(E // G, G, R)[:, 0].any(0)].float().mean().item()
    mean_reward = reward[trained].mean().item()
    rec = {'iter': it, 'stage': stage, 'reward': round(mean_reward, 4), 'signal_groups': round(signal_groups, 3), 'decisions': D, 'updates': len(kls),
           'kl': round(sum(kls) / max(len(kls), 1), 4), 'clip_frac': round(sum(clips) / max(len(clips), 1), 4),
           'std': round(policy.log_std.exp().mean().item(), 4), 'collect_s': round(collect_time, 1)}
    log.write(json.dumps(rec) + '\n')
    log.flush()
    print(f'iter {it} {stage}: reward {rec["reward"]} signal groups {rec["signal_groups"]} kl {rec["kl"]} clip {rec["clip_frac"]} '
          f'std {rec["std"]} decisions {D} ({time.time() - start:.0f}s)')
    out_of_time = time.time() - start > args.hours * 3600
    if it % args.eval_every == 0 or it == args.iters or out_of_time:
        metrics = {s: evaluate(VLADriver(policy), teacher_driver, s, args.eval_envs, device) for s in STAGES}
        print(f'iter {it} eval {metrics}')
        log.write(json.dumps({'iter': it, 'metrics': metrics}) + '\n')
        log.flush()
        torch.save({'model': policy.state_dict(), 'iter': it, 'metrics': metrics}, args.out)
    if out_of_time:
        print(f'stopping: {args.hours} h limit')
        break

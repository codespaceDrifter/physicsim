# PPO fine-tuning of the pixel policy with the env's shaped reward, starting from the behavior cloning checkpoint.
# clipped PPO (Schulman et al. 2017) with GAE, reward scaling by the running std of the discounted return,
# gaussian exploration around the policy's mean action.
#
# run from back/:  python -m ml.train_ppo --init checkpoints/bc.pt --iters 2000
# writes checkpoints/ppo.pt  {"policy": state_dict, "iter": int, "success": float}
import argparse
import os
import time
import torch
from robot.pick_place import PickPlaceEnv
from ml.policy import Policy, FrameStack
from ml.rollout import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--init', default='checkpoints/bc.pt', help='behavior cloning checkpoint, "" to start from scratch')
parser.add_argument('--envs', type=int, default=256)
parser.add_argument('--horizon', type=int, default=32, help='control steps per env per iteration')
parser.add_argument('--iters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--minibatch', type=int, default=1024)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae-lambda', type=float, default=0.95)
parser.add_argument('--clip', type=float, default=0.2)
parser.add_argument('--value-coef', type=float, default=0.5)
parser.add_argument('--init-log-std', type=float, default=-1.6)
parser.add_argument('--eval-every', type=int, default=100)
parser.add_argument('--eval-envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--out', default='checkpoints/ppo.pt')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = PickPlaceEnv(args.envs, device, num_blocks=1, seed=args.seed)
if device == 'cuda':
    env.world.enable_cuda_graph()
    env.reset(torch.ones(args.envs, dtype=torch.bool, device=device))
policy = Policy().to(device)
if args.init:
    policy.load_state_dict(torch.load(args.init, map_location=device)['policy'])
    print(f'loaded {args.init}')
with torch.no_grad():
    policy.log_std.fill_(args.init_log_std)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
os.makedirs(os.path.dirname(args.out), exist_ok=True)

H, E = args.horizon, args.envs
obs = env.observe()
stack = FrameStack(obs['rgb'])
proprio = obs['proprio']
# rollout buffers, (H, E, ...)
buf_frames = torch.zeros(H, E, *stack.frames.shape[1:], dtype=torch.uint8, device=device)
buf_proprio = torch.zeros(H, E, proprio.shape[1], device=device)
buf_action = torch.zeros(H, E, policy.mean.out_features, device=device)
buf_logp = torch.zeros(H, E, device=device)
buf_value = torch.zeros(H, E, device=device)
buf_reward = torch.zeros(H, E, device=device)
buf_done = torch.zeros(H, E, device=device)
# running discounted return per env and its running variance, for reward scaling
ret = torch.zeros(E, device=device)
ret_var, ret_count = 1.0, 1e-4
best = -1.0
start = time.time()
for it in range(1, args.iters + 1):
    finished, succeeded = 0, 0
    with torch.no_grad():
        for t in range(H):
            mean, value = policy(stack.frames, proprio)
            dist = torch.distributions.Normal(mean, policy.log_std.exp())
            action = dist.sample()
            buf_frames[t], buf_proprio[t], buf_action[t] = stack.frames, proprio, action
            buf_logp[t], buf_value[t] = dist.log_prob(action).sum(-1), value
            obs, reward, done, info = env.step(action)
            proprio = obs['proprio']
            stack.push(obs['rgb'], done)
            # reward scaling: divide by the running std of the discounted return (Engstrom et al. 2020)
            ret = ret * args.gamma * (~done) + reward
            batch_var, n = ret.var().item(), E
            total = ret_count + n
            ret_var = (ret_var * ret_count + batch_var * n) / total
            ret_count = total
            buf_reward[t] = reward / (ret_var ** 0.5 + 1e-8)
            buf_done[t] = done.float()
            finished += int(done.sum())
            succeeded += int(info['success'].sum())
        _, next_value = policy(stack.frames, proprio)
        # generalized advantage estimation, episodes end (no bootstrap) where done
        adv = torch.zeros(H, E, device=device)
        gae = torch.zeros(E, device=device)
        for t in reversed(range(H)):
            nv = next_value if t == H - 1 else buf_value[t + 1]
            delta = buf_reward[t] + args.gamma * nv * (1 - buf_done[t]) - buf_value[t]
            gae = delta + args.gamma * args.gae_lambda * (1 - buf_done[t]) * gae
            adv[t] = gae
        returns = adv + buf_value

    # PPO update over the flattened (H*E) batch
    flat = lambda x: x.reshape(H * E, *x.shape[2:])
    f_frames, f_proprio, f_action, f_logp, f_adv, f_ret = map(flat, (buf_frames, buf_proprio, buf_action, buf_logp, adv, returns))
    f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
    for _ in range(args.epochs):
        for mb in torch.randperm(H * E, device=device).split(args.minibatch):
            mean, value = policy(f_frames[mb], f_proprio[mb])
            dist = torch.distributions.Normal(mean, policy.log_std.exp())
            ratio = (dist.log_prob(f_action[mb]).sum(-1) - f_logp[mb]).exp()
            policy_loss = -torch.minimum(ratio * f_adv[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * f_adv[mb]).mean()
            value_loss = ((value - f_ret[mb]) ** 2).mean()
            loss = policy_loss + args.value_coef * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
    print(f'iter {it} episodes {finished} success {succeeded / max(finished, 1):.0%} policy {policy_loss.item():.3f} '
          f'value {value_loss.item():.3f} std {policy.log_std.exp().mean().item():.3f} ({time.time() - start:.0f}s)')
    if it % args.eval_every == 0 or it == args.iters:
        success = evaluate(policy, args.eval_envs, device)
        print(f'iter {it} eval success {success:.0%}')
        if success >= best:
            best = success
            torch.save({'policy': policy.state_dict(), 'iter': it, 'success': success}, args.out)
            print(f'saved {args.out}')

# Evaluate a policy checkpoint and record one episode for the frontend's playback mode.
#
# run from back/:  python -m ml.eval --ckpt checkpoints/ppo.pt
# prints the success rate, writes logs/{--record}/ (default "policy_rollout")
import argparse
import torch
from ml.policy import Policy
from ml.rollout import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/ppo.pt')
parser.add_argument('--envs', type=int, default=64)
parser.add_argument('--blocks', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--record', default='policy_rollout')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = Policy().to(device)
policy.load_state_dict(torch.load(args.ckpt, map_location=device)['policy'])
success = evaluate(policy, args.envs, device, num_blocks=args.blocks, seed=args.seed, record_name=args.record)
print(f'{args.ckpt}: success {success:.0%} over {args.envs} episodes, recording in logs/{args.record}/')

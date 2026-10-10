# Evaluate a policy checkpoint and record one episode for the frontend's playback mode.
#
# run from back/:  python -m ml.eval --task race --ckpt checkpoints/ppo.pt
# prints the metrics (pick: success rate; race: win rate and scores vs the scripted expert), writes logs/{--record}/
import argparse
import torch
from ml.policy import Policy
from ml.rollout import evaluate
from ml.tasks import TASKS

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/ppo.pt')
parser.add_argument('--task', default='race', choices=TASKS)
parser.add_argument('--envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--record', default='policy_rollout')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = Policy().to(device)
policy.load_state_dict(torch.load(args.ckpt, map_location=device)['policy'])
metrics = evaluate(policy, args.task, args.envs, device, seed=args.seed, record_name=args.record)
print(f'{args.ckpt}: {metrics} over {args.envs} episodes, recording in logs/{args.record}/')

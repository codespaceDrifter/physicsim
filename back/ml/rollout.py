# Run a policy in a task and measure it, optionally recording env 0 for the frontend's playback.
#   pick: the policy drives the robot; metric "success" = fraction of episodes delivered
#   race: the policy drives robot 0 against the scripted expert on robot 1; metrics "win", "score" (policy's mean
#         final score), "expert_score"
import numpy as np
import torch
from robot.env import DT, ACTION_REPEAT
from ml.policy import FrameStack
from ml.expert import Expert
from ml.tasks import make_env
from recording import Recorder


@torch.no_grad()
def evaluate(policy, task, num_envs, device, seed=123, record_name=None):
    # run every env for one full episode with the deterministic (mean) action. returns {metric: float}
    env = make_env(task, num_envs, device, seed=seed)
    if str(device).startswith('cuda'):
        env.enable_cuda_graph()
    expert = Expert(num_envs, env.R)
    policy.eval()
    E, R = env.E, env.R
    obs = env.observe()
    stack = FrameStack(obs['rgb'].flatten(0, 1))
    # (E,) episode finished; per finished env its final info
    finished = torch.zeros(E, dtype=torch.bool, device=device)
    success = torch.zeros(E, dtype=torch.bool, device=device)
    final_score = torch.zeros(E, R, dtype=torch.long, device=device)
    # [(time, (N, 7) poses)] env 0's episode, written as a recording at the end
    frames0 = []
    for t in range(env.max_steps):
        if record_name and not finished[0]:
            frames0.append((t * DT * ACTION_REPEAT, env.world.poses()[0].cpu().numpy()))
        # (E*R, A) policy actions for every robot
        action, _ = policy(stack.frames, obs['proprio'].flatten(0, 1))
        action = action.view(E, R, -1)
        if task == 'race':
            # robot 1 is the scripted expert
            action[:, 1] = env.targets_to_action(torch.tensor(expert.act(env, robots=[1]), device=device))[:, 1]
        obs, reward, done, info = env.step(action)
        for e in done.nonzero().view(-1).tolist():
            expert.reset(e)
        newly = done & ~finished
        if task == 'pick':
            success |= info['success'] & newly
        else:
            final_score = torch.where(newly[:, None], info['score'], final_score)
        finished |= done
        stack.push(obs['rgb'].flatten(0, 1), done.repeat_interleave(R))
        if finished.all():
            break
    if record_name:
        rec = Recorder(record_name, env.world.scene, DT * ACTION_REPEAT, len(frames0))
        for time, poses in frames0:
            rec.add(time, poses)
    policy.train()
    if task == 'pick':
        return {'success': success.float().mean().item()}
    s = final_score.float()
    return {'win': (s[:, 0] > s[:, 1]).float().mean().item(), 'score': s[:, 0].mean().item(), 'expert_score': s[:, 1].mean().item()}

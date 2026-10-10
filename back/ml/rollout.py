# Run a policy in the pick-and-place env: success rate, optionally recording env 0 for the frontend's playback.
import torch
from robot.pick_place import PickPlaceEnv, DT, ACTION_REPEAT
from ml.policy import FrameStack
from recording import Recorder


@torch.no_grad()
def evaluate(policy, num_envs, device, num_blocks=1, seed=123, record_name=None):
    # run every env for one full episode with the deterministic (mean) action. returns success rate in [0, 1].
    # record_name: also save env 0's episode to logs/{record_name}/ for playback
    env = PickPlaceEnv(num_envs, device, num_blocks=num_blocks, seed=seed)
    if str(device).startswith('cuda'):
        env.enable_cuda_graph()
    policy.eval()
    obs = env.observe()
    stack = FrameStack(obs['rgb'])
    # (E,) episode finished, (E,) it was a success
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # [(time, (N, 7) poses)] env 0's episode, written as a recording at the end
    frames0 = []
    for t in range(env.max_steps):
        if record_name and not finished[0]:
            frames0.append((t * DT * ACTION_REPEAT, env.world.poses()[0].cpu().numpy()))
        action, _ = policy(stack.frames, obs['proprio'])
        obs, reward, done, info = env.step(action)
        success |= info['success'] & ~finished
        finished |= done
        stack.push(obs['rgb'], done)
    if record_name:
        rec = Recorder(record_name, env.world.scene, DT * ACTION_REPEAT, len(frames0))
        for time, poses in frames0:
            rec.add(time, poses)
    policy.train()
    return success.float().mean().item()

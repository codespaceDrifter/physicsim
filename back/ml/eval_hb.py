# Official HumanoidBench score of a policy trained on robot/hb_course.py: runs it in HumanoidBench's own environment
# (its pinned MuJoCo on CPU, gymnasium registration g1-{task}-v0), mean actions, --episodes per task; prints each
# task's mean episode return and the fraction of episodes reaching the task's success_bar. the policy's inputs are
# built from the environment's observation (qpos + qvel) the way robot/hb_course.py builds them: the robot's part as
# proprio, the objects' part padded into the task vector after the task one-hot and the gait clock.
#
# --record: the first episode of each task is also written as logs/hb_{task}.simrec for the frontend's playback (every
# visible geom's pose each control step, robot/mjcf.py geom_scene; the meshes are robot/assets/mujoco/, written by
# write_hb_scenes.py), info {"task", "checkpoint", "return"}.
#
# run from back/ with the HumanoidBench environment (~/hbenv):
#   python -m ml.eval_hb --ckpt checkpoints/hb_v4.pt --tasks stand walk --episodes 10 [--record]
import argparse
import gymnasium as gym
import numpy as np
import torch
import humanoid_bench
from ml.humanoid_policy import Policy, HISTORY
from recording import Recorder
from robot.mjcf import geom_scene, geom_poses

# robot/hb_course.py's tasks (STAGES order) with their gait clock, sizes and PD layer (that module needs MuJoCo Warp,
# which this environment doesn't have)
GAIT_FREQ = {'stand': 0.0, 'walk': 1.25, 'run': 3.0, 'crawl': 0.0, 'stair': 1.25, 'slide': 1.25, 'hurdle': 3.0,
             'sit_simple': 0.0, 'sit_hard': 0.0, 'balance_simple': 0.0, 'balance_hard': 0.0, 'pole': 1.0}
STAGES = list(GAIT_FREQ)
ROBOT_NQ, ROBOT_NV, EXTRA_DIM = 44, 43, 26
PROPRIO_DIM, TASK_DIM, AUX_DIM, ACTION_DIM = ROBOT_NQ + ROBOT_NV, len(STAGES) + 2 + EXTRA_DIM, 4, 37
PD_SCALE, PD_DAMPING = 0.5, 1 / 40

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='checkpoints/hb_v4.pt')
parser.add_argument('--tasks', nargs='+', default=STAGES, choices=STAGES)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--record', action='store_true', help='also record each task\'s first episode (see the header)')
args = parser.parse_args()

ck = torch.load(args.ckpt, map_location='cpu')
assert ck['args']['backend'] == 'humanoidbench'
policy = Policy(PROPRIO_DIM, ACTION_DIM, AUX_DIM, TASK_DIM, 0, False, len(STAGES), ck['args']['width'], ck['args']['layers'], ck['args']['group'])
policy.load_state_dict(ck['policy'])
# left in train mode (dropout is 0, the same function): eval mode takes torch's fused CPU transformer path, which
# fails under the policy's bfloat16 autocast
print(f'{args.ckpt} (iter {ck["iter"]})')
for task in args.tasks:
    env = gym.make(f'g1-{task}-v0', render_mode=None)
    bar = env.unwrapped.task.success_bar
    m, d = env.unwrapped.model, env.unwrapped.data
    # (A,) qpos / qvel index of each motor's joint, its keyframe angle (the PD layer, robot/hb_course.py)
    joints = m.actuator_trnid[:, 0]
    act_q, act_v = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    key_joints = (m.key_qpos[0] if m.nkey else m.qpos0)[act_q]
    # seconds per control step
    dt = m.opt.timestep * env.unwrapped.frame_skip

    def inputs(obs, step):
        # the environment's observation (nq + nv,) at the episode's step -> (1, PROPRIO_DIM) proprio, (1, TASK_DIM) task
        proprio = np.concatenate([obs[:ROBOT_NQ], obs[m.nq:m.nq + ROBOT_NV]])
        extra = np.concatenate([obs[ROBOT_NQ:m.nq], obs[m.nq + ROBOT_NV:]])
        phase = step * dt * GAIT_FREQ[task] % 1.0
        clock = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]) * float(GAIT_FREQ[task] > 0)
        one_hot = np.eye(len(STAGES))[STAGES.index(task)]
        command = np.concatenate([one_hot, clock, np.pad(extra, (0, EXTRA_DIM - len(extra)))])
        return torch.tensor(proprio, dtype=torch.float32)[None], torch.tensor(command, dtype=torch.float32)[None]

    returns = []  # [float] per episode
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        rec = None
        if args.record and episode == 0:
            scene, _, geoms = geom_scene(m, d, f'hb_{task}')
            rec = Recorder(f'hb_{task}', scene, dt)
            rec.add(0.0, geom_poses(m, d, geoms))
        proprio, command = inputs(obs, 0)
        # (1, HISTORY, PROPRIO_DIM) the last HISTORY proprio readings, oldest first (filled with the first at the start)
        history = proprio[:, None].repeat(1, HISTORY, 1)
        ret, done, step = 0.0, False, 0
        while not done:
            with torch.no_grad():
                mean, _ = policy(history, command)
            action = np.clip(key_joints + PD_SCALE * mean[0].numpy() - d.qpos[act_q] - PD_DAMPING * d.qvel[act_v], -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            step += 1
            done = terminated or truncated
            proprio, command = inputs(obs, step)
            history = torch.cat([history[:, 1:], proprio[:, None]], 1)
            if rec is not None:
                rec.add(step * dt, geom_poses(m, d, geoms))
        if rec is not None:
            rec.save({'task': task, 'checkpoint': args.ckpt, 'return': ret})
        returns.append(ret)
    returns = np.array(returns)
    print(f'{task}: mean return {returns.mean():.1f} (min {returns.min():.1f}, max {returns.max():.1f}), '
          f'success (return >= {bar}) {(returns >= bar).mean():.2f} over {args.episodes} episodes')
    env.close()

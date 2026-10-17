# Drivers (who controls the robots) and evaluation, shared by distillation, GRPO, ml/eval.py and the live server.
# a driver is called once per control step: driver(env, fresh) -> (E, R, A) actions in [-1, 1] for every robot,
# fresh (E,) = envs whose episode just started. fresh.all() (the first call on an env) (re)allocates its buffers.
import torch
from robot.course import CourseEnv, STAGES
from ml.expert import Expert
from ml.vla import CHUNK, tokenize


class TeacherDriver:
    # the teacher's mean action from the privileged state
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, env, fresh):
        return self.model(env.state())[0]


class VLADriver:
    # the VLA from cameras + command + proprio: plans CHUNK actions per robot, executes the first `execute` of them, then
    # plans again (and always on an episode's first step). sample: gaussian around the means (GRPO), else the means.
    # execute = 1 (receding horizon, replan every step) is the default: the 96 px student brought 3.6 speedrun blocks
    # home with it vs 3.1 executing whole chunks. GRPO samples whole chunks (execute = CHUNK), one decision per chunk
    # after each call, self.plan holds the decisions just made, for GRPO:
    # {"envs": (n,) envs that planned, "rgb": (n, R, 6, 96, 96) uint8, "words": (n, R, MAX_WORDS), "proprio": (n, R, P),
    #  "chunk": (n, R, CHUNK, A) actions to execute} or None if nobody planned
    def __init__(self, model, sample=False, execute=1):
        assert 1 <= execute <= CHUNK
        self.model = model
        self.sample = sample
        self.execute = execute

    @torch.no_grad()
    def __call__(self, env, fresh, obs=None):
        # obs: this step's env.observe() if the caller already rendered it
        E, R = env.E, env.R
        if fresh.all():
            # (E, R, CHUNK, A) current plan, (E,) next action in it
            self.chunk = torch.zeros(E, R, CHUNK, self.model.head.out_features, device=env.device)
            self.idx = torch.zeros(E, dtype=torch.long, device=env.device)
        replan = fresh | (self.idx >= self.execute)
        self.plan = None
        if replan.any():
            envs = replan.nonzero().view(-1)
            obs = env.observe() if obs is None else obs
            rgb, proprio = obs['rgb'][envs], obs['proprio'][envs]
            words = tokenize([env.command(e, r) for e in envs.tolist() for r in range(R)], env.device).view(len(envs), R, -1)
            # (n * R, CHUNK, A) -> (n, R, CHUNK, A)
            chunk = self.model(rgb.flatten(0, 1), words.flatten(0, 1), proprio.flatten(0, 1)).view(len(envs), R, CHUNK, -1)
            if self.sample:
                chunk = chunk + self.model.log_std.exp() * torch.randn_like(chunk)
            self.chunk[envs] = chunk
            self.idx[envs] = 0
            self.plan = {'envs': envs, 'rgb': rgb, 'words': words, 'proprio': proprio, 'chunk': chunk}
        action = self.chunk[torch.arange(E, device=env.device), :, self.idx]
        self.idx += 1
        return action


class ExpertDriver:
    # the scripted expert (ml/expert.py), joint targets converted to actions. it fetches blocks, so it idles in drive.
    # it has no memory, so it also labels states anyone else reached (DAgger)
    def __call__(self, env, fresh):
        if fresh.all():
            self.expert = Expert(env.E, env.R)
        return env.targets_to_action(torch.tensor(self.expert.act(env), device=env.device))


@torch.no_grad()
def evaluate(driver, opponent, stage, num_envs, device, seed=123):
    # one episode in each of num_envs envs of this stage. the driver drives every robot, except robot 1 in the race,
    # which the opponent drives. returns {"success": fraction of (active, driver-driven) robot-episodes that succeeded
    # (race: robot 0 won), "score": robot 0's mean final score, "opponent_score": robot 1's (race)}
    env = CourseEnv(num_envs, device, stage_probs=[float(s == stage) for s in STAGES], seed=seed)
    if str(device).startswith('cuda'):
        env.enable_cuda_graph()
    E, R = env.E, env.R
    race = stage == 'race'
    # (E, R) robots counted: active ones, and only robot 0 in the race
    counted = env.active()
    counted[:, 1] &= not race
    finished = torch.zeros(E, dtype=torch.bool, device=device)
    success = torch.zeros(E, R, dtype=torch.bool, device=device)
    score = torch.zeros(E, R, device=device)
    fresh = torch.ones(E, dtype=torch.bool, device=device)
    for _ in range(int(env.stage_steps.max())):
        action = driver(env, fresh)
        if race:
            action[:, 1] = opponent(env, fresh)[:, 1]
        reward, done, info = env.step(action)
        newly = done & ~finished
        success |= info['success'] & newly[:, None]
        score = torch.where(newly[:, None], info['score'].float(), score)
        finished |= done
        fresh = done
        if finished.all():
            break
    assert finished.all()
    return {'success': (success & counted).sum().item() / counted.sum().item(), 'score': score[:, 0].mean().item(),
            'opponent_score': score[:, 1].mean().item()}

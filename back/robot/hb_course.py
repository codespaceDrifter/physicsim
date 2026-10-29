# HumanoidBench (github.com/carlosferrazza/humanoid-bench) on MuJoCo Warp, its G1 (torque control, Dex3 hands), as one
# multi-task course for ml/train_humanoid.py (--backend humanoidbench): every task in STAGES trains at once, one
# policy. a task's scene is its own Warp model (assets/envs/g1_torque_{scene}.xml, unchanged; stand / walk / run share
# one), holding a fixed share of the envs; everything the benchmark scores is reproduced exactly:
#   physics     the scene file: 2 ms Newton steps (100 iterations, 50 line search), 10 per action
#   action      (37,) in [-1, 1] (clipped), scaled to each motor's torque range (tasks.py unnormalize_action). the
#               agent computes it with a PD layer (part of the agent, the benchmark still gets torques): the policy
#               outputs joint target offsets a around the keyframe, the torque is kp (target - q) - kd qdot with
#               kp = the motor's torque limit per rad (hips 88, knees 139, ankles 40: near unitree_rl_gym's G1 gains
#               100 / 150 / 40) and kd = kp * PD_DAMPING, so the benchmark action (torque / limit; the ranges are
#               symmetric) is clamp(q0 + PD_SCALE a - q - PD_DAMPING qdot, -1, 1). with raw torques the policy stood
#               and never found a gait: its noise was +-half of full torque on every joint each step
#   observation qpos and qvel of the whole model (tasks.py get_obs): the robot's (87,) is the proprio, the rest (the
#               scene's free objects) goes in the task vector, padded to EXTRA_DIM
#   reset       the scene's keyframe (pole: the model's qpos0) + uniform(-0.01, 0.01) on every qpos, zero velocity (env.py reset_model), then the
#               task's own (sit_hard: robot at a random spot and heading in front of the chair)
#   reward      the task's get_reward (envs/basic_locomotion_envs.py, balance.py, pole.py), dm_control's tolerance()
#               ported to torch (rewards cross-checked against HumanoidBench's on the same states)
#   end         the task's get_terminated, or 1000 steps. a world whose state diverged (non-finite qpos, |qvel| >= 1e4: MuJoCo
#               on CPU resets such a simulation itself, mj_checkAcc; Warp carries the NaN on, and one poisoned the
#               policy's normalizers) ends too, as a failure with reward 0
# gait clock (tasks that walk): the agent keeps a phase advancing at GAIT_FREQ from the episode's step count (its own
# clock: the official evaluation, ml/eval_hb.py, runs it too), sees its sin / cos in the task vector, and training adds
# GAIT_W x the feet_phase reward of our courses (each foot site's height against a swing reference, robot/humanoid.py)
# to the benchmark's: from the benchmark reward alone the policy stood still on walk (0.18 per step, standing scores
# 0.17) and never found a gait (unitree_rl_gym's G1 trains its gait the same way). run at 2 Hz plateaued near 1.4 m/s
# (its bar needs ~3.8): 3 Hz for the fast tasks. the swing reference is measured from the terrain under each foot (the
# scene's height profile along x, raycast at start-up on a TERRAIN_DX grid with the robot out of the way; the courses
# are uniform across y): from world height 0 it asked for 8 cm of clearance under 18 cm stairs and counted a foot
# standing on a step as swinging, and stair / slide stood still at the start.
# domain randomization (training episodes; pass tests and the official evaluation run the exact benchmark): the official
# scores come from HumanoidBench's pinned MuJoCo 3.1.6 on CPU, this port follows MuJoCo 3.17 (Warp), and contact-rich
# tasks scored ~300 lower there (crawl: ~0.85 per step here, 511 returns there). so each training episode scales the PD
# layer's kp by U(KP_RANGE) and kd by U(KD_RANGE), and the base gets a random horizontal push of up to PUSH_SPEED m/s
# every PUSH_STEPS steps.
# pass test: test episodes (TEST_PROB, mean actions, not trained on) whose return reaches the task's success_bar; a task
# passes at PASS of them. training rewards are the benchmark's times REWARD_SCALE (returns of up to ~100 at gamma 0.99
# otherwise); returns are counted unscaled. stage_probs only gates which tasks train (> 0: in its scene, uniformly).
#
# the task vector: the task one-hot, the clock's sin / cos (0 without a gait), the scene's object state (padded).
# privileged (critic): proprio, task vector, head height, torso upright, center of mass velocity, actuator forces / 100.
# aux (state decoder targets): center of mass velocity, head height.
import math
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from robot.humanoid import TEST_PROB, SWING_HEIGHT

NAME = 'humanoidbench'
# per task: scene file, success_bar, termination, gait clock (steps per second, 0: none)
TASKS = {'stand': {'scene': 'walk', 'bar': 800.0, 'end': 'low_0.2', 'gait': 0.0},
         'walk': {'scene': 'walk', 'bar': 700.0, 'end': 'low_0.2', 'gait': 1.25},
         'run': {'scene': 'walk', 'bar': 700.0, 'end': 'low_0.2', 'gait': 3.0},
         'crawl': {'scene': 'crawl', 'bar': 700.0, 'end': 'never', 'gait': 0.0},
         'stair': {'scene': 'stair', 'bar': 700.0, 'end': 'tilt_0.1', 'gait': 1.25},
         'slide': {'scene': 'slide', 'bar': 700.0, 'end': 'tilt_0.1', 'gait': 1.25},
         'hurdle': {'scene': 'hurdle', 'bar': 700.0, 'end': 'low_0.2', 'gait': 3.0},
         'sit_simple': {'scene': 'sit_simple', 'bar': 750.0, 'end': 'low_0.5', 'gait': 0.0},
         'sit_hard': {'scene': 'sit_hard', 'bar': 750.0, 'end': 'low_0.5', 'gait': 0.0},
         'balance_simple': {'scene': 'balance_simple', 'bar': 800.0, 'end': 'balance', 'gait': 0.0},
         'balance_hard': {'scene': 'balance_hard', 'bar': 800.0, 'end': 'balance', 'gait': 0.0},
         'pole': {'scene': 'pole', 'bar': 700.0, 'end': 'low_0.5', 'gait': 1.0}}
STAGES = list(TASKS)
PASS = {t: 0.8 for t in STAGES}
# humanoid_bench checkout (cloned on the training machine)
HB_ENVS = Path.home() / 'humanoid-bench' / 'humanoid_bench' / 'assets' / 'envs'
# G1 (the tasks' __init__): head height (m) above which standing scores 1; crawl height (m)
STAND_HEIGHT = 1.28
CRAWL_HEIGHT = 0.6
EPISODE_STEPS = 1000
FRAME_SKIP = 10
RANDOMNESS = 0.01
REWARD_SCALE = 0.1
# PD layer (see the header): rad of joint target per unit action, kd / kp (s)
PD_SCALE = 0.5
PD_DAMPING = 1 / 40
# weight of feet_phase in the training reward (see the header)
GAIT_W = 0.5
# domain randomization (see the header): kp / kd scale ranges, push speed (m/s), steps between pushes
KP_RANGE, KD_RANGE = (0.85, 1.15), (0.7, 1.3)
PUSH_SPEED = 0.5
PUSH_STEPS = (200, 400)
# terrain profile (see the header): the scenes with one (uniform across y; elsewhere flat), x range (m), grid step
TERRAIN_SCENES = ('stair', 'slide', 'hurdle')
TERRAIN_X, TERRAIN_DX = (-5.0, 60.0), 0.02
# Warp buffers per world: contacts, constraint rows
NCONMAX = 128
NJMAX = 768
# no walk commands (the trainer's command curriculum is off)
COMMANDED = False
# robot qpos (44) + qvel (43); the scenes' object qpos + qvel, padded (balance_hard's two free bodies: 14 + 12)
ROBOT_NQ, ROBOT_NV = 44, 43
PROPRIO_DIM, EXTRA_DIM = ROBOT_NQ + ROBOT_NV, 26
TASK_DIM, AUX_DIM = len(STAGES) + 2 + EXTRA_DIM, 4
# the motors in the model's order: legs, torso, arms, then the hands' fingers
ACTUATORS = [f'{side}_{j}_joint' for side in ('left', 'right') for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')] + \
            ['torso_joint'] + [f'{side}_{j}_joint' for side in ('left', 'right') for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch', 'elbow_roll')] + \
            [f'{side}_{j}_joint' for side in ('left', 'right') for j in ('zero', 'one', 'two', 'three', 'four', 'five', 'six')]
ACTION_DIM = len(ACTUATORS)
PRIV_DIM = PROPRIO_DIM + TASK_DIM + 1 + 1 + 3 + ACTION_DIM


def tolerance(x, lo, hi, margin, sigmoid='gaussian', value_at_margin=0.1):
    # dm_control.utils.rewards.tolerance: 1 within [lo, hi], falling off outside over margin (> 0; to value_at_margin
    # at distance margin) by the sigmoid
    inside = (x >= lo) & (x <= hi)
    d = torch.where(x < lo, lo - x, x - hi).clamp_min(0) / margin
    if sigmoid == 'gaussian':
        scale = math.sqrt(-2 * math.log(value_at_margin))
        out = torch.exp(-0.5 * (d * scale) ** 2)
    elif sigmoid == 'linear':
        scaled = d * (1 - value_at_margin)
        out = torch.where(scaled.abs() < 1, 1 - scaled, 0.0)
    else:
        assert sigmoid == 'quadratic'
        scaled = d * math.sqrt(1 - value_at_margin)
        out = torch.where(scaled.abs() < 1, 1 - scaled ** 2, 0.0)
    return torch.where(inside, 1.0, out)


class Scene:
    # one scene file on Warp with its envs: the state views, the ids its tasks' rewards read, the PD layer's indices
    def __init__(self, name, tasks, num_envs, device):
        self.name, self.tasks, self.E, self.device = name, tasks, num_envs, device
        mjm = mujoco.MjModel.from_xml_path(str(HB_ENVS / f'g1_torque_{name}.xml'))
        assert [mjm.actuator(i).name for i in range(mjm.nu)] == ACTUATORS and mjm.nkey <= 1
        assert mjm.nq - ROBOT_NQ + mjm.nv - ROBOT_NV <= EXTRA_DIM
        self.mjm = mjm
        # (nq,) the start state: the scene's keyframe, or the model's qpos0 if it has none (pole; mj_resetDataKeyframe)
        start = mjm.key_qpos[0] if mjm.nkey else mjm.qpos0
        mjd = mujoco.MjData(mjm)
        # (G,) the terrain height along x (see the header): raycast down at y = 0 with the robot moved far away
        mjd.qpos[:] = start
        mjd.qpos[0] = -1000.0
        mujoco.mj_forward(mjm, mjd)
        xs = torch.arange(TERRAIN_X[0], TERRAIN_X[1], TERRAIN_DX)
        heights = []  # [float]
        for x in xs.tolist() if name in TERRAIN_SCENES else []:
            geom = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(mjm, mjd, np.array([x, 0.0, 10.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geom)
            heights.append(10.0 - dist if dist >= 0 else 0.0)
        self.terrain = torch.tensor(heights, dtype=torch.float32, device=device) if heights else torch.zeros(len(xs), device=device)
        mjd.qpos[:] = start
        mujoco.mj_forward(mjm, mjd)
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=NCONMAX, njmax=NJMAX)
        t, d = wp.to_torch, self.d
        # torch views of the warp state (shared memory: writes go straight into the simulation)
        self.qpos, self.qvel, self.ctrl = t(d.qpos), t(d.qvel), t(d.ctrl)
        self.xpos, self.xquat, self.xmat = t(d.xpos), t(d.xquat), t(d.xmat)
        self.site_xpos, self.sensordata, self.actuator_force = t(d.site_xpos), t(d.sensordata), t(d.actuator_force)
        self.contact_geom, self.contact_world, self.nacon = t(d.contact.geom), t(d.contact.worldid), t(d.nacon)
        self.key_qpos = torch.tensor(start, dtype=torch.float32, device=device)
        self.ctrl_lo = torch.tensor(mjm.actuator_ctrlrange[:, 0], dtype=torch.float32, device=device)
        self.ctrl_hi = torch.tensor(mjm.actuator_ctrlrange[:, 1], dtype=torch.float32, device=device)
        assert torch.equal(self.ctrl_lo, -self.ctrl_hi) and (mjm.actuator_gear[:, 0] == 1).all()
        # (A,) qpos / qvel index of each motor's joint (motor order isn't joint order), its keyframe angle
        joints = mjm.actuator_trnid[:, 0]
        self.act_q = torch.tensor(mjm.jnt_qposadr[joints], device=device)
        self.act_v = torch.tensor(mjm.jnt_dofadr[joints], device=device)
        self.key_joints = self.key_qpos[self.act_q]
        site = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, n)
        body = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)
        geom = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.head, self.imu = site('head'), site('imu')
        self.feet = torch.tensor([site('left_foot'), site('right_foot')], device=device)
        self.foot_rest = torch.tensor(mjd.site_xpos[self.feet.cpu().numpy(), 2], dtype=torch.float32, device=device)
        self.torso, self.pelvis = body('torso_link'), body('pelvis')
        self.com_vel_adr = int(mjm.sensor_adr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, 'pelvis_subtreelinvel')])
        assert min(self.head, self.imu, self.torso, self.pelvis) >= 0
        # task-specific ids (-1 where the scene lacks them): hurdle walls, the chair, balance's sphere / board, the poles
        self.walls = torch.tensor([geom(f'{s}_barrier_collision') for s in ('left', 'right', 'behind')], device=device)
        self.chair = body('chair')
        self.sphere, self.board = geom('pivot_sphere_collision'), geom('stand_board_collision')
        self.poles = torch.tensor([g for g in range(mjm.ngeom) if 'pole_r' in (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or '')], dtype=torch.long, device=device)
        # (len(tasks),) the scene's task indices and their target speeds (Walk / Stand / Run), crawl's pelvis quaternion
        self.task_ids = torch.tensor(tasks, device=device)
        self.speeds = torch.tensor([{'stand': 0.0, 'walk': 1.0, 'run': 5.0}.get(STAGES[i], 0.0) for i in tasks], device=device)
        self.crawl_quat = torch.tensor([0.75, 0.0, 0.65, 0.0], device=device)
        # the physics as a CUDA graph on the scene's own stream (enable_cuda_graph): scenes are latency bound (40-60 ms a
        # step whether 341 or 1027 envs), so they run concurrently
        self.graph, self.stream = None, None

    def physics(self):
        # FRAME_SKIP physics steps, then kinematics / sensors of the new state (for observing)
        for _ in range(FRAME_SKIP):
            mjw.step(self.m, self.d)
        mjw.forward(self.m, self.d)

    def contacts(self):
        # (C, 2) geom pairs, (C,) their world, of this step's active contacts
        n = int(self.nacon[0])
        return self.contact_geom[:n], self.contact_world[:n]

    def any_contact(self, mask):
        # mask (C,) bool over contacts() -> (E,) envs with any such contact
        _, world = self.contacts()
        return torch.zeros(self.E, dtype=torch.bool, device=self.device).index_fill_(0, world[mask].long(), True)


class HbCourse:
    NAME, STAGES = NAME, STAGES

    # num_envs worlds on device (cuda), stage_probs (S,): tasks with > 0 train, vision: unused (state-based), seed
    def __init__(self, num_envs, device, stage_probs, vision=False, seed=0):
        assert not vision
        self.R, self.A = 1, ACTION_DIM
        self.device = device
        self.vision = False
        self.gen = torch.Generator(device=device).manual_seed(seed)
        # scenes in first-task order, each with envs in proportion to its tasks
        names = list(dict.fromkeys(TASKS[t]['scene'] for t in STAGES))  # [str]
        tasks = {n: [i for i, t in enumerate(STAGES) if TASKS[t]['scene'] == n] for n in names}  # {scene: [task index]}
        counts = [num_envs * len(tasks[n]) // len(STAGES) for n in names]
        counts[0] += num_envs - sum(counts)
        self.scenes = [Scene(n, tasks[n], c, device) for n, c in zip(names, counts)]
        self.E = num_envs
        # (E,) each env's scene index, (scenes + 1,) env offsets
        self.offsets = [0]
        for c in counts:
            self.offsets.append(self.offsets[-1] + c)
        self.stage_probs = torch.tensor(stage_probs, dtype=torch.float32, device=device)
        self.bar = torch.tensor([TASKS[t]['bar'] for t in STAGES], device=device)
        self.freq = torch.tensor([TASKS[t]['gait'] for t in STAGES], device=device)
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)
        # per env: task, steps, unscaled return so far, pass-test flag
        self.stage, self.steps, self.ret = zeros(num_envs, dtype=torch.long), zeros(num_envs, dtype=torch.long), zeros(num_envs)
        self.test = zeros(num_envs, dtype=torch.bool)
        # per env: PD gain scales (1 in pass tests), steps to the next push
        self.kp_scale, self.kd_scale = torch.ones(num_envs, 1, device=device), torch.ones(num_envs, 1, device=device)
        self.push_timer = zeros(num_envs, dtype=torch.long)
        self.test_prob = TEST_PROB
        self.auto_reset = True
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def enable_cuda_graph(self):
        # each scene's physics as a CUDA graph on its own stream (capture warms up: re-reset after); torch and warp
        # share one non-default main stream
        self.main = torch.cuda.Stream()
        torch.cuda.set_stream(self.main)
        wp.set_stream(wp.stream_from_torch(self.main))
        for sc in self.scenes:
            sc.stream = torch.cuda.Stream()
            sc.stream.wait_stream(self.main)
            with wp.ScopedStream(wp.stream_from_torch(sc.stream)):
                sc.physics()
                with wp.ScopedCapture() as capture:
                    sc.physics()
            sc.graph = capture.graph
            self.main.wait_stream(sc.stream)
        self.reset(torch.ones(self.E, dtype=torch.bool, device=self.device))

    def simulate(self):
        # every scene's physics: concurrently on their streams after the main stream's writes, the main stream waiting
        # for all (sequentially without the graphs)
        if self.scenes[0].graph is None:
            for sc in self.scenes:
                sc.physics()
            return
        for sc in self.scenes:
            sc.stream.wait_stream(self.main)
            wp.capture_launch(sc.graph, stream=wp.stream_from_torch(sc.stream))
        for sc in self.scenes:
            self.main.wait_stream(sc.stream)

    def reset(self, mask):
        # new episodes where mask (E,): a trained task of the env's scene, the keyframe + uniform(-RANDOMNESS,
        # RANDOMNESS) per qpos at rest, then the task's own reset
        rand = lambda *shape: torch.rand(*shape, generator=self.gen, device=self.device)
        for sc, lo, hi in zip(self.scenes, self.offsets, self.offsets[1:]):
            envs = mask[lo:hi].nonzero().view(-1)
            n = len(envs)
            if n == 0:
                continue
            # (len(tasks),) the scene's tasks that train, uniformly
            probs = (self.stage_probs[sc.tasks] > 0).float()
            assert probs.sum() > 0, f'scene {sc.name}: none of its tasks trains'
            task = sc.task_ids[torch.multinomial(probs, n, replacement=True, generator=self.gen)]
            self.stage[lo + envs] = task
            q = sc.key_qpos + (rand(n, sc.key_qpos.shape[0]) * 2 - 1) * RANDOMNESS
            if sc.name == 'sit_hard':
                # SitHard.reset_model: the robot 0.2-0.4 m ahead, +-0.15 m aside, heading +-1.8 rad
                yaw = (rand(n) * 2 - 1) * 1.8
                q[:, 0] = 0.2 + 0.2 * rand(n)
                q[:, 1] = (rand(n) * 2 - 1) * 0.15
                q[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
            sc.qpos[envs] = q
            sc.qvel[envs] = 0.0
            mjw.forward(sc.m, sc.d)
        self.steps[mask] = 0
        self.ret[mask] = 0.0
        self.test[mask] = rand(self.E)[mask] < self.test_prob
        # domain randomization for training episodes (see the header)
        kp = KP_RANGE[0] + (KP_RANGE[1] - KP_RANGE[0]) * rand(self.E, 1)
        kd = KD_RANGE[0] + (KD_RANGE[1] - KD_RANGE[0]) * rand(self.E, 1)
        train = (mask & ~self.test)[:, None]
        self.kp_scale = torch.where(mask[:, None], torch.where(train, kp, 1.0), self.kp_scale)
        self.kd_scale = torch.where(mask[:, None], torch.where(train, kd, 1.0), self.kd_scale)
        self.push_timer = torch.where(mask, torch.randint(*PUSH_STEPS, (self.E,), generator=self.gen, device=self.device), self.push_timer)

    # ---- observations ----

    def gather(self, fn):
        # fn(scene) -> (scene E, ...) tensor, concatenated over scenes -> (E, ...)
        return torch.cat([fn(sc) for sc in self.scenes], 0)

    def clock(self):
        # (E,) gait phase in [0, 1), (E, 2) its sin / cos (0 in tasks without a gait)
        freq = self.freq[self.stage]
        phase = (self.steps * self.scenes[0].mjm.opt.timestep * FRAME_SKIP * freq) % 1.0
        return phase, torch.stack([torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], -1) * (freq > 0)[:, None]

    def observe(self, noise=True):
        # {"proprio", "command", "privileged", "aux"}, each (E, 1, dim): see the header (the benchmark has no
        # observation noise: noise is ignored)
        proprio = self.gather(lambda sc: torch.cat([sc.qpos[:, :ROBOT_NQ], sc.qvel[:, :ROBOT_NV]], -1))
        # (E, EXTRA_DIM) the scene's object qpos + qvel, zero padded
        extra = self.gather(lambda sc: torch.nn.functional.pad(torch.cat([sc.qpos[:, ROBOT_NQ:], sc.qvel[:, ROBOT_NV:]], -1),
                                                               (0, EXTRA_DIM - (sc.mjm.nq - ROBOT_NQ + sc.mjm.nv - ROBOT_NV))))
        _, clock = self.clock()
        task = torch.cat([(self.stage[:, None] == torch.arange(len(STAGES), device=self.device)).float(), clock, extra], -1)
        head = self.gather(lambda sc: sc.site_xpos[:, sc.head, 2:3])
        upright = self.gather(lambda sc: sc.xmat[:, sc.torso, 2, 2:3])
        vel = self.gather(lambda sc: sc.sensordata[:, sc.com_vel_adr:sc.com_vel_adr + 3])
        force = self.gather(lambda sc: sc.actuator_force)
        priv = torch.cat([proprio, task, head, upright, vel, force / 100.0], -1)
        out = {'proprio': proprio, 'command': task, 'privileged': priv, 'aux': torch.cat([vel, head], -1)}
        return {k: v[:, None] for k, v in out.items()}

    # ---- rewards ----

    def scene_reward(self, sc, task):
        # (scene E,) the benchmark reward of each env's task, (scene E,) terminated (envs/*.py get_reward,
        # get_terminated); task (scene E,) its task index
        head = sc.site_xpos[:, sc.head, 2]
        upright = sc.xmat[:, sc.torso, 2, 2]
        vel = sc.sensordata[:, sc.com_vel_adr:sc.com_vel_adr + 3]
        z = sc.qpos[:, 2]
        small_control = (4 + tolerance(sc.actuator_force, 0.0, 0.0, 10.0, 'quadratic', 0.0).mean(-1)) / 5
        dont_move = tolerance(vel[:, :2], 0.0, 0.0, 2.0).mean(-1)
        stand = tolerance(head, STAND_HEIGHT, math.inf, STAND_HEIGHT / 4) * tolerance(upright, 0.9, math.inf, 1.9, 'linear', 0.0)
        move = lambda speed: (5 * tolerance(vel[:, 0], speed, math.inf, speed, 'linear', 0.0) + 1) / 6
        name = STAGES[sc.tasks[0]] if sc.name != 'walk' else None
        if sc.name == 'walk':
            # Walk / Stand / Run: per env by its task
            s = sc.speeds[torch.searchsorted(sc.task_ids, task)]
            moving = (5 * tolerance(vel[:, 0], s, math.inf, s.clamp_min(1e-6), 'linear', 0.0) + 1) / 6
            reward = small_control * stand * torch.where(s == 0, dont_move, moving)
        elif name == 'crawl':
            imu = sc.site_xpos[:, sc.imu]
            crawling = tolerance(imu[:, 2], CRAWL_HEIGHT - 0.2, CRAWL_HEIGHT + 0.2, 1.0)
            crawling_head = tolerance(head, CRAWL_HEIGHT - 0.2, CRAWL_HEIGHT + 0.2, 1.0)
            pose = tolerance(torch.linalg.norm(sc.xquat[:, sc.pelvis] - sc.crawl_quat, dim=-1), 0.0, 0.0, 1.0)
            # tolerance with margin 0: inside the tunnel's width
            in_tunnel = ((imu[:, 1] >= -1.0) & (imu[:, 1] <= 1.0)).float()
            reward = (0.1 * small_control + 0.25 * torch.minimum(crawling, crawling_head) + 0.4 * move(1.0) + 0.25 * pose) * in_tunnel
        elif name in ('stair', 'slide'):
            feet = sc.site_xpos[:, sc.feet, 2]
            standing = tolerance(head - feet[:, 0], 1.2, math.inf, 0.45) * tolerance(head - feet[:, 1], 1.2, math.inf, 0.45)
            reward = standing * tolerance(upright, 0.5, math.inf, 1.9, 'linear', 0.0) * small_control * move(1.0)
        elif name == 'hurdle':
            geom, _ = sc.contacts()
            hit = sc.any_contact(torch.isin(geom, sc.walls).any(-1))
            stand_h = tolerance(head, STAND_HEIGHT, math.inf, STAND_HEIGHT / 4) * tolerance(upright, 0.8, math.inf, 1.9, 'linear', 0.0)
            reward = small_control * stand_h * move(5.0) * torch.where(hit, 0.1, 1.0)
        elif name in ('sit_simple', 'sit_hard'):
            chair = sc.xpos[:, sc.chair]
            sitting = tolerance(z, 0.68, 0.72, 0.2)
            on_chair = tolerance(sc.qpos[:, 0] - chair[:, 0], -0.19, 0.19, 0.2) * tolerance(sc.qpos[:, 1] - chair[:, 1], 0.0, 0.0, 0.1)
            posture = tolerance(head - sc.site_xpos[:, sc.imu, 2], 0.35, 0.45, 0.3)
            sit = (0.5 * sitting + 0.5 * on_chair) * tolerance(upright, 0.95, math.inf, 0.9, 'linear', 0.0) * posture
            reward = small_control * sit * dont_move
        elif name in ('balance_simple', 'balance_hard'):
            stand_b = tolerance(head, STAND_HEIGHT + 0.37, math.inf, STAND_HEIGHT / 4) * tolerance(upright, 0.9, math.inf, 1.9, 'linear', 0.0)
            reward = small_control * stand_b * dont_move
        else:
            assert name == 'pole'
            geom, _ = sc.contacts()
            hit = sc.any_contact(torch.isin(geom, sc.poles).any(-1) & (geom != 0).all(-1))
            reward = (0.5 * small_control * stand + 0.5 * move(0.5)) * torch.where(hit, 0.1, 1.0)
        end = {t: TASKS[STAGES[t]]['end'] for t in sc.tasks}
        kinds = set(end.values())
        assert len(kinds) == 1, f'scene {sc.name}: one termination per scene'
        kind = kinds.pop()
        if kind.startswith('low_'):
            terminated = z < float(kind[4:])
        elif kind.startswith('tilt_'):
            terminated = upright < float(kind[5:])
        elif kind == 'never':
            terminated = torch.zeros_like(z, dtype=torch.bool)
        else:
            # BalanceBase.get_terminated: pelvis below 0.8 m, the sphere touching anything but the floor / board, or
            # the floor touching anything but the sphere
            geom, _ = sc.contacts()
            sphere = (geom == sc.sphere).any(-1) & ~((geom == 0) | (geom == sc.board)).any(-1)
            floor = (geom == 0).any(-1) & ~(geom == sc.sphere).any(-1)
            terminated = (z < 0.8) | sc.any_contact(sphere | floor)
        return reward, terminated

    # ---- the step ----

    def step(self, action):
        # action (E, 1, A) -> reward (E, 1), done (E,), timeout (E,), info {"terms": {name: (E,)}, "metrics": {...}}
        rewards, terminated = [], []  # [(scene E,)], [(scene E,) bool]
        # pushes due in training episodes: (E, 2) base velocity kick
        self.push_timer -= 1
        due = (self.push_timer <= 0) & ~self.test
        kick = (torch.rand(self.E, 2, generator=self.gen, device=self.device) * 2 - 1) * PUSH_SPEED * due[:, None]
        self.push_timer = torch.where(due, torch.randint(*PUSH_STEPS, (self.E,), generator=self.gen, device=self.device), self.push_timer)
        for sc, lo, hi in zip(self.scenes, self.offsets, self.offsets[1:]):
            sc.qvel[:, :2] += kick[lo:hi]
            # the PD layer (gains scaled in training episodes): (scene E, A) benchmark action in [-1, 1]
            kp, kd = self.kp_scale[lo:hi], self.kd_scale[lo:hi]
            a = (kp * (sc.key_joints + PD_SCALE * action[lo:hi, 0] - sc.qpos[:, sc.act_q]) - kd * PD_DAMPING * sc.qvel[:, sc.act_v]).clamp(-1.0, 1.0)
            sc.ctrl[:] = sc.ctrl_lo + (a + 1) / 2 * (sc.ctrl_hi - sc.ctrl_lo)
        self.simulate()
        for sc, lo, hi in zip(self.scenes, self.offsets, self.offsets[1:]):
            r, t = self.scene_reward(sc, self.stage[lo:hi])
            rewards.append(r)
            terminated.append(t)
        reward, fell = torch.cat(rewards), torch.cat(terminated)
        # (E,) worlds whose physics diverged: a failed episode, reward 0
        diverged = ~self.gather(lambda sc: torch.isfinite(sc.qpos).all(-1) & (sc.qvel.abs() < 1e4).all(-1))
        reward = torch.where(diverged, 0.0, reward)
        fell = fell | diverged
        self.steps += 1
        self.ret += reward
        # feet_phase (robot/humanoid.py rewards()): each foot site's height against the swing reference of the clock
        phase, _ = self.clock()
        legs = torch.stack([phase, (phase + 0.5) % 1.0], -1)
        x = (2 * legs) % 1.0
        bezier = x ** 3 + 3 * x ** 2 * (1 - x)
        ref = SWING_HEIGHT * torch.where(legs < 0.5, bezier, 1 - bezier)
        # (E, 2) each foot site's height over the terrain under it, less its standing height
        feet = self.gather(lambda sc: sc.site_xpos[:, sc.feet, 2] - sc.foot_rest - sc.terrain[((sc.site_xpos[:, sc.feet, 0] - TERRAIN_X[0]) / TERRAIN_DX).round().long().clamp(0, len(sc.terrain) - 1)])
        gait = torch.where(diverged, 0.0, torch.exp(-((feet - ref) ** 2).sum(-1) / 0.01) * (self.freq[self.stage] > 0))
        timeout = (self.steps >= EPISODE_STEPS) & ~fell
        done = fell | timeout
        ended = self.test & done
        count = lambda m: m.sum().clamp_min(1)
        metrics = {'falls': fell.float().mean(), 'diverged': diverged.float().sum()}
        for i, name in enumerate(STAGES):
            here = self.stage == i
            metrics[f'{name}_reward'] = (reward * here).sum() / count(here)
            metrics[f'tried_{name}'] = (ended & here).float().sum()
            metrics[f'passed_{name}'] = (ended & here & (self.ret >= self.bar[i])).float().sum()
        info = {'terms': {'benchmark': reward, 'feet_phase': gait}, 'metrics': metrics}
        if done.any() and self.auto_reset:
            self.reset(done)
        return ((reward + GAIT_W * gait) * REWARD_SCALE)[:, None], done, timeout, info

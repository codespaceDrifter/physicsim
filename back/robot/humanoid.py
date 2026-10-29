# Base for humanoid (Unitree G1) courses: robots in a batched world, the stage machinery, and everything every
# course shares: actions, proprioception, falls, pushes, the head camera, the walk stage and the locomotion rewards.
# a course (robot/warehouse.py) adds its scene and task through four hooks: reset_task (lay out new episodes),
# task_tokens (its command words), task_state (its privileged / aux state), task_step (reward, episodes finished,
# metrics). every tensor has a robot axis R: proprio (E, R, PROPRIO_DIM), actions (E, R, A).
#
# control: 50 Hz. one world step (DT = 20 ms, 5 substeps of 4 ms like MuJoCo Playground's G1) per action.
# action -> joint targets: target = rest + ACTION_SCALE * action (rad), rest = the scene's drive targets (the pose it
# builds the robot in, plus a course's gravity compensation), tracked by each motor's PD (from the model).
# frames: "pelvis frame" = the pelvis body's axes (x forward, y left, z up: the MJCF link axes), like the robot's IMU.
#
# stages: each env runs one stage per episode, sampled at reset. "walk" is every course's first stage: follow a spoken
# velocity command ("walk forward +0.50 left +0.00 turn -0.25"), resampled every COMMAND_STEPS, with a gait clock
# (MuJoCo Playground G1 joystick recipe) whose frequency rises with speed.
# pass tests: every stage has one; a course's training unlocks its next stage once the newest one passes (ml/
# train_humanoid.py). TEST_PROB of every stage's episodes are tests (self.test): the trainer runs them on the policy's
# mean action (no exploration noise) and leaves them out of the policy gradient, and only they count toward the pass
# rate. walk's test is the dash: command "walk forward +1.00" for DASH_STEPS without resampling, passed if the robot
# covered DASH_DIST along its starting heading without falling, smoothly (action second difference rms under
# DASH_SMOOTH). step() reports each stage's tries and passes as metrics "tried_<stage>" / "passed_<stage>" (counts
# per step).
#
# observations per robot (observe()):
#   proprio    (E, R, PROPRIO_DIM)  the robot's own sensors (gyro, gravity, joint angles / speeds, last action) + gait
#                                   clock [sin, cos]
#   command    (E, R, TOKENS)       the command as word ids: the stage's words, padded, then the course's name last (the
#                                   game being played; last, so a model trained before courses had names keeps its
#                                   words' positions)
#   image      (E, R, 3, H, W)      head camera (uint8), refreshed every RENDER_EVERY steps (only with vision=True; a new
#                                   episode sees the last frame until the next refresh)
#   privileged (E, R, priv dim)     training only (the critic): noise-free proprio, stage one-hot, walk command, true
#                                   base velocity, height, feet, then the course's task state
#   aux        (E, R, aux dim)      training only: what the policy's state decoder learns to read out of its inputs:
#                                   base velocity, height, foot contacts / heights, then the course's task state
import math
import torch
from physics.world import World
from physics.render import Renderer
from physics.math3d import quat_rotate, quat_rotate_inv, quat_mul, quat_from_rotvec, const

DT = 0.02
SUBSTEPS = 5
ACTION_SCALE = 0.25
# foot bodies (left, right) the gait rewards look at
FEET = ['left_ankle_roll_link', 'right_ankle_roll_link']
# a foot touches the ground when its contact normal force exceeds this (N)
CONTACT_FORCE = 5.0
# hand bodies (left, right): they and the feet may touch the ground
HANDS = ['left_wrist_yaw_link', 'right_wrist_yaw_link']
# fall: any other part touches the ground (the scene's "ground" body) with more than CONTACT_FORCE, or the pelvis center
# of mass drops below its stage's height or tips past its stage's tilt (a course's FALL_LIMITS: walking stages keep
# the robot tall, or a fresh policy learns to sag into a squat until its hips hit the floor; a floor pick squats to
# ~0.35 m, pelvis tipped 60° forward)
# random pushes: every PUSH_INTERVAL steps on average, each robot's whole body gets a horizontal velocity kick up to this
PUSH_INTERVAL = 250
PUSH_SPEED = 0.5
# observation noise (uniform ± this, in the units fed to the policy), so the policy doesn't rely on exact readings
NOISE = {'gyro': 0.05, 'gravity': 0.05, 'joint_pos': 0.01, 'joint_vel': 0.075}
# reading scales into ~unit range (legged_gym)
GYRO_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
# head camera refresh: every RENDER_EVERY control steps (10 Hz)
RENDER_EVERY = 5
# walk commands: [forward, left] (m/s), turn (rad/s, + = left), each on a grid of COMMAND_STEP, resampled every
# COMMAND_STEPS; STAND_PROB of commands are all zero
COMMAND_RANGES = [(-0.5, 1.5), (-0.5, 0.5), (-1.0, 1.0)]
COMMAND_STEP = 0.25
COMMAND_STEPS = 500
STAND_PROB = 0.1
# walk command words: values are the grid points from -1.00 to +2.00 spelled like "+0.25" (room for faster commands
# than COMMAND_RANGES gives now). courses append their words after these
VALUES = [f'{v * COMMAND_STEP:+.2f}' for v in range(-4, 9)]
WALK_WORDS = ['<pad>', 'walk', 'forward', 'left', 'turn'] + VALUES
# words per command: up to 9, then the course's name
TOKENS = 10
# gait clock: its frequency (Hz) goes from GAIT_FREQ[0] standing to GAIT_FREQ[1] at GAIT_FAST (m/s) and above, by the
# commanded speed in walk and the pelvis's actual speed in other stages (the policy picks its own pace there); each
# foot follows a reference height over its cycle (up to SWING_HEIGHT and back down, the two feet half a cycle apart),
# and an air time between AIR_TIME (s) counts toward feet_air_time (MuJoCo Playground G1)
GAIT_FREQ = (1.25, 2.0)
GAIT_FAST = 1.5
SWING_HEIGHT = 0.08
AIR_TIME = (0.2, 0.5)
# walk command below this speed counts as "stand still" (no stepping rewards for air time, joints held at the pose)
STILL = 0.1
TRACKING_SIGMA = 0.25
# locomotion reward terms (rewards()); a course weights them per stage, the weighted sum is scaled by DT and clipped at
# 0 (falling early only loses future reward)
TERMS = ['track_lin_vel', 'track_ang_vel', 'ang_vel_xy', 'orientation', 'dof_pos_limits', 'hip_pos', 'feet_slip',
         'feet_phase', 'feet_air_time', 'stand_still', 'alive', 'upper_pose']
# joint limits for dof_pos_limits: this fraction of each joint's range, around its middle
SOFT_LIMIT = 0.9
# joints whose angles hip_pos / upper_pose penalize (away from the rest pose)
HIP_JOINTS = ['left_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint']
UPPER_JOINTS = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'] + \
               [f'{s}_{j}_joint' for s in ('left', 'right') for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                                                                      'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
# recordings' walk command sequence (ml/record_humanoid.py, ml/record_mj.py): [(seconds, [forward, left, turn])]
SCRIPT = [(2.0, [0.0, 0.0, 0.0]), (4.0, [0.5, 0.0, 0.0]), (4.0, [1.0, 0.0, 0.0]), (4.0, [0.5, 0.0, 0.5]),
          (3.0, [0.0, 0.5, 0.0]), (3.0, [-0.5, 0.0, 0.0]), (3.0, [0.0, 0.0, -1.0]), (2.0, [0.0, 0.0, 0.0])]
# pass tests (see the header): share of episodes that are tests; walk's: 10 m in 12 s at the dash command, action
# second difference rms under DASH_SMOOTH (a clean ~2 Hz gait's stride alone gives ~0.05-0.12; the jittery v6 walk
# measured 0.30)
TEST_PROB = 0.25
DASH_SPEED = 1.0
DASH_STEPS = 600
DASH_DIST = 10.0
DASH_SMOOTH = 0.15



class HumanoidCourse:
    # a course subclass sets: NAME (its word), STAGES (["walk", ...]), STAGE_STEPS {stage: control steps}, FALL_LIMITS
    # {stage: (pelvis height m, tilt rad)},
    # WEIGHTS {term in TERMS: (weight per stage)}, VOCAB (WALK_WORDS + its words, NAME among them).
    # scene: scene JSON with G1s whose ids start with prefixes (robot axis order). actuators: [joint name] in action
    # order. stage_probs: (S,) probability of each stage at reset. vision: render the head cameras
    def __init__(self, scene, num_envs, device, prefixes, actuators, stage_probs, vision, seed):
        self.E = num_envs
        self.R = len(prefixes)
        self.device = device
        self.world = World(scene, num_envs, device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        w = self.world
        assert scene['settings']['dt'] == DT and scene['settings']['substeps'] == SUBSTEPS
        assert self.STAGES[0] == 'walk' and len(stage_probs) == len(self.STAGES) and sum(stage_probs) > 0
        # [LongTensor] body indices of each robot's parts (moved / pushed together)
        self.parts = [torch.tensor([i for i, bid in enumerate(w.body_ids) if bid.startswith(p)], device=device) for p in prefixes]
        # (R,) pelvis body, (R, 2) feet (left, right)
        self.pelvis = torch.tensor([w.body_index[p + 'pelvis'] for p in prefixes], device=device)
        self.feet = torch.tensor([[w.body_index[p + f] for f in FEET] for p in prefixes], device=device)
        self.hands = torch.tensor([[w.body_index[p + h] for h in HANDS] for p in prefixes], device=device)
        # (R, P') each robot's parts that must not touch the ground (all but the feet and hands), the ground body
        self.trunk = torch.tensor([[i for i, bid in enumerate(w.body_ids) if bid.startswith(p) and bid.removeprefix(p) not in FEET + HANDS]
                                   for p in prefixes], device=device)
        self.ground = torch.tensor([w.body_index['ground']], device=device)
        # (R, A) joint of each actuator, (A,) rest command = the action-0 targets: the scene's drive targets (the pose
        # the robot is built in, plus any gravity compensation the course holds it with)
        self.act_joints = torch.tensor([[w.joints.index[p + n] for n in actuators] for p in prefixes], device=device)
        self.A = len(actuators)
        self.rest = w.joints.init_target[self.act_joints[0]]
        # (2,) foot center-of-mass height at rest: foot heights are measured from it
        self.foot_rest = w.init_pos[self.feet[0], 1]
        joint = lambda names: torch.tensor([actuators.index(n) for n in names], device=device)
        # (H,) hip roll / yaw actuators, (U,) waist and arm actuators
        self.hip = joint(HIP_JOINTS)
        self.upper = joint(UPPER_JOINTS)
        # (A,) soft joint limits of the actuated joints (minus the rest pose, like joint_pos())
        j = w.joints
        lo, hi = j.lower[self.act_joints[0]], j.upper[self.act_joints[0]]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_LIMIT
        self.limit_lo = torch.where(j.has_limit[self.act_joints[0]], mid - half, -math.inf) - self.rest
        self.limit_hi = torch.where(j.has_limit[self.act_joints[0]], mid + half, math.inf) - self.rest
        self.stage_probs = torch.tensor(stage_probs, dtype=torch.float32, device=device)
        # (S, terms) reward weight table, (S,) episode length of each stage
        self.weights = torch.tensor([[self.WEIGHTS[k][s] for k in TERMS] for s in range(len(self.STAGES))], device=device)
        self.stage_steps = torch.tensor([self.STAGE_STEPS[s] for s in self.STAGES], device=device)
        # (S,) pelvis height (m) and tilt (rad) limits per stage
        self.fall_height = torch.tensor([self.FALL_LIMITS[s][0] for s in self.STAGES], device=device)
        self.fall_tilt = torch.tensor([self.FALL_LIMITS[s][1] for s in self.STAGES], device=device)
        # (3,) / (3,) walk command grid bounds in grid units
        self.cmd_lo = torch.tensor([round(lo / COMMAND_STEP) for lo, hi in COMMAND_RANGES], device=device)
        self.cmd_hi = torch.tensor([round(hi / COMMAND_STEP) for lo, hi in COMMAND_RANGES], device=device)
        self.word = {wd: i for i, wd in enumerate(self.VOCAB)}  # {word: id}
        # per env / robot: (E,) stage index, (E,) control steps taken, (E,) episode length, (E, R, 3) walk command,
        # (E, R, A) the last three actions (newest first: last_action is in the proprio), (E, R) gait phase in [0, 1)
        # of the left leg (the right leg is half a cycle later), (E, R) steps until the next push, (E, R, 2) each
        # foot's time in the air and its contact last step
        self.stage = torch.zeros(self.E, dtype=torch.long, device=device)
        self.steps = torch.zeros(self.E, dtype=torch.long, device=device)
        self.ep_steps = torch.zeros(self.E, dtype=torch.long, device=device)
        self.command = torch.zeros(self.E, self.R, 3, device=device)
        self.last_action = torch.zeros(self.E, self.R, self.A, device=device)
        self.prev_action = torch.zeros(self.E, self.R, self.A, device=device)
        self.prev2_action = torch.zeros(self.E, self.R, self.A, device=device)
        self.phase = torch.zeros(self.E, self.R, device=device)
        # (E,) this episode is a pass test. the dash (walk's test): (E,) this episode is one, (E, R, 2) start (x, z),
        # (E, R, 3) start heading, (E,) sum over its steps of the robots' mean squared action second difference
        self.test = torch.zeros(self.E, dtype=torch.bool, device=device)
        self.dash = torch.zeros(self.E, dtype=torch.bool, device=device)
        self.dash_start = torch.zeros(self.E, self.R, 2, device=device)
        self.dash_heading = torch.zeros(self.E, self.R, 3, device=device)
        self.dash_jerk = torch.zeros(self.E, device=device)
        self.push_timer = torch.randint(PUSH_INTERVAL // 2, PUSH_INTERVAL * 3 // 2, (self.E, self.R), generator=self.gen, device=device)
        self.pushes = True
        # done envs start a new episode inside step() (off: a caller drives one continuous episode, ml/demo_humanoid.py)
        self.auto_reset = True
        self.air_time = torch.zeros(self.E, self.R, 2, device=device)
        self.last_contact = torch.zeros(self.E, self.R, 2, dtype=torch.bool, device=device)
        # camera: [Renderer] per robot (always built: a trainer turns vision on when a stage needs it), control steps
        # since the start (render clock)
        self.vision = vision
        self.cameras = [Renderer(w, c) for c in scene['cameras']]
        self.clock_steps = 0

    def enable_cuda_graph(self):
        # physics step and cameras as CUDA graphs, then a fresh reset (capture warm-up stepped the world)
        self.world.enable_cuda_graph()
        for cam in self.cameras:
            cam.enable_cuda_graph()
        self.reset(torch.ones(self.E, dtype=torch.bool, device=self.device))
        self.render()

    def is_stage(self, name):
        # (E,) bool
        return self.stage == self.STAGES.index(name)

    def reset(self, mask):
        # new episode in envs where mask (E,) is True: back to the scene's pose, a stage, a walk command, then the
        # course's own layout
        self.world.reset(mask)
        envs = mask.nonzero().view(-1)
        n = len(envs)
        if n == 0:
            return
        self.last_action[envs] = 0.0
        self.prev_action[envs] = 0.0
        self.prev2_action[envs] = 0.0
        self.phase[envs] = 0.0
        self.stage[envs] = torch.multinomial(self.stage_probs, n, replacement=True, generator=self.gen)
        self.steps[envs] = 0
        self.air_time[envs] = 0.0
        self.last_contact[envs] = False
        walk = self.stage[envs] == 0
        self.test[envs] = torch.rand(n, generator=self.gen, device=self.device) < TEST_PROB
        self.dash[envs] = walk & self.test[envs]
        self.ep_steps[envs] = torch.where(self.dash[envs], DASH_STEPS, self.stage_steps[self.stage[envs]])
        self.dash_jerk[envs] = 0.0
        self.new_command(mask)
        self.reset_task(mask, envs)
        # dash start: where the course placed the robot
        fwd, _ = self.heading()
        self.dash_start[envs] = self.world.pos[envs][:, self.pelvis][..., [0, 2]]
        self.dash_heading[envs] = fwd[envs]
        # joint angles are refreshed by the physics step: recompute them so the first observation reads the new pose
        self.world.joints.prepare(errors_only=True)

    def new_command(self, mask):
        # sample a grid walk command for every robot in envs where mask (E,) is True (zero outside the walk stage, the
        # dash's in a dash)
        steps = self.cmd_lo + torch.floor(torch.rand(self.E, self.R, 3, generator=self.gen, device=self.device)
                                          * (self.cmd_hi - self.cmd_lo + 1)).long()
        stand = (torch.rand(self.E, self.R, 1, generator=self.gen, device=self.device) < STAND_PROB) | ~self.is_stage('walk')[:, None, None]
        cmd = torch.where(stand, 0.0, steps * COMMAND_STEP)
        cmd = torch.where(self.dash[:, None, None], const([DASH_SPEED, 0.0, 0.0], self.device), cmd)
        self.command = torch.where(mask[:, None, None], cmd, self.command)

    def apply(self, action):
        # (E, R, A) actions -> joint targets, then one control step of physics (with random pushes)
        w = self.world
        w.joints.target[:, self.act_joints.reshape(-1)] = (self.rest + ACTION_SCALE * action).reshape(self.E, -1)
        self.prev2_action = self.prev_action
        self.prev_action = self.last_action
        self.last_action = action
        if self.pushes:
            self.push_timer -= 1
            due = self.push_timer <= 0
            for r, parts in enumerate(self.parts):
                kick = (torch.rand(self.E, 2, generator=self.gen, device=self.device) * 2 - 1) * PUSH_SPEED * due[:, r, None]
                w.velocity[:, parts, 0] += kick[:, :1]
                w.velocity[:, parts, 2] += kick[:, 1:]
            self.push_timer = torch.where(due, torch.randint(PUSH_INTERVAL // 2, PUSH_INTERVAL * 3 // 2, (self.E, self.R), generator=self.gen, device=self.device), self.push_timer)
        w.step()

    def render(self):
        # refresh the camera frames: (E, R, 3, H, W) uint8
        self.image = torch.stack([cam.render().permute(0, 3, 1, 2) for cam in self.cameras], 1).clone()

    # ---- the robot's state ----

    def joint_pos(self):
        # (E, R, A) joint angles minus the rest pose
        return self.world.joints.value[:, self.act_joints] - self.rest

    def joint_vel(self):
        # (E, R, A) joint speeds (rad/s)
        return self.world.joints.speeds()[:, self.act_joints]

    def pelvis_frame(self):
        # (E, R, 4) pelvis orientation: its body frame is the pelvis link's center-of-mass frame, whose axes match the
        # link's to 0.05° (the model's inertial quat), so it stands in for the IMU frame
        return self.world.quat[:, self.pelvis]

    def gravity(self):
        # (E, R, 3) unit gravity direction in the pelvis frame: (0, 0, -1) when upright
        q = self.pelvis_frame()
        return quat_rotate_inv(q, const([0.0, -1.0, 0.0], self.device).expand(q.shape[:-1] + (3,)))

    def gyro(self):
        # (E, R, 3) pelvis angular velocity in the pelvis frame
        return quat_rotate_inv(self.pelvis_frame(), self.world.velocity[:, self.pelvis, 3:])

    def heading(self):
        # (E, R, 3) forward and (E, R, 3) left unit vectors of the pelvis, flattened onto the ground
        fwd = quat_rotate(self.pelvis_frame(), const([1.0, 0.0, 0.0], self.device).expand(self.E, self.R, 3)) * const([1.0, 0.0, 1.0], self.device)
        fwd = fwd / torch.linalg.norm(fwd, dim=-1, keepdim=True).clamp_min(1e-6)
        # left = forward turned +90° about up (+y): (x, z) -> (z, -x)
        left = torch.stack([fwd[..., 2], torch.zeros_like(fwd[..., 0]), -fwd[..., 0]], -1)
        return fwd, left

    def relative(self, points):
        # (E, R, K, 3) world points -> (E, R, K, 3) [forward, left, up] from robot r's pelvis, in its heading frame
        fwd, left = self.heading()
        d = points - self.world.pos[:, self.pelvis][:, :, None]
        return torch.stack([(d * fwd[:, :, None]).sum(-1), (d * left[:, :, None]).sum(-1), d[..., 1]], -1)

    def base_velocity(self):
        # (E, R, 3) [forward, left, turn rate] of the pelvis: ground velocity in its heading frame + yaw rate (rad/s, + = left)
        v = self.world.velocity[:, self.pelvis]
        fwd, left = self.heading()
        return torch.stack([(v[..., :3] * fwd).sum(-1), (v[..., :3] * left).sum(-1), v[..., 4]], -1)

    def height(self):
        # (E, R) pelvis center of mass height
        return self.world.pos[:, self.pelvis, 1]

    def contact_force(self, bodies, others=None):
        # (E, *bodies.shape) normal force (N) on each of the bodies (any shape of body indices) from everything they
        # touch, or only from the bodies others (M,)
        c = self.world.contacts
        b = bodies.view(1, 1, -1)
        a_side, b_side = c.idx_a.unsqueeze(-1) == b, c.idx_b.unsqueeze(-1) == b
        if others is not None:
            # (E, C, 1) the contact's other side is one of others
            a_side = a_side & (c.idx_b.unsqueeze(-1) == others).any(-1, keepdim=True)
            b_side = b_side & (c.idx_a.unsqueeze(-1) == others).any(-1, keepdim=True)
        on = c.valid.unsqueeze(-1) & (a_side | b_side)
        impulse = (c.lam[..., :1] * on).sum(1)
        return (impulse / self.world.h).view(self.E, *bodies.shape)

    def foot_height(self):
        # (E, R, 2) foot height above its rest height
        return self.world.pos[:, self.feet, 1] - self.foot_rest

    def foot_speed(self):
        # (E, R, 2) horizontal foot speed
        v = self.world.velocity[:, self.feet]
        return torch.linalg.norm(v[..., [0, 2]], dim=-1)

    def fallen(self):
        # (E, R) a part other than the feet and hands on the ground, or the pelvis below / tipped past its stage's limits
        tipped = -self.gravity()[..., 2] < torch.cos(self.fall_tilt[self.stage])[:, None]
        low = self.height() < self.fall_height[self.stage][:, None]
        return (self.contact_force(self.trunk, self.ground) > CONTACT_FORCE).any(-1) | low | tipped

    def legs(self):
        # (E, R, 2) gait phase of each leg (left, right) in [0, 1)
        return torch.stack([self.phase, (self.phase + 0.5) % 1.0], -1)

    def proprio(self, noise):
        # (E, R, PROPRIO_DIM) what the robot's own sensors read (gyro, gravity, joint angles and speeds, last action),
        # then the gait clock. noise: add observation noise (training) or not
        parts = {'gyro': self.gyro() * GYRO_SCALE, 'gravity': self.gravity(), 'joint_pos': self.joint_pos(),
                 'joint_vel': self.joint_vel() * JOINT_VEL_SCALE}
        if noise:
            parts = {k: v + (torch.rand(v.shape, generator=self.gen, device=self.device) * 2 - 1) * NOISE[k] for k, v in parts.items()}
        clock = torch.stack([torch.sin(2 * math.pi * self.phase), torch.cos(2 * math.pi * self.phase)], -1)
        return torch.cat(list(parts.values()) + [self.last_action, clock], -1)

    def place_robot(self, envs, r, xz, yaw, pose=None):
        # move robot r rigidly in envs (n,): its pelvis to (x, z) (n, 2), turned by yaw (n,) about +y from its scene pose,
        # or from pose: ((n, P, 3) positions, (n, P, 4) orientations of its parts, built at the scene pose's pelvis spot
        # facing +x: another joint configuration, see robot/warehouse.py)
        w = self.world
        parts = self.parts[r]
        n, P = len(envs), len(parts)
        pos, quat = pose if pose is not None else (w.init_pos[parts].expand(n, P, 3), w.init_quat[parts].expand(n, P, 4))
        q_yaw = quat_from_rotvec(torch.stack([torch.zeros_like(yaw), yaw, torch.zeros_like(yaw)], -1))[:, None].expand(n, P, 4)
        base0 = w.init_pos[self.pelvis[r]]
        rel = quat_rotate(q_yaw, pos - base0)
        target = torch.stack([xz[:, 0], torch.full_like(xz[:, 0], float(base0[1])), xz[:, 1]], -1)
        w.pos[envs[:, None], parts] = rel + target[:, None]
        w.quat[envs[:, None], parts] = quat_mul(q_yaw, quat)
        w.velocity[envs[:, None], parts] = 0.0
        w.inv_inertia_world = w.compute_inv_inertia_world()

    # ---- observations, rewards, the step ----

    def tokens(self):
        # (E, R, TOKENS) the command: walk forward <v> left <v> turn <v> (walk) or the course's words for its stage,
        # then the course's name
        w = self.word
        value = (torch.round(self.command / COMMAND_STEP).long() + 4) + w[VALUES[0]]
        fixed = lambda name: torch.full((self.E, self.R), w[name], device=self.device)
        words = [fixed('walk'), fixed('forward'), value[..., 0], fixed('left'), value[..., 1], fixed('turn'), value[..., 2]]
        walk = torch.stack(words + [fixed('<pad>')] * (TOKENS - 1 - len(words)) + [fixed(self.NAME)], -1)
        return torch.where(self.is_stage('walk')[:, None, None], walk, self.task_tokens())

    def observe(self, noise=True):
        # {"proprio", "tokens", ("image"), "privileged", "aux"}, see the header
        vel, height = self.base_velocity(), self.height()[..., None]
        contact = (self.contact_force(self.feet) > CONTACT_FORCE).float()
        feet = torch.cat([contact, self.foot_height(), self.foot_speed()], -1)
        stage = (self.stage[:, None] == torch.arange(len(self.STAGES), device=self.device)).float()[:, None].expand(self.E, self.R, len(self.STAGES))
        task_priv, task_aux = self.task_state()
        clean = self.proprio(noise=False)
        out = {'proprio': self.proprio(noise=True) if noise else clean, 'command': self.tokens(),
               'privileged': torch.cat([clean, stage, self.command, vel, height, feet, task_priv], -1),
               'aux': torch.cat([vel, height, feet[..., :4], task_aux], -1)}
        if self.vision:
            out['image'] = self.image
        return out

    def rewards(self):
        # {name in TERMS: (E, R) unweighted locomotion term}. updates the feet's air time
        vel = self.base_velocity()
        q = self.joint_pos()
        contact = self.contact_force(self.feet) > CONTACT_FORCE
        # (E, R, 2) foot height the gait reference wants now: a cubic bezier up to SWING_HEIGHT over the first half of
        # the foot's cycle, back down over the second
        legs = self.legs()
        x = (2 * legs) % 1.0
        bezier = x ** 3 + 3 * x ** 2 * (1 - x)
        ref = SWING_HEIGHT * torch.where(legs < 0.5, bezier, 1 - bezier)
        # air time: credited on touchdown
        touchdown = contact & ~self.last_contact
        self.air_time = self.air_time + DT
        air = ((self.air_time - AIR_TIME[0]).clamp(max=AIR_TIME[1] - AIR_TIME[0]) * touchdown).sum(-1)
        self.air_time = torch.where(contact, 0.0, self.air_time)
        self.last_contact = contact
        moving = torch.linalg.norm(self.command, dim=-1) > STILL
        lin_err = ((self.command[..., :2] - vel[..., :2]) ** 2).sum(-1)
        ang_err = (self.command[..., 2] - vel[..., 2]) ** 2
        return {'track_lin_vel': torch.exp(-lin_err / TRACKING_SIGMA), 'track_ang_vel': torch.exp(-ang_err / TRACKING_SIGMA),
                'ang_vel_xy': (self.gyro()[..., :2] ** 2).sum(-1),
                'orientation': (self.gravity()[..., :2] ** 2).sum(-1),
                'dof_pos_limits': ((self.limit_lo - q).clamp_min(0) + (q - self.limit_hi).clamp_min(0)).sum(-1),
                'hip_pos': (q[..., self.hip] ** 2).sum(-1),
                'feet_slip': (self.foot_speed() ** 2 * contact).sum(-1),
                'feet_phase': torch.exp(-((self.foot_height() - ref) ** 2).sum(-1) / 0.01),
                'feet_air_time': air * moving,
                'stand_still': q.abs().sum(-1) * ~moving,
                'alive': torch.ones(self.E, self.R, device=self.device),
                'upper_pose': (q[..., self.upper] ** 2).sum(-1)}

    def step(self, action):
        # action (E, R, A). returns reward (E, R), done (E,), timeout (E,) (done by the time limit, not by falling or
        # by finishing the task: the value of the state it ended in still counts), info {"terms": {name: (E,) mean over robots, weighted},
        # "metrics": {"walk_lin_err": training walk envs' mean |command - velocity| (m/s), "action_jerk": walk envs' rms action
        # second difference, "falls": falls per env-step, "tried_walk" / "passed_walk": dashes ended / passed, plus the
        # course's}}.
        # done envs are already reset; read observe() after the step
        self.apply(action)
        self.steps += 1
        self.clock_steps += 1
        walk = self.is_stage('walk')
        speed = torch.where(walk[:, None], torch.linalg.norm(self.command[..., :2], dim=-1), torch.linalg.norm(self.base_velocity()[..., :2], dim=-1))
        freq = GAIT_FREQ[0] + (GAIT_FREQ[1] - GAIT_FREQ[0]) * (speed / GAIT_FAST).clamp(max=1.0)
        self.phase = (self.phase + freq * DT) % 1.0
        # (E, R, terms) weighted by the env's stage
        raw = self.rewards()
        weighted = torch.stack([raw[k] for k in TERMS], -1) * self.weights[self.stage][:, None]
        reward = (weighted.sum(-1) * DT).clamp_min(0.0)
        fell = self.fallen()
        task_reward, finished, metrics = self.task_step(fell)
        reward = reward + task_reward
        lin_err = torch.linalg.norm(self.command[..., :2] - self.base_velocity()[..., :2], dim=-1).mean(-1)
        # training walk episodes only (the dash tests command DASH_SPEED whatever a trainer's command curriculum allows)
        trained = walk & ~self.test
        metrics['walk_lin_err'] = (lin_err * trained).sum() / trained.sum().clamp_min(1)
        # (E,) the robots' mean squared action second difference (per joint)
        jerk = ((self.last_action - 2 * self.prev_action + self.prev2_action) ** 2).mean((-2, -1))
        metrics['action_jerk'] = ((jerk * walk).sum() / walk.sum().clamp_min(1)).sqrt()
        # falls per env-step (episodes also end by finishing the task or by the time limit)
        metrics['falls'] = fell.any(-1).float().mean()
        resample = (self.steps % COMMAND_STEPS == 0) & walk & ~self.dash
        if resample.any():
            self.new_command(resample)
        fell = fell.any(-1)
        timeout = (self.steps >= self.ep_steps) & ~fell & ~finished
        done = fell | timeout | finished
        # dash: covered DASH_DIST along the start heading (every robot), smoothly, by its end without falling
        self.dash_jerk += jerk * self.dash
        covered = ((self.world.pos[:, self.pelvis][..., [0, 2]] - self.dash_start) * self.dash_heading[..., [0, 2]]).sum(-1).amin(-1)
        ended = self.dash & done
        passed = ended & timeout & (covered >= DASH_DIST) & ((self.dash_jerk / self.steps).sqrt() < DASH_SMOOTH)
        metrics['tried_walk'] = ended.float().sum()
        metrics['passed_walk'] = passed.float().sum()
        info = {'terms': {k: weighted[..., i].mean(-1) for i, k in enumerate(TERMS)}, 'metrics': metrics}
        if done.any() and self.auto_reset:
            self.reset(done)
        if self.vision and self.clock_steps % RENDER_EVERY == 0:
            self.render()
        return reward, done, timeout, info

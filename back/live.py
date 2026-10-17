# Live simulation for the editor: one world stepped in real time, driven by websocket messages (see api.py).
#
# a session holds one world (1 env). messages in: play / pause / resume / step / stop / drive / autopilot.
# frames out: {"type": "frame", "time", "poses": [[px,py,pz,qw,qx,qy,qz], ...], "joints": {id: value}, "cameras": [...],
#              "info": {"stage": str, "commands": [str per robot], "score": [orange, teal], "success": [bool per robot]}
#              in a course scene}  (success = the robot's stage goal holds right now)
#
# a course scene (settings "stage", see robot/course.py) runs that stage: you drive lekiwi/, the autopilot drives the
# other robot, or both with the "_both" modes. in the speedrun robot 1 sits parked, so the autopilot drives yours.
# a robot knocked over (base tipped past 60°) for RESPAWN_AFTER seconds is put back at its start pose.
import base64
import os
import torch
from physics.world import World
from physics.render import Renderer
from robot.env import ACTION_REPEAT
from robot.course import CourseEnv, STAGES
from ml.teacher import Teacher
from ml.vla import VLA
from ml.rollout import TeacherDriver, VLADriver, ExpertDriver

# {mode: checkpoint} the "teacher" autopilot (state-based) and the "student" autopilot (the VLA, cameras + command)
CHECKPOINTS = {m: os.path.join(os.path.dirname(__file__), 'checkpoints', 'final', n)
               for m, n in (('teacher', 'teacher.pt'), ('student', 'vla_best_v13.pt'))}
# physics steps between camera renders sent to the client (60 Hz physics -> 10 Hz camera)
CAMERA_EVERY = 6
# seconds a knocked-over robot lies before it respawns
RESPAWN_AFTER = 3.0
AUTOPILOT_MODES = ('off', 'expert', 'teacher', 'student', 'expert_both', 'teacher_both', 'student_both')


def make_driver(kind, device):
    # "expert" | "teacher" | "student" -> its rollout driver (ml/rollout.py), models from CHECKPOINTS
    if kind == 'expert':
        return ExpertDriver()
    assert kind in CHECKPOINTS, f'unknown driver {kind}'
    path = CHECKPOINTS[kind]
    if not os.path.exists(path):
        raise ValueError(f'no trained {kind}: {path} is missing')
    model = (Teacher() if kind == 'teacher' else VLA()).to(device)
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    model.eval()
    return TeacherDriver(model) if kind == 'teacher' else VLADriver(model)


class LiveSession:
    def __init__(self, scene, device):
        self.device = device
        self.env = CourseEnv(1, device, scene_json=scene) if 'stage' in scene['settings'] else None
        self.world = self.env.world if self.env else World(scene, 1, device)
        self.cameras = [Renderer(self.world, c) for c in scene['cameras']]
        if str(device).startswith('cuda'):
            if self.env:
                self.env.enable_cuda_graph()
            else:
                self.world.enable_cuda_graph()
                # capture warm-up stepped the world, start from the scene again
                self.world.reset(torch.ones(1, dtype=torch.bool, device=device))
            for cam in self.cameras:
                cam.enable_cuda_graph()
            self.world.time = 0.0
        self.world.joints.prepare()
        self.steps = 0
        self.autopilot = 'off'
        # [(rollout driver (ml/rollout.py), [robot])] who drives which robots (the autopilot: one entry), and whether
        # the drivers' next call starts fresh
        self.drivers = []
        self.fresh = True
        if self.env:
            # (R, 2) each robot's start base xz (its respawn spot), [float] seconds each robot has been knocked over
            self.start_xz = self.env.base_frame()[0][0].clone()
            self.down_for = [0.0] * self.env.R

    def set_autopilot(self, mode):
        assert mode in AUTOPILOT_MODES, f'unknown autopilot mode {mode}'
        if mode != 'off' and self.env is None:
            raise ValueError('autopilot needs a course scene (course_1_drive, course_4_race, ...)')
        self.fresh = True
        self.autopilot = mode
        if mode == 'off':
            self.drivers = []
            return
        # you keep robot 0 unless "_both"; only active robots (the speedrun's parked robot 1 stays parked)
        active = self.env.active()[0].tolist()
        robots = [r for r in range(self.env.R) if active[r] and (mode.endswith('_both') or r != 0)]
        self.drivers = [(make_driver(mode.removesuffix('_both'), self.device), robots or [0])]

    def drive(self, targets):
        # {joint_id: target} from teleop (robot 0's joints). ignored while a driver drives robot 0
        if not any(0 in robots for d, robots in self.drivers):
            self.world.joints.set_targets(targets)

    @torch.no_grad()
    def step(self):
        # one physics step, with the drivers choosing new targets at the 10 Hz control rate
        if self.drivers and self.steps % ACTION_REPEAT == 0:
            env = self.env
            for driver, robots in self.drivers:
                # (1, R, A)
                targets = env.action_to_targets(driver(env, torch.tensor([self.fresh], device=self.device)))
                for r in robots:
                    self.world.joints.target[:, env.act_joints[r]] = targets[:, r]
            self.fresh = False
        if self.env and self.steps % ACTION_REPEAT == 0:
            self.respawn_knocked_over()
        self.world.step()
        self.steps += 1

    def respawn_knocked_over(self):
        # called at the control rate: count how long each robot has been knocked over, put it back at its start pose
        # (scene orientation, at rest) once it reaches RESPAWN_AFTER
        env, w = self.env, self.world
        flipped = env.flipped()[0].tolist()
        env0 = torch.zeros(1, dtype=torch.long, device=self.device)
        for r in range(env.R):
            self.down_for[r] = self.down_for[r] + ACTION_REPEAT * w.dt if flipped[r] else 0.0
            if self.down_for[r] < RESPAWN_AFTER:
                continue
            env.place_robot(env0, r, self.start_xz[r:r + 1], torch.zeros(1, device=self.device))
            w.velocity[0, env.parts[r]] = 0.0
            w.inv_inertia_world.copy_(w.compute_inv_inertia_world())
            self.down_for[r] = 0.0

    def frame(self, with_cameras):
        # websocket frame message for env 0
        w = self.world
        msg = {'type': 'frame', 'time': w.time, 'poses': w.poses()[0].cpu().tolist(),
               'joints': dict(zip(w.joints.ids, w.joints.value[0].cpu().tolist())), 'cameras': []}
        if self.env:
            msg['info'] = {'stage': STAGES[int(self.env.stage[0])], 'commands': [self.env.command(0, r) for r in range(self.env.R)],
                           'score': self.env.score()[0].tolist(), 'success': self.env.success_now()[0].tolist()}
        if with_cameras:
            for cam in self.cameras:
                rgb = cam.render()[0].cpu().numpy()
                msg['cameras'].append({'id': cam.id, 'width': cam.W, 'height': cam.H, 'rgb': base64.b64encode(rgb.tobytes()).decode()})
        return msg

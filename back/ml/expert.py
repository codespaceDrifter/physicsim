# Scripted expert for pick-and-place. it cheats: it reads true positions from the simulator (privileged state),
# which the learned policy never sees. its job is to produce demonstrations (pixels -> actions) for behavior cloning.
#
# per env state machine:
#   approach -> pre (arm above block) -> lower -> close -> lift -> carry (drive to zone) -> place -> release -> retreat
#   -> approach next block ... -> done. a failed grasp or a dropped block goes back to approach.
import math
import numpy as np
from robot.tofubot import (AXLE_X, TRACK_Z, WHEEL_RADIUS, TURRET_X, SHOULDER_Y, CHASSIS_Y, UPPER_ARM, FOREARM, FINGER_SIZE,
                           FINGER_OPEN, STEER_LIMIT, ARM_LIMITS, ACTUATORS)
from robot.pick_place import BLOCK_SIZE

WHEELBASE = 2 * AXLE_X
# smallest turn radius at full steering lock, measured at the rear axle center
MIN_TURN = WHEELBASE / math.tan(STEER_LIMIT)
# elbow -> finger center
LINK2 = FOREARM + FINGER_SIZE[1] / 2
# preferred horizontal distance from the turret axis to the block when stopping to grasp
REACH = 0.19
# accepted grasp / release spot: at least FRONT_MIN ahead of the turret (clear of the chassis and front wheels),
# within REACH_MAX, and at most REACH_BEARING (rad) to the side, which the turret covers instead of driving
FRONT_MIN = 0.155
REACH_MAX = 0.23
REACH_BEARING = 0.7
DRIVE_SPEED = 0.35
# finger-center heights (m): above the block, grasping, releasing
Y_PRE, Y_GRASP, Y_RELEASE = 0.10, 0.028, 0.05
# carry pose (horizontal reach, height): arm up and back, out of the front camera's view
CARRY = (0.06, 0.28)
# finger-center height reached by lifting straight up before stowing into the carry pose
Y_LIFT = 0.12
# max change of an arm / steering target per control step (rad): 1.5 rad/s at 10 Hz. smooth like a real servo,
# so a held block isn't flung
RATE = 0.15
# control steps (10 Hz) spent in each timed phase
WAIT = {'pre': 12, 'lower': 10, 'close': 7, 'lift': 8, 'stow': 12, 'place': 22, 'release': 5, 'retreat': 24}
TIMED_NEXT = {'pre': 'lower', 'lower': 'close', 'close': 'lift', 'lift': 'stow', 'place': 'release', 'release': 'retreat'}
# {actuator name: index in the action/target vector}
A = {name: i for i, (name, kind) in enumerate(ACTUATORS)}


def arm_ik(rho, y, shoulder_y):
    # finger center to horizontal distance rho from the turret axis and height y. angles from vertical, positive = forward.
    # returns (shoulder, elbow) for the elbow-up solution, clamped to reachable / joint limits
    dy = y - shoulder_y
    d = min(max(math.hypot(rho, dy), abs(UPPER_ARM - LINK2) + 1e-3), UPPER_ARM + LINK2 - 1e-3)
    elbow = math.acos((d * d - UPPER_ARM ** 2 - LINK2 ** 2) / (2 * UPPER_ARM * LINK2))
    shoulder = math.atan2(rho, dy) - math.atan2(LINK2 * math.sin(elbow), UPPER_ARM + LINK2 * math.cos(elbow))
    return (min(max(shoulder, ARM_LIMITS['shoulder'][0]), ARM_LIMITS['shoulder'][1]),
            min(max(elbow, ARM_LIMITS['elbow'][0]), ARM_LIMITS['elbow'][1]))


def ackermann(speed, steer):
    # center steer angle + chassis speed -> (steer_l, steer_r, wheel_rl, wheel_rr) so every wheel rolls about one turn center
    if abs(steer) < 1e-3:
        return 0.0, 0.0, speed / WHEEL_RADIUS, speed / WHEEL_RADIUS
    # signed turn radius at the rear axle center, positive = turning left (center on the -z side)
    R = WHEELBASE / math.tan(steer)
    steer_l = min(max(math.atan(WHEELBASE / (R - TRACK_Z)), -STEER_LIMIT), STEER_LIMIT)
    steer_r = min(max(math.atan(WHEELBASE / (R + TRACK_Z)), -STEER_LIMIT), STEER_LIMIT)
    return steer_l, steer_r, speed * (R - TRACK_Z) / R / WHEEL_RADIUS, speed * (R + TRACK_Z) / R / WHEEL_RADIUS


class Expert:
    def __init__(self, num_envs, num_blocks):
        self.num_blocks = num_blocks
        # per env: phase name, control steps spent in it, block being handled
        self.phase = ['approach'] * num_envs
        self.timer = [0] * num_envs
        self.block = [0] * num_envs
        # per env control steps left of a committed back-up maneuver (hysteresis, so it doesn't chatter forward/back)
        self.backing = [0] * num_envs
        # per env (reach, bearing) where the block was grabbed, to lift straight up from there
        self.grab = [(0.0, 0.0)] * num_envs
        # (E, A) targets sent last step, for rate limiting. None until the first act()
        self.last = None

    def reset(self, e):
        self.phase[e] = 'approach'
        self.timer[e] = 0
        self.block[e] = 0
        self.backing[e] = 0
        if self.last is not None:
            self.last[e] = 0.0

    def act(self, env):
        # (E, A) joint targets for every env, from the simulator's true state
        w = env.world
        chassis_pos = w.pos[:, env.chassis].cpu().numpy()
        chassis_q = w.quat[:, env.chassis].cpu().numpy()
        blocks = w.pos[:, env.blocks].cpu().numpy()
        zones = w.pos[:, env.zones].cpu().numpy()
        grip = w.pos[:, env.fingers].mean(1).cpu().numpy()
        targets = np.zeros((env.E, len(ACTUATORS)), dtype=np.float32)
        for e in range(env.E):
            qw, qx, qy, qz = chassis_q[e]
            # chassis forward (+x) and right (+z) in world, xz plane only
            fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qz - qw * qy)])
            right = np.array([2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy)])
            shoulder_y = chassis_pos[e, 1] + (SHOULDER_Y - CHASSIS_Y)

            def local(p):
                # world point -> (forward, right) offsets from the turret axis
                d = p[[0, 2]] - chassis_pos[e, [0, 2]]
                return float(d @ fwd) - TURRET_X, float(d @ right)

            b = min(self.block[e], self.num_blocks - 1)
            phase = self.phase[e]
            self.timer[e] += 1
            t = targets[e]
            # default: stopped, arm in carry pose, gripper open
            t[A['finger_l']] = t[A['finger_r']] = FINGER_OPEN
            t[A['shoulder']], t[A['elbow']] = arm_ik(*CARRY, shoulder_y)

            if phase in ('approach', 'carry'):
                goal = blocks[e, b] if phase == 'approach' else zones[e, b]
                if phase == 'carry':
                    t[A['finger_l']] = t[A['finger_r']] = 0.0
                    # dropped it on the way: go pick it up again
                    if blocks[e, b, 1] < BLOCK_SIZE and np.linalg.norm(blocks[e, b] - grip[e]) > 0.05:
                        self.set(e, 'approach')
                fx, rz = local(goal)
                rho = math.hypot(fx, rz)
                bearing = math.atan2(-rz, fx)
                if fx > FRONT_MIN and rho < REACH_MAX and abs(bearing) < REACH_BEARING:
                    self.set(e, 'pre' if phase == 'approach' else 'place')
                else:
                    # pure pursuit toward the goal. goal behind: far -> full lock forward to turn around,
                    # close -> back up on opposite lock (a 3-point turn)
                    alpha = math.atan2(-rz, fx + TURRET_X)
                    # goal inside the full-lock turning circle on its side: unreachable driving forward. commit to ~1 s
                    # of reversing on opposite lock (3-point turn: swings the nose toward the goal)
                    center_z = -MIN_TURN if rz < 0 else MIN_TURN
                    if math.hypot(fx + TURRET_X + AXLE_X, rz - center_z) < MIN_TURN + 0.05 and abs(alpha) > 0.3:
                        self.backing[e] = max(self.backing[e], 10)
                    if self.backing[e] > 0:
                        self.backing[e] -= 1
                        speed, steer = -0.2, -math.copysign(STEER_LIMIT, alpha)
                    elif abs(alpha) < math.pi / 2:
                        speed = min(max(2.0 * (rho - REACH), -DRIVE_SPEED), DRIVE_SPEED) * max(0.3, math.cos(alpha))
                        steer = min(max(1.5 * alpha, -STEER_LIMIT), STEER_LIMIT) if speed > 0 else 0.0
                    elif rho > 0.45:
                        speed, steer = 0.6 * DRIVE_SPEED, math.copysign(STEER_LIMIT, alpha)
                    else:
                        speed, steer = -0.2, -math.copysign(STEER_LIMIT, alpha)
                    t[A['steer_fl']], t[A['steer_fr']], t[A['drive_rl']], t[A['drive_rr']] = ackermann(speed, steer)

            elif phase in ('pre', 'lower', 'close', 'lift', 'stow', 'place', 'release'):
                holding = phase in ('close', 'lift', 'stow', 'place')
                goal = zones[e, b] if phase in ('place', 'release') else blocks[e, b]
                fx, rz = local(goal)
                if phase == 'stow':
                    (rho, y), bearing = CARRY, 0.0
                elif phase == 'lift':
                    # straight up from where the block was grabbed: keep the turret and reach, only raise
                    rho, bearing, y = self.grab[e][0], self.grab[e][1], Y_LIFT
                else:
                    rho, bearing = math.hypot(fx, rz), math.atan2(-rz, fx)
                    y = {'pre': Y_PRE, 'lower': Y_GRASP, 'close': Y_GRASP, 'place': Y_RELEASE, 'release': Y_RELEASE}[phase]
                    if phase == 'close':
                        self.grab[e] = (rho, bearing)
                t[A['turret']] = bearing
                t[A['shoulder']], t[A['elbow']] = arm_ik(rho, y, shoulder_y)
                if holding:
                    t[A['finger_l']] = t[A['finger_r']] = 0.0
                if phase == 'stow' and self.timer[e] >= WAIT['stow']:
                    # grasp check: block came up with the gripper
                    self.set(e, 'carry' if blocks[e, b, 1] > 0.05 else 'approach')
                elif phase in TIMED_NEXT and self.timer[e] >= WAIT[phase]:
                    self.set(e, TIMED_NEXT[phase])

            elif phase == 'retreat':
                # raise the open gripper clear of the block first, then back away
                t[A['shoulder']], t[A['elbow']] = arm_ik(*CARRY, shoulder_y)
                if self.timer[e] > WAIT['retreat'] // 2:
                    t[A['drive_rl']] = t[A['drive_rr']] = -0.2 / WHEEL_RADIUS
                if self.timer[e] >= WAIT['retreat']:
                    self.block[e] += 1
                    self.set(e, 'done' if self.block[e] >= self.num_blocks else 'approach')
        # rate limit arm and steering targets (wheel speeds and fingers switch instantly)
        if self.last is None:
            self.last = targets.copy()
        limited = [A[n] for n in ('steer_fl', 'steer_fr', 'turret', 'shoulder', 'elbow')]
        targets[:, limited] = self.last[:, limited] + np.clip(targets[:, limited] - self.last[:, limited], -RATE, RATE)
        self.last = targets.copy()
        return targets

    def set(self, e, phase):
        self.phase[e] = phase
        self.timer[e] = 0

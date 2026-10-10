# Scripted expert for LeKiwi block tasks (pick-and-place, block race). it cheats: it reads true positions from the
# simulator (privileged state), which the learned policy never sees. its job is to produce demonstrations
# (pixels -> actions) for behavior cloning, and an opponent for the race.
#
# per (env, robot) state machine. the task picks the goal (env.expert_goal: which block, where to drop it):
#   approach -> pre (gripper above block) -> lower -> close -> lift -> stow -> carry (drive to drop point) -> place
#   -> release -> retreat -> approach the next goal ... -> done. a failed grasp or a dropped block goes back to approach.
import math
import numpy as np
from robot.lekiwi import (URDF_JOINTS, URDF_TOOL, ARM_LIMITS, ARM_MOUNT_X, BASE_TOP, BASE_Y, WHEEL_ANGLES, WHEEL_RING,
                          WHEEL_RADIUS, STS3215_SPEED, ACTUATORS, REST)
from robot.env import BLOCK_SIZE

# ---- arm geometry in the arm's vertical plane (h = forward from the pan axis, v = up from the arm base), URDF zero pose
PAN_X = URDF_JOINTS['shoulder_pan'][0][0]
LIFT = (URDF_JOINTS['shoulder_lift'][0][0] - PAN_X, URDF_JOINTS['shoulder_lift'][0][2])
ELBOW = (URDF_JOINTS['elbow_flex'][0][0] - PAN_X, URDF_JOINTS['elbow_flex'][0][2])
WRIST = (URDF_JOINTS['wrist_flex'][0][0] - PAN_X, URDF_JOINTS['wrist_flex'][0][2])
TOOL = (URDF_TOOL[0] - PAN_X, URDF_TOOL[2])
# link lengths and their zero-pose angles from vertical (positive = leaning forward; positive joint angles lean forward)
L1, A1 = math.hypot(ELBOW[0] - LIFT[0], ELBOW[1] - LIFT[1]), math.atan2(ELBOW[0] - LIFT[0], ELBOW[1] - LIFT[1])
L2, A2 = math.hypot(WRIST[0] - ELBOW[0], WRIST[1] - ELBOW[1]), math.atan2(WRIST[0] - ELBOW[0], WRIST[1] - ELBOW[1])
L3, A3 = math.hypot(TOOL[0] - WRIST[0], TOOL[1] - WRIST[1]), math.atan2(TOOL[0] - WRIST[0], TOOL[1] - WRIST[1])
# the moving jaw closes onto the fixed jaw from the outer side: put the block this far beyond the tool point
JAW_OFFSET = 0.008 + BLOCK_SIZE / 2

# preferred forward distance from the pan axis to the block when stopping to grasp, and the accepted error
REACH = 0.19
REACH_TOL = 0.025
DRIVE_SPEED = 0.25
# tool tip heights above the ground (m): above the block, grasping, lifted, releasing
Y_PRE, Y_GRASP, Y_LIFT, Y_RELEASE = 0.09, 0.006, 0.10, 0.015
# carry pose: (forward from pan axis, tool tip height) with the block held above the front of the base
CARRY = (0.13, 0.17)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.8, -0.17
# control steps (10 Hz) spent in each timed phase
WAIT = {'pre': 12, 'lower': 10, 'close': 8, 'lift': 8, 'stow': 12, 'place': 15, 'release': 6, 'retreat': 15}
TIMED_NEXT = {'pre': 'lower', 'lower': 'close', 'close': 'lift', 'lift': 'stow', 'place': 'release', 'release': 'retreat'}
# max change of an arm target per control step (rad): 1.5 rad/s at 10 Hz, smooth like a real servo
RATE = 0.15
# {actuator name: index in the action/target vector}
A = {name: i for i, (name, kind) in enumerate(ACTUATORS)}
ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


def arm_ik(h, tip_y, base_y):
    # gripper pointing straight down with its tip at forward distance h from the pan axis and height tip_y.
    # base_y: current height of the base center. returns (shoulder_lift, elbow_flex, wrist_flex), elbow-up solution
    v = tip_y - (base_y + BASE_TOP - BASE_Y)
    # tool straight down: wrist_flex joint sits L3 above the tip (tool angle from vertical = π)
    dx, dy = h - LIFT[0], v + L3 - LIFT[1]
    d = min(max(math.hypot(dx, dy), abs(L1 - L2) + 1e-3), L1 + L2 - 1e-3)
    bend = math.acos(max(-1.0, min(1.0, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2))))
    phi1 = math.atan2(dx, dy) - math.atan2(L2 * math.sin(bend), L1 + L2 * math.cos(bend))
    lift = phi1 - A1
    elbow = phi1 + bend - A2 - lift
    wrist = math.pi - A3 - lift - elbow
    return tuple(min(max(q, ARM_LIMITS[n][0]), ARM_LIMITS[n][1]) for q, n in zip((lift, elbow, wrist), ARM[1:4]))


def wheel_speeds(vx, vz, wy):
    # kiwi drive inverse kinematics: base velocity (forward vx, right vz, yaw rate wy about +y) -> wheel speeds (rad/s)
    out = []  # [float] in WHEEL_ANGLES order
    for angle in WHEEL_ANGLES.values():
        ax, az = math.cos(angle), math.sin(angle)
        # wheel center velocity = v + ω × p, with p = WHEEL_RING * axle; positive spin rolls the wheel along axle × up
        px, pz = WHEEL_RING * ax, WHEEL_RING * az
        cx, cz = vx + wy * pz, vz - wy * px
        out.append((cx * -az + cz * ax) / WHEEL_RADIUS)
    peak = max(abs(s) for s in out)
    return [s * min(1.0, STS3215_SPEED / max(peak, 1e-9)) for s in out]


class Expert:
    def __init__(self, num_envs, num_robots):
        E, R = num_envs, num_robots
        # per (env, robot): phase name, control steps spent in it, block being fetched, drop point xz
        self.phase = [['approach'] * R for _ in range(E)]
        self.timer = [[0] * R for _ in range(E)]
        self.block = [[0] * R for _ in range(E)]
        self.drop = [[np.zeros(2)] * R for _ in range(E)]
        # per (env, robot) (pan, reach, wrist roll) locked when the jaws start closing: following the block after
        # that would chase it as the jaw nudges it, dragging it across the floor
        self.grab = [[(0.0, 0.0, 0.0)] * R for _ in range(E)]
        # (E, R, A) targets sent last step, for rate limiting
        self.last = np.array([[[0.0] * 3 + [REST[n] for n in ARM] + [REST['gripper']]] * R] * E, dtype=np.float32)

    def reset(self, e):
        # env e started a new episode
        for r in range(len(self.phase[e])):
            self.set(e, r, 'approach')
            self.last[e, r] = [0.0] * 3 + [REST[n] for n in ARM] + [REST['gripper']]

    def set(self, e, r, phase):
        self.phase[e][r] = phase
        self.timer[e][r] = 0

    def act(self, env, robots=None):
        # (E, R, A) joint targets for every env and robot (robots: subset of robot indices to drive, default all;
        # the others get zeros), from the simulator's true state
        w = env.world
        robots = range(env.R) if robots is None else robots
        base_pos = w.pos[:, env.base].cpu().numpy()
        base_q = w.quat[:, env.base].cpu().numpy()
        blocks = w.pos[:, env.blocks].cpu().numpy()
        block_q = w.quat[:, env.blocks].cpu().numpy()
        grip = env.grip().cpu().numpy()
        targets = np.zeros((env.E, env.R, len(ACTUATORS)), dtype=np.float32)
        for e in range(env.E):
            for r in robots:
                self.act_one(env, e, r, targets[e, r], base_pos[e, r], base_q[e, r], blocks, block_q, grip[e, r])
        # rate limit arm targets (wheel speeds and the gripper switch instantly)
        arm = [A[n] for n in ARM]
        targets[..., arm] = self.last[..., arm] + np.clip(targets[..., arm] - self.last[..., arm], -RATE, RATE)
        self.last = targets.copy()
        return targets

    def act_one(self, env, e, r, t, base_pos, base_q, blocks, block_q, grip):
        # fill t (A,): robot r's targets in env e
        qw, qx, qy, qz = base_q
        # base forward (+x) and right (+z) in world, xz plane only
        fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qz - qw * qy)])
        right = np.array([2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy)])
        # base yaw about +y (positive = turned left)
        yaw_base = math.atan2(-fwd[1], fwd[0])
        base_y = base_pos[1]

        def local(p):
            # world xz point -> (forward, right) offsets from the pan axis
            d = p - base_pos[[0, 2]]
            return float(d @ fwd) - ARM_MOUNT_X - PAN_X, float(d @ right)

        phase = self.phase[e][r]
        self.timer[e][r] += 1
        timer = self.timer[e][r]
        # default: stopped, arm in the carry pose, gripper open
        t[A['gripper']] = GRIPPER_OPEN
        t[A['shoulder_lift']], t[A['elbow_flex']], t[A['wrist_flex']] = arm_ik(*CARRY, base_y)

        if phase == 'approach':
            # (re)pick the goal every step: the target block may move or be taken
            goal = env.expert_goal(e, r, blocks)
            if goal is None:
                self.set(e, r, 'done')
                return
            self.block[e][r], self.drop[e][r] = goal
        b = self.block[e][r]

        if phase in ('approach', 'carry'):
            holding = phase == 'carry'
            if holding:
                t[A['gripper']] = GRIPPER_CLOSED
                # dropped it on the way: go pick something up again
                if blocks[e, b, 1] < BLOCK_SIZE and np.linalg.norm(blocks[e, b] - grip) > 0.05:
                    self.set(e, r, 'approach')
            fx, rz = local(self.drop[e][r] if holding else blocks[e, b, [0, 2]])
            bearing = math.atan2(rz, fx)
            if abs(fx - REACH) < REACH_TOL and abs(rz) < REACH_TOL:
                self.set(e, r, 'place' if holding else 'pre')
            else:
                # holonomic: drive straight at the stopping point while turning to face the goal
                vx, vz = 2.0 * (fx - REACH), 2.0 * rz
                speed = math.hypot(vx, vz)
                if speed > DRIVE_SPEED:
                    vx, vz = vx * DRIVE_SPEED / speed, vz * DRIVE_SPEED / speed
                # yaw rate about +y turns left; the goal is to the right when bearing > 0
                t[A['wheel_back']], t[A['wheel_left']], t[A['wheel_right']] = wheel_speeds(vx, vz, -2.0 * bearing)

        elif phase in ('pre', 'lower', 'close', 'lift', 'stow', 'place', 'release'):
            goal = self.drop[e][r] if phase in ('place', 'release') else blocks[e, b, [0, 2]]
            fx, rz = local(goal)
            # pan to face the goal (positive pan turns right), reach so the block lands between the jaws
            pan = math.atan2(rz, fx)
            h = math.hypot(fx, rz) - JAW_OFFSET
            tip = {'pre': Y_PRE, 'lower': Y_GRASP, 'close': Y_GRASP, 'lift': Y_LIFT, 'place': Y_RELEASE, 'release': Y_RELEASE}.get(phase)
            if phase == 'stow':
                pan, (h, tip) = 0.0, CARRY
            if phase in ('pre', 'lower'):
                # roll so the jaws close on two opposite faces: block yaw relative to the arm's direction, mod 90°
                yaw_block = 2 * math.atan2(block_q[e, b, 2], block_q[e, b, 0])
                rel = (yaw_block - yaw_base + pan) % (math.pi / 2)
                roll = rel - math.pi / 2 if rel > math.pi / 4 else rel
                self.grab[e][r] = (pan, h, roll)
            if phase in ('pre', 'lower', 'close', 'lift'):
                pan, h, t[A['wrist_roll']] = self.grab[e][r]
            t[A['shoulder_pan']] = pan
            t[A['shoulder_lift']], t[A['elbow_flex']], t[A['wrist_flex']] = arm_ik(h, tip, base_y)
            if phase in ('close', 'lift', 'stow', 'place'):
                t[A['gripper']] = GRIPPER_CLOSED
            if phase == 'stow' and timer >= WAIT['stow']:
                # grasp check: block came up with the gripper
                self.set(e, r, 'carry' if blocks[e, b, 1] > 0.05 else 'approach')
            elif phase in TIMED_NEXT and timer >= WAIT[phase]:
                self.set(e, r, TIMED_NEXT[phase])

        elif phase == 'retreat':
            # raise the open gripper clear of the block first, then back away
            if timer > WAIT['retreat'] // 2:
                t[A['wheel_back']], t[A['wheel_left']], t[A['wheel_right']] = wheel_speeds(-0.15, 0.0, 0.0)
            if timer >= WAIT['retreat']:
                self.set(e, r, 'approach')

        elif phase == 'done':
            t[A['shoulder_lift']], t[A['elbow_flex']], t[A['wrist_flex']] = REST['shoulder_lift'], REST['elbow_flex'], REST['wrist_flex']
            # a block may become available again (stolen, dropped): look again now and then
            if timer >= 10:
                self.set(e, r, 'approach')

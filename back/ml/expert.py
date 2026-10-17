# Scripted expert for LeKiwi block tasks (pick-and-place, block race). it cheats: it reads true positions from the
# simulator (privileged state), which the learned policy never sees. its job is to produce demonstrations
# (pixels -> actions) for behavior cloning, and an opponent for the race.
#
# Markov: every step the expert reads what to do from the state alone (is a block in the jaws, where the arm is, how
# far the jaws are open, where the base stands): no remembered phase, timer or last command, so its label for any
# state a clone reaches (DAgger) is well defined. a hidden phase made DAgger labels follow the expert's own script and
# the clone got worse; arm targets stepped from the last command (a hidden rate limit) were tiny offsets from where the
# arm already was, and the clone learned to command where the arm is and stopped moving.
#   holding a block:  stow (arm to the carry pose) -> carry (drive in front of the first free home slot, facing the
#                     wall, the arm extending toward it on the way in) -> release (open once the block is over the
#                     slot, still driving: it drops from the carry height)
#   empty jaws:       raise (a low tool tip goes up before anything else) -> turn (at home facing the wall: spin
#                     left to face the arena, so a camera can see the blocks left) -> approach (drive to the stopping point of
#                     the nearest block not home) -> pre (tip above the block) -> lower -> close -> lift
#                     (a jaw that closed on nothing opens again)
# the arm aims at a point a little ahead of where its tool tip is now, toward the goal (pure pursuit, LOOK): the tip
# moves along straight lines at a speed the servos set.
import math
import numpy as np
from robot.lekiwi import (URDF_JOINTS, URDF_TOOL, ARM_LIMITS, ARM_MOUNT_X, BASE_TOP, BASE_Y, WHEEL_ANGLES, WHEEL_RING,
                          WHEEL_RADIUS, STS3215_SPEED, ACTUATORS, REST)
from robot.env import BLOCK_SIZE
from robot.course import ZONE_SIZE

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
# delivering: the home zone has slots (depth from the zone's arena-side edge, sideways offset), filled in this order
# (back row first), so delivered blocks land apart instead of on each other. a slot is free if no other block is within
# SLOT_FREE of it
SLOTS = ((0.23, 0.0), (0.23, -0.12), (0.23, 0.12), (0.12, 0.0), (0.12, -0.12), (0.12, 0.12))
SLOT_FREE = 0.06
# the base drives (facing the wall) to its stand, the pan axis EXTEND + JAW_OFFSET in front of the slot: the base stays
# out of the zone (driving and turning in it kept shoving delivered blocks). within RAMP (m) of the stand the arm
# extends smoothly from the carry pose to EXTEND (forward m, at the carry height), with no switch on the way
EXTEND = 0.23
RAMP = 0.3
# the jaws open once the held block is within DROP_NEAR (m, horizontally) of the slot, and the block drops from the
# carry height. the base keeps driving toward the stand while they open: with the same wheel command on both sides of
# DROP_NEAR the learned policy keeps rolling across it. (a placing step gated on the base standing at an exact spot,
# then lowering the arm, left the clones parked just outside the gate holding the block over the zone for most of
# the race)
DROP_NEAR = 0.05
# once the arm has left the carry pose (by ENGAGED in any of pan rad, forward m, height m), the base may drift up to
# KEEP x the stopping tolerance without the expert driving off again (the base coasts a little after stopping)
ENGAGED = np.array([0.05, 0.02, 0.02])
KEEP = 2.5
# empty jaws, within TURN_NEAR (m) of its home, facing more than TURN_FACE (rad) away from the arena (it just placed a
# block) and the next block behind it: spin left in place at TURN_RATE (rad/s) until that block is in front. a camera-only student facing the
# wall can't see where the remaining blocks are, so "turn toward the next block" was a guess it averaged into a stall
TURN_NEAR, TURN_FACE, TURN_RATE = 0.6, math.radians(100), 1.5
# top base speed (m/s): about the most the wheels give driving forward
DRIVE_SPEED = 0.33
# top speed (m/s) and turn rate (rad/s) while carrying a block: spinning at full rate flings it out of the jaws
CARRY_SPEED = 0.33
CARRY_TURN = 0.8
# driving steers around other blocks: each block within AVOID_DIST of the base center pushes the base away, up to
# AVOID_SPEED (m/s) at contact. the base rides 10 cm off the floor, so a block it drives at slides under it, out of
# the arm's reach
AVOID_DIST = 0.25
AVOID_SPEED = 0.5
# tool tip heights above the ground (m): above the block, grasping, lifted
Y_PRE, Y_GRASP, Y_LIFT = 0.09, 0.006, 0.10
# a tip below Y_LOW is raised before the arm moves sideways or the base drives
Y_LOW = Y_PRE - 0.02
# carry pose: (forward from pan axis, tool tip height) with the block held above the front of the base
CARRY = (0.13, 0.17)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.8, -0.17
# jaw angles (rad): above JAW_OPENED the jaws are open; below JAW_SHUT they have closed onto a block (about 0.3) and
# the arm lifts; below JAW_EMPTY they closed on nothing
JAW_OPENED, JAW_SHUT, JAW_EMPTY = 0.6, 0.45, 0.05
# a block whose center is within BETWEEN (horizontally) of the grip point sits between the jaws
BETWEEN = 0.02
# a block counts as held when its center is above HELD_Y and within HELD_DIST of the grip point
HELD_Y, HELD_DIST = 0.042, 0.06
# settled: the tool tip is within SETTLE_TIP of the goal (pan rad, forward m, height m); STOW_TIP for the carry pose.
# generous: a learned policy's arm is off by a centimeter or two, and tighter checks left its imitation stuck waiting
SETTLE_TIP = np.array([0.08, 0.02, 0.02])
STOW_TIP = np.array([0.2, 0.04, 0.04])
# the arm targets the point at most LOOK (pan rad, forward m, height m) from the tool tip toward the goal: the tip
# moves along straight lines (joint-space moves swung it in arcs that swept blocks under the base and shook held
# ones loose) at a speed the lookahead sets (faster tips shook grasped blocks out of the jaws)
LOOK = np.array([0.15, 0.03, 0.03])
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


def arm_fk(lift, elbow, wrist, base_y):
    # inverse of arm_ik for a gripper pointing straight down: joint angles -> (forward distance of the tip from the pan
    # axis, tip height)
    a1, a2 = lift + A1, lift + elbow + A2
    h = LIFT[0] + L1 * math.sin(a1) + L2 * math.sin(a2)
    v = LIFT[1] + L1 * math.cos(a1) + L2 * math.cos(a2) - L3
    return h, v + base_y + BASE_TOP - BASE_Y


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
        # [[str]] (E, R) what each robot did last step (for inspection only, never read back: the expert has no memory)
        self.phase = [['approach'] * num_robots for _ in range(num_envs)]

    def act(self, env, robots=None):
        # (E, R, A) joint targets for every env and robot (robots: subset of robot indices to drive, default all;
        # the others get zeros), from the simulator's true state
        w = env.world
        robots = range(env.R) if robots is None else robots
        base_pos = w.pos[:, env.base].cpu().numpy()
        base_q = w.quat[:, env.base].cpu().numpy()
        blocks = w.pos[:, env.blocks].cpu().numpy()
        block_yaw = env.block_yaw().cpu().numpy()
        homes = w.pos[:, env.homes][..., [0, 2]].cpu().numpy()
        # (E, R, A) joint angles (wheels: unused)
        joints = w.joints.value[:, env.act_joints].cpu().numpy()
        grip = env.grip().cpu().numpy()
        # (E, R) robots that fetch blocks: active, not in the drive stage
        fetch = (env.active() & ~env.is_stage('drive')[:, None]).cpu().numpy()
        targets = np.zeros((env.E, env.R, len(ACTUATORS)), dtype=np.float32)
        for e in range(env.E):
            for r in robots:
                q = joints[e, r]
                # tool tip (pan, forward, height) now
                tip = np.array([q[A['shoulder_pan']], *arm_fk(q[A['shoulder_lift']], q[A['elbow_flex']], q[A['wrist_flex']], base_pos[e, r, 1])])
                want = self.act_one(e, r, targets[e, r], tip, base_pos[e, r], base_q[e, r], blocks[e], block_yaw[e], homes[e], grip[e], q, fetch[e, r])
                aim = tip + np.clip(want - tip, -LOOK, LOOK)
                targets[e, r, A['shoulder_pan']] = aim[0]
                targets[e, r, [A['shoulder_lift'], A['elbow_flex'], A['wrist_flex']]] = arm_ik(aim[1], aim[2], base_pos[e, r, 1])
        return targets

    def act_one(self, e, r, t, tip, base_pos, base_q, blocks, block_yaw, homes, grips, joints, fetch):
        # fill t (A,): robot r's wheel, wrist roll and gripper targets in env e; returns where the tool tip should go
        # (pan, forward, height) (3,). tip (3,): the tool tip now. blocks (B, 3), block_yaw (B,), homes (R, 2) xz,
        # grips (R, 3): this env's. joints (A,): this robot's joint angles
        qw, qx, qy, qz = base_q
        # base forward (+x) and right (+z) in world, xz plane only
        fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qz - qw * qy)])
        right = np.array([2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy)])
        # base yaw about +y (positive = turned left)
        yaw_base = math.atan2(-fwd[1], fwd[0])
        carry = np.array([0.0, *CARRY])

        def local(p):
            # world xz point -> (forward, right) offsets from the pan axis
            d = p - base_pos[[0, 2]]
            return float(d @ fwd) - ARM_MOUNT_X - PAN_X, float(d @ right)

        def drive(vx, vz, bearing, skip, top, turn):
            # wheel targets: base velocity (forward, right) capped at top (m/s) plus a push away from every block
            # but those in skip ([int]: the one fetched or held, delivered ones) within AVOID_DIST, turning toward
            # bearing (rad, positive = to the right; yaw rate about +y turns left) at most turn rad/s
            # (B - 1, 2) base center - block, in the robot frame
            away = base_pos[[0, 2]] - np.delete(blocks[:, [0, 2]], skip, 0)
            away = np.stack([away @ fwd, away @ right], -1)
            dist = np.linalg.norm(away, axis=-1, keepdims=True)
            push = (AVOID_SPEED * np.clip(1 - dist / AVOID_DIST, 0, 1) * away / np.maximum(dist, 1e-6)).sum(0)
            vx, vz = vx + push[0], vz + push[1]
            scale = min(1.0, top / max(math.hypot(vx, vz), 1e-9))
            t[A['wheel_back']], t[A['wheel_left']], t[A['wheel_right']] = wheel_speeds(vx * scale, vz * scale, float(np.clip(-2.0 * bearing, -turn, turn)))

        def settled(want, tol=SETTLE_TIP):
            # the tool tip reached want
            return bool((np.abs(tip - want) < tol).all())

        def over(p):
            # (pan, forward) of the tip that puts world xz point p between the jaws
            fx, rz = local(p)
            return math.atan2(rz, fx), math.hypot(fx, rz) - JAW_OFFSET

        # default: stopped, gripper open, wrist roll and tip where they were
        t[A['gripper']] = GRIPPER_OPEN
        t[A['wrist_roll']] = joints[A['wrist_roll']]
        engaged = bool((np.abs(tip - carry) > ENGAGED).any())
        jaw = joints[A['gripper']]
        # blocks in the jaws: center up and near the grip point
        held = [b for b in range(len(blocks)) if blocks[b, 1] > HELD_Y and np.linalg.norm(blocks[b] - grips[r]) < HELD_DIST]
        if not fetch:
            self.phase[e][r] = 'idle'
            return carry

        if held:
            b = held[0]
            t[A['gripper']] = GRIPPER_CLOSED
            # first free slot. out: unit x pointing from the zone to the arena
            home = homes[r]
            out = -math.copysign(1.0, home[0])
            others = np.delete(blocks[:, [0, 2]], b, 0)
            slots = [np.array([home[0] + out * (ZONE_SIZE / 2 - d), home[1] + side]) for d, side in SLOTS]
            # [float] distance from each slot to the nearest other block
            clear = [np.linalg.norm(others - s, axis=-1).min() for s in slots]
            free = [s for s, c in zip(slots, clear) if c > SLOT_FREE]
            # a block lying between two slots can block both: then the slot with the most room
            if not free:
                free = [slots[int(np.argmax(clear))]]
            # (forward, right) from the pan axis to the stand
            fx, rz = local(free[0] + np.array([out * (EXTEND + JAW_OFFSET), 0.0]))
            facing = math.atan2(-out * right[0], -out * fwd[0])
            near = float(np.clip(1 - math.hypot(fx, rz) / RAMP, 0, 1))
            want = np.array([0.0, CARRY[0] + (EXTEND - CARRY[0]) * near, CARRY[1]])
            if not settled(want, STOW_TIP) and jaw < JAW_OPENED:
                self.phase[e][r] = 'stow'
                return want
            # open over the slot, or keep opening (the block is falling out of the jaws)
            if np.linalg.norm(blocks[b, [0, 2]] - free[0]) < DROP_NEAR or jaw > JAW_OPENED:
                self.phase[e][r] = 'release'
                t[A['gripper']] = GRIPPER_OPEN
            else:
                self.phase[e][r] = 'carry'
            # blocks resting in or just outside the zone don't push the base off the stand (one lying there would
            # keep it from ever arriving)
            homed = [c for c in range(len(blocks)) if (np.abs(blocks[c, [0, 2]] - home) < ZONE_SIZE / 2 + 0.1).all() and blocks[c, 1] < BLOCK_SIZE]
            drive(2.0 * fx, 2.0 * rz, facing, [b] + homed, CARRY_SPEED, CARRY_TURN)
            return want

        # empty jaws: the nearest block that is neither home (resting in this robot's zone) nor in the other robot's
        # jaws, measured from the tool tip while it is low (over the block being grasped), else from the base
        home = homes[r]
        tip_xz = base_pos[[0, 2]] + (ARM_MOUNT_X + PAN_X + tip[1] * math.cos(tip[0])) * fwd + tip[1] * math.sin(tip[0]) * right
        low = tip[2] < Y_LOW
        ref = tip_xz if low else base_pos[[0, 2]]
        cands = [b for b in range(len(blocks))
                 if not ((np.abs(blocks[b, [0, 2]] - home) < ZONE_SIZE / 2).all() and blocks[b, 1] < BLOCK_SIZE)
                 and not any(o != r and blocks[b, 1] > BLOCK_SIZE and np.linalg.norm(blocks[b] - grips[o]) < HELD_DIST for o in range(len(grips)))]
        if not cands:
            self.phase[e][r] = 'done'
            return np.array([tip[0], tip[1], Y_PRE]) if low else carry
        b = min(cands, key=lambda c: np.linalg.norm(blocks[c, [0, 2]] - ref))
        fx, rz = local(blocks[b, [0, 2]])
        off = max(abs(fx - REACH), abs(rz)) / REACH_TOL
        pan, h = over(blocks[b, [0, 2]])
        above = abs(pan - tip[0]) < SETTLE_TIP[0] and abs(h - tip[1]) < SETTLE_TIP[1]
        at_block = off < 1 or (engaged and off < KEEP)
        between = np.linalg.norm(blocks[b, [0, 2]] - grips[r, [0, 2]]) < BETWEEN
        if low and jaw < JAW_OPENED:
            # jaws (partly) shut, low: on the block (it sits between them) keep closing, then lift once they are shut
            # on it; shut on nothing (or the block is not between them): open and go up
            if between and jaw > JAW_EMPTY:
                t[A['gripper']] = GRIPPER_CLOSED
                if jaw > JAW_SHUT:
                    self.phase[e][r] = 'close'
                    return tip
                self.phase[e][r] = 'lift'
                return np.array([tip[0], tip[1], Y_LIFT])
            self.phase[e][r] = 'missed'
            return np.array([tip[0], tip[1], Y_PRE])
        if not at_block or not above:
            # moving to the block (base) or above it (arm): a low tip goes up first
            if low:
                self.phase[e][r] = 'raise'
                return np.array([tip[0], tip[1], Y_PRE])
            if not at_block:
                out = -math.copysign(1.0, home[0])
                if np.linalg.norm(base_pos[[0, 2]] - home) < TURN_NEAR and out * fwd[0] < math.cos(TURN_FACE) and fx < 0:
                    self.phase[e][r] = 'turn'
                    t[A['wheel_back']], t[A['wheel_left']], t[A['wheel_right']] = wheel_speeds(0.0, 0.0, TURN_RATE)
                    return carry
                self.phase[e][r] = 'approach'
                drive(2.0 * (fx - REACH), 2.0 * rz, math.atan2(rz, fx), [b], DRIVE_SPEED, math.inf)
                return carry
        # roll so the jaws close on two opposite faces: block yaw relative to the arm's direction, mod 90°
        rel = (block_yaw[b] - yaw_base + pan) % (math.pi / 2)
        t[A['wrist_roll']] = rel - math.pi / 2 if rel > math.pi / 4 else rel
        if not above or jaw < JAW_OPENED:
            self.phase[e][r] = 'pre'
            return np.array([pan, h, Y_PRE])
        want = np.array([pan, h, Y_GRASP])
        if settled(want):
            self.phase[e][r] = 'close'
            t[A['gripper']] = GRIPPER_CLOSED
            return want
        self.phase[e][r] = 'lower'
        return want

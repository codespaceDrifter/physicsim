# Physics Simulation - How It Works

This explains the whole codebase from the ground up: the physics engine, the robot, the learning pipeline, the
server and the editor. Read it top to bottom.

```
back/                      python, run everything from here as modules:  cd back && python -m <module>
  physics/                 the engine
    math3d.py              batched quaternion / vector math, SPD inverse
    world.py               World: bodies, integration, the step loop, CUDA graphs
    joints.py              revolute / prismatic / fixed joints, motors, limits (one global solve)
    contacts.py            contact solving: non-penetration, friction (incl. omni wheels), restitution
    collision.py           narrowphase: plane / sphere / box / cylinder pairs
    render.py              camera pixels by batched ray casting
  robot/
    lekiwi.py              the LeKiwi model (omni base + SO-101 arm, cameras)
    pick_place.py          batched pick-and-place environment (reset, step, reward, pixels)
  ml/
    expert.py              scripted expert that reads true simulator state
    collect.py             expert demonstrations -> dataset
    policy.py              pixel policy network (CNN)
    train_bc.py            behavior cloning
    train_ppo.py           PPO fine-tuning
    rollout.py / eval.py   success rate + recording a policy episode
  write_scenes.py          writes scenes/lekiwi_pick.json
  recording.py             recording format (playback)
  live.py                  real-time simulation session for the editor
  api.py                   FastAPI server: scenes, prefabs, recordings, websocket
  scenes/                  scene JSON files the editor opens / saves
front/                     Three.js + Vite editor (edit / simulate / playback)
main.py                    starts back (api) + front (vite) together
logs/{name}/               recordings
back/data/bc/              behavior cloning dataset (gitignored)
back/checkpoints/          trained policies (gitignored)
```

Units are SI everywhere (meters, kilograms, seconds, radians). Y is up. Quaternions are `[w, x, y, z]`.

---

## Part 1: The Scene

Everything starts from a **scene**: one JSON document that describes bodies, joints and cameras. The same JSON is
stored on disk (`back/scenes/*.json`), edited in the frontend, sent to the backend to simulate, and saved next to
recordings. One format everywhere.

```json
{
  "name": "lekiwi_pick",
  "settings": {"dt": 0.0167, "substeps": 4, "gravity": [0, -9.81, 0], "G": 0.0},
  "bodies": [ ... ],
  "joints": [ ... ],
  "cameras": [ ... ]
}
```

- `dt`: seconds per world step. `substeps`: solver substeps per step (Part 3).
- `gravity`: uniform gravity acceleration.
- `G`: gravitational constant for attraction between every pair of bodies (planets). 0 = off.

### Bodies

```json
{
  "id": "lekiwi/wheel_left",
  "shape": {"type": "cylinder", "radius": 0.0508, "height": 0.025, "omni": true},
  "position": [0.05, 0.0508, -0.0866], "rotation": [w, x, y, z],
  "velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
  "density": 493, "static": false, "collide": true, "group": 1,
  "friction": 1.0, "restitution": 0.0, "color": "#2b2b2b"
}
```

- shapes: `box` (`size` = full side lengths), `sphere` (`radius`), `cylinder` (`radius`, `height`, axis = local +y,
  optional `"omni": true` for an omni wheel, Part 3), `plane` (infinite, normal = local +y, must be static).
  MuJoCo and Bullet use exactly these primitives for robots because every pair has an exact, fast collision test.
- mass = volume × density. The inertia of each primitive is diagonal in its own frame.
- `static`: infinite mass, never moves. `collide: false`: visual only (target zones).
- `group`: bodies that share a nonzero group never collide with each other (robot parts don't self-collide).
- `friction` of a pair = `sqrt(fa · fb)`, `restitution` = `max(ra, rb)` (Box2D's mixing rules).

### Joints

```json
{
  "id": "lekiwi/shoulder_lift", "type": "revolute",
  "body_a": "lekiwi/shoulder", "body_b": "lekiwi/upper_arm",
  "anchor": [0.069, 0.282, 0.018], "axis": [0, 0, -1],
  "limits": [-1.745, 1.745], "initial": -1.7,
  "drive": {"mode": "position", "stiffness": 100, "damping": 2, "max_force": 2.0, "max_velocity": 5.8, "target": -1.7}
}
```

- `anchor` and `axis` are in **world** space at the moment the scene is defined, the way you place a joint in CAD.
- types: `revolute` (hinge), `prismatic` (slider), `fixed`.
- joint value: revolute = angle (rad), prismatic = displacement (m). `initial` (optional, default 0) is the value in
  the scene's pose. LeKiwi is built folded at rest, but its zeros stay where the real arm's calibration puts them.
- `limits`: `[lower, upper]` or `null`.
- `drive`: a motor. `position` mode is a spring-damper toward `target` (a servo), `velocity` mode (stiffness 0) is a
  damper toward a target speed (a wheel motor). `max_force` caps it (N·m or N), optional `max_velocity` caps the
  speed a position drive commands (a real servo's top speed).

### Cameras

```json
{"id": "lekiwi/front_cam", "body": "lekiwi/base", "position": [0.125, -0.025, 0], "rotation": [...],
 "fov": 90, "width": 64, "height": 64}
```

Rigidly attached to `body` at a local offset, looking down its local -z with up = local +y (three.js convention).

---

## Part 2: The World (state and integration)

`physics/world.py`. A `World` holds **E environments** that share one scene structure, stepped in parallel on the
GPU (E = 1 for the editor, 256 for training). All state is tensors:

```
pos       (E, N, 3)     position of each body's center of mass
quat      (E, N, 4)     orientation
velocity  (E, N, 6)     [linear velocity v, angular velocity ω]
```

**Integrating velocities.** Gravity (+ optional n-body gravity). The gyroscopic term `ω × Iω` is integrated
implicitly with one Newton step in the body frame (Bullet), because explicit integration explodes for spinning bodies.

**Integrating positions.** `x += v h`, orientation with the exact exponential map `q ← exp(h ω) ⊗ q`: exact for any
angle, so no drift and no re-orthonormalization (this is why quaternions are used).

---

## Part 3: The Solver (how constraints work)

Joints and contacts are **constraints**, each a set of *rows*: a scalar velocity that must be 0 (joints), ≥ 0
(contacts, limits) or bounded (friction, motors).

```
row velocity  = G v            G: jacobian rows, v = all body velocities [v, ω]
impulse λ changes velocities by  Δv = M⁻¹ Gᵀ λ  =  W λ
effective inverse mass  K = G M⁻¹ Gᵀ
```

### The step: Box2D v3 "soft step"

```
detect contacts once
for each substep (h = dt / substeps):
    integrate velocities
    warm start      apply last step's impulses (they're usually almost right)
    solve           impulses with position correction ("bias")
    integrate positions
    relax           impulses again without bias (removes the energy the bias added)
restitution         bounces (only if anything in the scene can bounce)
```

### Joints: one exact global solve

All locked rows and all motor rows of all joints are solved **together, in one linear solve** (Baraff 1996): a
`6J × 6J` system (78 × 78 for LeKiwi), inverted per env by Cholesky every substep.

Why: solving joints one at a time (Gauss-Seidel), each joint only "feels" its two neighbouring links. The SO-101's
links weigh ~0.1 kg but hold up the whole arm, so its shoulder bent 5.7° under gravity. A global solve treats the
mechanism as one rigid whole: 0.00° bend, tip within 1-2 mm of the inverse kinematics target.

- **Locked rows** are rigid (`A Δλ = -(G v + bias · C)`), with Box2D's 60 Hz position-correction rate.
- **Motor rows** are PhysX's implicit spring-damper `F = k (x_t - x) + c (v_t - v)`, solved for the end-of-substep
  velocity: on the diagonal of A they add `γ = 1 / (h (c + h k))`. The commanded velocity is capped at `max_velocity`.
- **Saturation.** Motors can't exceed `max_force`. After clamping, the locked rows are re-solved for the motor impulse
  actually applied (`K_LL Δλ_L = -(r_L + K_LD Δλ_D)`), otherwise the joints and motors disagree and inject energy.
- **Limits** run after, per joint, along the free direction projected off the locked rows (Schur complement), so a
  motor pushing into a limit can't overrule it.

Angles come from a **swing-twist** split of the relative rotation: the twist about the axis is the joint angle, the
swing is the error the locked rows remove.

### Contacts: mass splitting

Contacts are solved Jacobi style, all at once, with **mass splitting** (Tonge 2012): a body touching k contacts acts
like k copies of mass m/k, one per contact, averaged afterwards. Each contact has 3 rows: normal (λ ≥ 0) and two
friction directions with the Coulomb cone `|λt| ≤ μ λn`. Soft contacts: min(120 Hz, substep rate / 4).

**Contact buffer.** Every candidate pair has fixed "slots" (box vs plane: 8, one per corner). Each step the 64
deepest touching slots per env go into a fixed-size buffer, and only those are solved. Shapes never change (CUDA
graphs), and a slot always means the same feature, so last step's impulse is handed back to it (warm starting).

**Sticky friction** (Catto's Solver2D "sticky", like PhysX friction anchors). Each contact remembers where it first
touched on both bodies, and friction pulls those points back together, so a held block can't creep out of the
gripper. The anchor resets when friction hits the Coulomb limit (real sliding) or drifts > 5 mm (a rolling wheel).

**Omni wheels.** A cylinder with `"omni": true` grips along its rolling direction (μ) but only has its passive
rollers' friction (0.02) along its axle. The two directions are clamped independently, which is what the rollers do,
so kiwi drive works: forward / strafe / spin within 2-5 % of the commanded speed.

**Soft contacts and light parts.** Soft contact stiffness is relative to the effective mass, so a contact sinks by
about F / (m_eff ω²). That's why the SO-101's 12 g jaw is simulated at 60 g. Through the servo's 345:1 gearbox the
motor's rotor makes it act far heavier (MuJoCo calls this armature). At 12 g, the squeeze went through the block.

### Collision detection

`physics/collision.py`, batched over (envs, pairs):

- plane vs sphere / box (8 corners) / cylinder (4 rim points per cap): exact.
- sphere vs sphere / box / cylinder: closest point on the solid, exact.
- box vs box: **separating axis test** over 15 axes. For a face axis, the contact polygon (reference face ∩ incident
  face) has its corners among 24 candidate features; for an edge axis, the closest points of the two edges.
- box vs cylinder: sampled rim points and box corners (good enough for wheels bumping blocks).

### Speed

Everything is tensor ops, so cost is mostly GPU kernel launches. `World.enable_cuda_graph()` and
`Renderer.enable_cuda_graph()` record a step / a camera render once and replay them as single launches. On an RTX
4070 laptop with the LeKiwi scene: ~18 ms per step for 1 env (≈ 0.9× realtime in the editor), **~6,500 env-steps/s
for 256 envs**, 2 cameras in ~1.8 ms for 1 env.

Checked against analytic answers:

| test | measured | expected |
|---|---|---|
| pendulum period | 1.425 s | 1.421 s |
| box sliding to rest (μ = 0.5, 2 m/s) | 0.404 m | 0.408 m |
| sphere sliding → rolling (2 m/s) | 1.424 m/s | 5/7 · 2 = 1.429 m/s |
| ball dropped 0.5 m, e = 0.8 | rebounds 0.296 m | 0.302 m |
| SO-101 arm under gravity | 0.00° joint bend | rigid |

---

## Part 4: LeKiwi

`robot/lekiwi.py`. [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) (Hugging Face / SIGRobotics-UIUC): a round
base on 3 omni wheels ("kiwi drive") with an [SO-101 arm](https://github.com/TheRobotStudio/SO-ARM100). Frame: +x
forward, +y up, +z to the robot's right.

```
base             cylinder r 12.5 cm, 1.5 kg; battery (0.3 kg) at the back as counterweight
wheels           3 × 4" omni wheels at 120° (back, front-left, front-right), velocity motors
arm              shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll -> gripper (jaw)
cameras          front_cam (base front edge, 15° down) and wrist_cam (side of the gripper), 64x64 each
```

The arm's joint positions, axes, limits and link masses come from the URDF (`so101_new_calib.urdf`, forward
kinematics of its zero pose), so joint angles match the real arm's calibration. Links are boxes between joints.
Every motor is a Feetech STS3215 servo: 2 N·m, 5.8 rad/s. The robot is built folded (`REST`), with each joint's
`initial` set so the zeros stay at the URDF's.

**Actions** (9 numbers in [-1, 1], in order): wheel_back, wheel_left, wheel_right (speeds, ±5.8 rad/s), then
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (absolute angles within the URDF limits).

---

## Part 5: The Pick-and-Place Environment

`robot/pick_place.py`. `PickPlaceEnv(num_envs, device, num_blocks)`. Each env has LeKiwi, colored 3.5 cm blocks and
matching colored target zones on the floor. The robot must carry block i into zone i.

- **reset**: random robot heading, blocks and zones 0.45-0.95 m away and at least 0.25 m apart, random block yaw.
- **step(action)**: sets joint targets, runs 6 physics steps (control at 10 Hz), returns `obs, reward, done, info`.
  Envs that finish are reset automatically.
- **observation**: `rgb` (E, 6, 64, 64) uint8: front camera RGB then wrist camera RGB, rendered by ray casting.
  `proprio` (E, 9): the 6 arm joint angles and 3 wheel speeds, normalized.
- **reward**: potential-based shaping, `r = Φ(s') - Φ(s) + 10 per delivered block`, with
  `Φ = -(|gripper - block| + 2 |block - zone|)`. For PPO; behavior cloning ignores it.
- **success**: every block resting inside its zone. Episodes end on success, a flipped base, or after 300 control
  steps per block.

---

## Part 6: Learning (`back/ml/`)

Pure RL from pixels almost never solves a long task like drive → grasp → carry → place (the reward is too sparse).
Real robot learning starts from demonstrations, so this does too.

**1. Scripted expert** (`ml/expert.py`). It cheats: it reads true positions from the simulator. A state machine per
env: approach → pre-grasp → lower → close → lift → stow → carry → place → release → retreat.
- driving: holonomic, straight at the stopping point while turning to face the goal, kiwi inverse kinematics to wheel
  speeds.
- arm: top-down 5-joint inverse kinematics (gripper straight down, pan toward the block, wrist roll aligns the jaws
  with the block's faces). The grasp pose is locked once the jaws close, so it doesn't chase a nudged block.
- targets are rate-limited like a real servo. 90 % of test episodes succeed.

**2. Collect** (`python -m ml.collect --envs 128 --episodes 500`) keeps only successful episodes and writes
`back/data/bc/`, one row per control step:

```
frames.npy   (T, 64, 64, 6) uint8     front RGB, then wrist RGB
proprio.npy  (T, 9) float32           joint readings
actions.npy  (T, 9) float32           what the expert did, in [-1, 1]
episode.npy  (T,) int32               episode id (frame stacks never cross episodes)
meta.json    {"frames": T, "episodes": n, "attempts": m}   only the first T rows are valid
```

**3. Policy** (`ml/policy.py`). Input: the last 3 frames of both cameras stacked (18 channels) + the 9 joint readings.
Nature-DQN conv encoder → 512 → 256 → 9 actions (tanh), plus a value head for PPO.

**4. Behavior cloning** (`python -m ml.train_bc`): MSE to the expert's actions with random-shift augmentation (DrQ).
Every 5000 steps it measures the real success rate in 64 envs and keeps the best as `back/checkpoints/bc.pt`
(`{"policy": state_dict, "step": int, "success": float}`).

**5. PPO** (`python -m ml.train_ppo --init checkpoints/bc.pt`): 256 envs, 32-step rollouts, GAE, clipped objective,
gaussian exploration around the BC actions, reward scaled by the running std of returns. Saves `checkpoints/ppo.pt`.

**6. Evaluate** (`python -m ml.eval --ckpt checkpoints/ppo.pt`) prints the success rate and records one episode to
`logs/policy_rollout/` to watch in Playback mode.

---

## Part 7: Recordings

`recording.py`. The scene once, then poses per frame:

```
logs/{name}/scene.json     the scene (shapes, colors, ids)
logs/{name}/{chunk}.bin    little-endian, 1000 frames per chunk:
  "SIMR"  seconds_per_frame f32  total_frames u32  frames_in_chunk u32  num_bodies u32
  per frame: time f32, then num_bodies × (px py pz qw qx qy qz) f32 in scene body order
```

`python -m ml.eval` writes one (`logs/policy_rollout/`).

---

## Part 8: Server and Editor

`api.py` (FastAPI, port 8080):

```
GET  /scenes                    {"scenes": [names]}
GET  /scenes/{name}             scene JSON
PUT  /scenes/{name}             save scene JSON
GET  /prefabs/lekiwi            LeKiwi bodies / joints / cameras at the origin
GET  /recordings                {"recordings": [names]}
GET  /recordings/{name}/scene   scene of a recording
GET  /recordings/{name}/{chunk} binary chunk
WS   /ws                        live simulation
```

Websocket messages:

```
client -> server   {"type": "play", "scene": {...}}   pause / resume / step / stop
                   {"type": "drive", "targets": {"lekiwi/shoulder_lift": 0.3, ...}}
                   {"type": "autopilot", "mode": "off" | "expert" | "policy"}
server -> client   {"type": "frame", "time", "poses": [[px,py,pz,qw,qx,qy,qz], ...], "joints": {id: value},
                    "cameras": [{"id", "width", "height", "rgb": base64}]}     ~30 per second
                   {"type": "status", "state": "running" | "paused" | "stopped", "autopilot"}
                   {"type": "error", "message"}
```

`live.py` holds the session: one world (physics + cameras as CUDA graphs on GPU) stepped in real time. The autopilot
runs the expert, or the trained policy (`checkpoints/ppo.pt`, else `bc.pt`) from the robot's cameras, at 10 Hz.

**The editor** (`front/`) is CAD-style: scene tree left, properties right, toolbar on top.

- **Edit**: add boxes / spheres / cylinders, insert a LeKiwi, click to select, move/rotate gizmo with snapping, joints
  between two shift-selected bodies, every field editable (rotation shown as XYZ degrees, stored as a quaternion).
- **Simulate**: Play sends the edited scene to the server. Drive LeKiwi with the keyboard:
  W/S forward/back, A/D strafe, Q/E turn, Y/U pan, H/J shoulder, N/M elbow, I/K wrist, O/L wrist roll,
  Space gripper. Or switch the autopilot to expert / policy. Both camera feeds and the joint values are shown live.
- **Playback**: pick a recording, play / pause / speed / scrub.
- **Camera**: left-drag orbits, ctrl+left-drag or right-drag pans (camera and pivot together), wheel zooms at a
  constant speed (pushing the pivot ahead when you pass it).

---

## Running Everything

```bash
pip install -r back/requirements.txt
python main.py                       # backend (:8080) + editor (http://localhost:5173)

# learning (run in tmux, from back/)
python -m ml.collect --envs 128 --episodes 500
python -m ml.train_bc --steps 30000
python -m ml.train_ppo --init checkpoints/bc.pt --iters 2000
python -m ml.eval --ckpt checkpoints/ppo.pt
```

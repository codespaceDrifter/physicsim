# Physics Simulation - How It Works

This explains the whole codebase from the ground up: the physics engine, the robot, the learning pipeline, the
server and the editor. Read it top to bottom.

```
back/                      python, run everything from here as modules:  cd back && python -m <module>
  physics/                 the engine
    math3d.py              batched quaternion / vector math
    world.py               World: bodies, integration, the step loop, CUDA graphs
    joints.py              revolute / prismatic / fixed joints, drives, limits
    contacts.py            contact solving: non-penetration, friction, restitution
    collision.py           narrowphase: plane / sphere / box / cylinder pairs
    render.py              camera pixels by batched ray casting
  robot/
    tofubot.py             the ToFuBot model (bodies, joints, camera)
    pick_place.py          batched pick-and-place environment (reset, step, reward, pixels)
  ml/
    expert.py              scripted expert that reads true simulator state
    collect.py             expert demonstrations -> dataset
    policy.py              pixel policy network (CNN)
    train_bc.py            behavior cloning
    train_ppo.py           PPO fine-tuning
    rollout.py / eval.py   success rate + recording a policy episode
  example_scenes.py        three_body, billiards, tofubot_pick scene builders
  simulate.py              writes scenes/*.json and records the passive scenes into logs/
  recording.py             recording format (playback)
  live.py                  real-time simulation session for the editor
  api.py                   FastAPI server: scenes, prefabs, recordings, websocket
  scenes/                  scene JSON files the editor opens / saves
front/                     Three.js + Vite editor (edit / simulate / playback)
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
  "name": "tofubot_pick",
  "settings": {"dt": 0.0167, "substeps": 4, "gravity": [0, -9.81, 0], "G": 0.0},
  "bodies": [ ... ],
  "joints": [ ... ],
  "cameras": [ ... ]
}
```

- `dt`: seconds per world step. `substeps`: solver substeps per step (Part 3).
- `gravity`: uniform gravity acceleration (the "mechanical level": magic g, no attraction between objects).
- `G`: gravitational constant for attraction between every pair of bodies (the "planetary level"). 0 = off.

### Bodies

```json
{
  "id": "tofubot/chassis",
  "shape": {"type": "box", "size": [0.3, 0.06, 0.18]},
  "position": [0, 0.07, 0], "rotation": [1, 0, 0, 0],
  "velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
  "density": 460, "static": false, "collide": true, "group": 1,
  "friction": 0.8, "restitution": 0.0, "color": "#f2ead3"
}
```

- shapes: `box` (`size` = full side lengths), `sphere` (`radius`), `cylinder` (`radius`, `height`, axis = local +y),
  `plane` (infinite, normal = local +y, must be static). Real engines (MuJoCo, Bullet) use exactly these primitives
  for robots because every pair has an exact, fast collision test.
- mass = volume × density. The inertia tensor of each primitive is diagonal in its own frame (box `m(y²+z²)/12`...,
  sphere `2/5 m r²`, cylinder `m(3r²+h²)/12` and `m r²/2` about its axis).
- `static`: infinite mass. Never moves.
- `collide: false`: visual only (the target zones).
- `group`: bodies that share a nonzero group never collide with each other (robot parts don't self-collide).
- `friction` of a pair = `sqrt(fa · fb)`, `restitution` = `max(ra, rb)` (Box2D's mixing rules).

### Joints

```json
{
  "id": "tofubot/shoulder", "type": "revolute",
  "body_a": "tofubot/turret", "body_b": "tofubot/upper_arm",
  "anchor": [0.03, 0.135, 0], "axis": [0, 0, -1],
  "limits": [-0.6, 1.9],
  "drive": {"mode": "position", "stiffness": 40, "damping": 1.5, "max_force": 4, "target": 0}
}
```

- `anchor` and `axis` are in **world** space at the moment the scene is defined, the way you place a joint in CAD.
  The engine converts them into a frame fixed on each body.
- types: `revolute` (hinge: 1 rotation about the axis), `prismatic` (slider: 1 translation along the axis), `fixed`.
- joint value: revolute = angle (rad), prismatic = displacement (m), both 0 in the scene's pose.
- `limits`: `[lower, upper]` or `null`.
- `drive`: a motor. `position` mode is a spring-damper toward `target` (a servo). `velocity` mode
  (stiffness 0) is a damper toward a target speed (a wheel motor). `max_force` caps it (N·m or N).

### Cameras

```json
{"id": "tofubot/cam", "body": "tofubot/chassis", "position": [0.14, 0.10, 0], "rotation": [...],
 "fov": 100, "width": 64, "height": 64}
```

Rigidly attached to `body` at a local offset. It looks down its local -z with up = local +y (three.js convention).

---

## Part 2: The World (state and integration)

`physics/world.py`. A `World` holds **E environments** that share one scene structure, stepped in parallel on the
GPU: E = 1 for the editor, E = 256 for training. All state is tensors:

```
pos       (E, N, 3)     position of each body's center of mass
quat      (E, N, 4)     orientation
velocity  (E, N, 6)     [linear velocity v, angular velocity ω]
```

The velocity is one 6-vector per body so a constraint can read both its bodies with one gather and write them with
one scatter.

**Integrating velocities.** Gravity, plus optional n-body gravity `a_i = G Σ m_j (x_j - x_i) / r³`. Rotation needs
the gyroscopic term `ω × Iω`, which explodes if integrated explicitly for spinning bodies. It is integrated
implicitly with one Newton step in the body frame (what Bullet does).

**Integrating positions.** `x += v h`, and orientation with the exact exponential map
`q ← exp(h ω) ⊗ q` (a rotation of angle |ω|h about ω, exact for any angle, then renormalized against float
round-off). This is why quaternions are used: no drift and no re-orthonormalization.

---

## Part 3: The Solver (how constraints work)

Joints and contacts are **constraints**. Each is a set of *rows*. A row is a scalar velocity that must be 0
(joints), ≥ 0 (contacts, limits) or bounded (friction, motors):

```
row velocity  = J_a · V_a + J_b · V_b         J: (rows, 6) jacobian per body, V = [v, ω]
impulse λ changes velocities by  ΔV_a = M_a⁻¹ J_aᵀ λ  =  W_a λ
```

The solver repeatedly computes the impulse that makes each row's velocity right and applies it. That's
**sequential impulses** (Box2D, Bullet, PhysX).

### The step: Box2D v3 "soft step"

```
detect contacts once
for each substep (h = dt / substeps):
    integrate velocities
    warm start      apply last step's impulses (they're usually almost right)
    solve           impulses with position correction ("bias")
    integrate positions
    relax           impulses again without bias (removes the energy the bias added)
restitution         bounces
```

**Soft constraints.** Position errors are corrected like a stiff, damped spring (`hertz`, `damping ratio`) solved
implicitly, instead of a raw position fix-up. Each impulse is
`Δλ = -m_eff · mass_scale · (Jv + bias_rate · C) - impulse_scale · λ`, which is stable at any mass. Contacts use
min(120 Hz, substep rate / 4). Above a quarter of the substep rate, resting stacks jitter. Joints use 60 Hz.

**Speculative contacts.** Contacts are created up to 2 cm before touching. A contact with a gap only removes the
closing speed that would make the bodies overlap, so fast objects don't tunnel through each other.

### Joints: graph coloring

Joints are solved Gauss-Seidel: each sees the velocities the previous one produced. That's needed for chains like an
arm. To do it in parallel, joints are **colored** so joints of the same color share no moving body. Each color is
solved as one batch, colors one after another. GPU PhysX and Box2D do the same.

For each joint, the locked rows (e.g. 3 linear + 2 angular for a hinge) are solved together as one 6×6 block.
The free row (the hinge angle) carries the **drive** and the **limits**. Their impulses are projected so they don't
pull on the locked rows (the Schur complement `e_free - K_LL⁻¹ K_L,free`). Without that, a strong motor pushing into a
limit gets undone by the hinge's own anchor rows every pass.

The drive is PhysX's implicit spring-damper: `F = k (x_target - x) + c (v_target - v)`, solved for the end-of-substep
velocity, so even stiff servos stay stable.

Angles come from a **swing-twist** split of the relative rotation: the twist about the axis is the joint angle, and
the swing is the error the locked rows remove.

### Contacts: mass splitting

Contacts are solved Jacobi style, all at once, with **mass splitting** (Tonge 2012). A body touching k contacts acts
like k copies of mass m/k, one per contact, averaged afterwards. That converges like Gauss-Seidel with no ordering,
which is how GPU solvers do contacts.

Each contact point has 3 rows: normal (λ ≥ 0, pushes apart) and two friction directions with the Coulomb cone
`|λ_t| ≤ μ λ_n`.

**Contact buffer.** Every candidate pair has fixed "slots", for example box-vs-plane has 8 (one per corner). Only a
few touch at once, so each step the 64 deepest touching slots per env are copied into a fixed-size buffer and only
those are solved. Shapes never change, which makes CUDA graphs possible. A slot always means the same feature
("corner 3 of this box"), so last step's impulse is handed back to it (warm starting).

**Sticky friction** (Catto's Solver2D "sticky", like PhysX friction anchors). Velocity-level friction alone lets a
held block creep slowly out of a gripper. Each contact therefore remembers where it first touched on both bodies,
and friction also pulls those two points back together. When friction hits the Coulomb limit (real sliding), or
an anchor drifts more than 5 mm from the contact point (a rolling wheel), the anchor resets.

**Restitution** runs after the substeps. Contacts that were closing faster than 0.5 m/s get pushed to
`-e × (closing speed)`.

### Collision detection

`physics/collision.py`, batched over (envs, pairs):

- plane vs sphere / box (8 corners) / cylinder (4 rim points per cap, deepest first): exact.
- sphere vs sphere / box / cylinder: closest point on the solid, exact.
- box vs box: **separating axis test** over 15 axes (3 + 3 face normals, 9 edge × edge). For a face axis, the contact
  polygon is the intersection of the reference face rectangle and the incident face, and its corners are always among
  24 candidates: incident corners inside the rectangle, rectangle corners inside the incident face, and edge × edge
  crossings. For an edge axis it's the closest points of the two edges.
- box vs cylinder: approximated by sampling rim points and box corners (good enough for wheels bumping blocks).

### Speed

Everything is tensor ops, so the per-step cost is mostly kernel launches. `World.enable_cuda_graph()` records one
step and replays it as a single launch. Measured on an RTX 4070 laptop with the ToFuBot scene: about 15 ms per step
for 1 env (realtime is 16.7 ms) and about **12,000 env-steps/s for 256 envs**.

The engine is checked against analytic answers:

| test | measured | expected |
|---|---|---|
| pendulum period | 1.425 s | 1.421 s |
| box sliding to rest (μ = 0.5, 2 m/s) | 0.404 m | 0.408 m |
| sphere sliding → rolling (2 m/s) | 1.424 m/s | 5/7 · 2 = 1.429 m/s |
| ball dropped 0.5 m, e = 0.8 | rebounds 0.296 m | 0.302 m |
| head-on billiard hit | e ≈ 0.89, momentum 0.99 | e = 0.95 minus friction |

---

## Part 4: ToFuBot

`robot/tofubot.py`. A small cart with an arm. Frame: +x forward, +y up, +z to the robot's right.

```
chassis          30 x 6 x 18 cm box, 1.5 kg
front wheels     chassis --steer_fl/fr (revolute about +y, position servo, ±0.6 rad)--> knuckle
                 knuckle --spin_fl/fr (free revolute)--> wheel
rear wheels      chassis --drive_rl/rr (revolute about -z, velocity motor, ±15 rad/s)--> wheel
arm              chassis --turret (about +y)--> turret --shoulder (about -z)--> upper arm (16 cm)
                 --elbow (about -z)--> forearm (14 cm)
gripper          forearm --finger_l / finger_r (prismatic, 0 = closed, 0.035 m = open)--> fingers
camera           64x64, 100° fov, fixed on the chassis front, 17 cm up, pitched 40° down
```

Signs: steer + = left, wheel + = forward, shoulder/elbow + = lean forward (0 = arm straight up).

**Actions** (9 numbers in [-1, 1], in this order) are mapped linearly to joint targets: steer_fl, steer_fr
(individual steering), drive_rl, drive_rr (individual wheel speeds), turret, shoulder, elbow (absolute angles),
finger_l, finger_r (opening).

---

## Part 5: The Pick-and-Place Environment

`robot/pick_place.py`. `PickPlaceEnv(num_envs, device, num_blocks)`. Each env has the robot, colored 4 cm blocks and
matching colored 14 cm target zones on the floor. The robot must carry block i into zone i.

- **reset**: random robot heading, blocks and zones 0.45-0.95 m away and at least 0.25 m apart, random block yaw.
- **step(action)**: sets joint targets, runs 6 physics steps (control at 10 Hz), returns
  `obs, reward, done, info`. Envs that finish are reset automatically.
- **observation**: `rgb` (E, 3, 64, 64) uint8 from the robot's own camera (rendered by ray casting in
  `physics/render.py`), and `proprio` (E, 9): the 7 servo positions plus the 2 rear wheel speeds, normalized. Real
  robots have joint encoders, so the policy gets them too.
- **reward**: potential-based shaping, `r = Φ(s') - Φ(s) + 10 per delivered block`, with
  `Φ = -(|gripper - block| + 2 |block - zone|)`. This reward is for PPO. Behavior cloning ignores it.
- **success**: every block resting inside its zone. Episodes end on success, if the robot flips, or after 300 control
  steps per block.

---

## Part 6: Learning (`back/ml/`)

Reinforcement learning from pixels alone almost never solves a long task like drive → grasp → carry → place. The
reward is too sparse. Real robot learning starts with demonstrations, so this does too:

**1. Scripted expert** (`ml/expert.py`). It cheats: it reads true positions from the simulator. It's a state machine
per env: approach → pre-grasp → lower → close → lift → stow → carry → place → release → retreat. It drives with
pure pursuit and true Ackermann steering: each front wheel gets its own angle and each rear wheel its own speed, so
every wheel rolls around the same turn center. It backs up in a 3-point turn when the target is inside the turning
circle. It reaches with 2-link inverse kinematics and rate-limits its targets like a real servo. It succeeded in 48 of
48 test episodes.

**2. Collect** (`python -m ml.collect --envs 128 --episodes 500`) keeps only successful expert episodes and
writes `back/data/bc/`, one row per control step:

```
frames.npy   (T, 64, 64, 3) uint8     what the camera saw
proprio.npy  (T, 9) float32           joint readings
actions.npy  (T, 9) float32           what the expert did, in [-1, 1]
episode.npy  (T,) int32               episode id (frame stacks never cross episodes)
meta.json    {"frames": T, "episodes": n, "attempts": m}   only the first T rows are valid
```

500 episodes ≈ 60k frames ≈ 0.75 GB.

**3. Policy** (`ml/policy.py`). Input: the last 3 camera frames stacked (9 channels, so it sees motion) plus the 9 joint
readings. Nature-DQN conv encoder → 512 → 256 → 9 actions (tanh), plus a value head for PPO.

**4. Behavior cloning** (`python -m ml.train_bc`) regresses the expert's actions from pixels (MSE), with random-shift
augmentation (DrQ: pad 4 px, random crop) so it doesn't overfit exact pixel positions. Every 5000 steps it runs 64
envs to measure the real success rate and keeps the best as `back/checkpoints/bc.pt`
(`{"policy": state_dict, "step": int, "success": float}`).

**5. PPO** (`python -m ml.train_ppo --init checkpoints/bc.pt`) fine-tunes it on the shaped reward: 256 envs,
32-step rollouts, GAE, clipped objective, gaussian exploration around the BC actions, reward scaled by the running
std of returns. Saves `back/checkpoints/ppo.pt` in the same format.

**6. Evaluate** (`python -m ml.eval --ckpt checkpoints/ppo.pt`) prints the success rate and records one episode
to `logs/policy_rollout/` to watch in Playback mode.

---

## Part 7: Recordings

`recording.py`. A recording is the scene once plus poses per frame, so meshes aren't stored every frame:

```
logs/{name}/scene.json     the scene (shapes, colors, ids)
logs/{name}/{chunk}.bin    little-endian, 1000 frames per chunk:
  "SIMR"  seconds_per_frame f32  total_frames u32  frames_in_chunk u32  num_bodies u32
  per frame: time f32, then num_bodies × (px py pz qw qx qy qz) f32 in scene body order
```

`python -m simulate` writes the example scenes to `back/scenes/` and records `three_body` (3000 s) and `billiards`
(10 s).

---

## Part 8: Server and Editor

`api.py` (FastAPI, port 8080):

```
GET  /scenes                    {"scenes": [names]}
GET  /scenes/{name}             scene JSON
PUT  /scenes/{name}             save scene JSON
GET  /prefabs/tofubot           ToFuBot bodies / joints / cameras at the origin
GET  /recordings                {"recordings": [names]}
GET  /recordings/{name}/scene   scene of a recording
GET  /recordings/{name}/{chunk} binary chunk
WS   /ws                        live simulation
```

Websocket messages:

```
client -> server   {"type": "play", "scene": {...}}   pause / resume / step / stop
                   {"type": "drive", "targets": {"tofubot/steer_fl": 0.3, ...}}
                   {"type": "autopilot", "mode": "off" | "expert" | "policy"}
server -> client   {"type": "frame", "time", "poses": [[px,py,pz,qw,qx,qy,qz], ...], "joints": {id: value},
                    "cameras": [{"id", "width", "height", "rgb": base64}]}     ~30 per second
                   {"type": "status", "state": "running" | "paused" | "stopped", "autopilot"}
                   {"type": "error", "message"}
```

`live.py` holds the session: one world (CUDA graph on GPU) stepped in real time. The autopilot runs the expert, or the
trained policy (`checkpoints/ppo.pt`, else `bc.pt`) from the robot's camera, at 10 Hz.

**The editor** (`front/`) is CAD-style: scene tree on the left, properties on the right, toolbar on top.

- **Edit**: add boxes / spheres / cylinders, insert a ToFuBot, click to select, move/rotate gizmo with snapping,
  create joints between two shift-selected bodies, edit every field (rotation is shown as XYZ degrees but stored as a
  quaternion), see volume and mass. Standard views (Top / Front / Right / Iso), perspective / ortho. Save to
  `back/scenes/`.
- **Simulate**: Play sends the edited scene to the server. Drive ToFuBot with the keyboard (W/S drive, A/D steer,
  Y/U turret, H/J shoulder, N/M elbow, Space gripper), or switch the autopilot to expert / policy. The robot's
  camera feed and joint values are shown live. Stop returns to the unmodified scene.
- **Playback**: pick a recording, play / pause / speed / scrub.

---

## Running Everything

```bash
pip install fastapi uvicorn          # torch, numpy already installed
cd back
python -m simulate                   # scenes + example recordings (three_body takes a while)
python -m api                        # server on :8080
cd ../front && npm run dev           # editor on http://localhost:5173

# learning (run in tmux, from back/)
python -m ml.collect --envs 128 --episodes 500
python -m ml.train_bc --steps 30000
python -m ml.train_ppo --init checkpoints/bc.pt --iters 2000
python -m ml.eval --ckpt checkpoints/ppo.pt
```

# Physics Simulation - How It Works

This explains the whole codebase from the ground up: the physics engine, the robot, the learning pipeline, the
server and the editor. Read it top to bottom.

```
back/                      python, run everything from here as modules:  cd back && python -m <module>
  physics/                 the engine
    math3d.py              batched quaternion / vector math, SPD inverse
    world.py               World: bodies, integration, the step loop, CUDA graphs
    joints.py              revolute / prismatic / fixed joints, motors, limits (one exact solve per articulation)
    contacts.py            contact solving: non-penetration, friction (incl. omni wheels), restitution
    collision.py           narrowphase: plane / sphere / box / cylinder / capsule pairs
    render.py              camera pixels by batched ray casting
  robot/
    lekiwi.py              the LeKiwi model (omni base + SO-101 arm, cameras)
    env.py                 base for batched LeKiwi tasks (robots, actions, pixels, proprio)
    course.py              the course: one arena, five stages (drive, pick, bring, race, speedrun)
    mjcf.py                MuJoCo MJCF importer (bodies, hinges + armature, defaults, collision geoms, actuators)
    g1.py                  the Unitree G1 humanoid (imported from MuJoCo Menagerie), built in any pose
    humanoid.py            base for G1 courses: robots, stages, walking, locomotion rewards, camera, observations
    warehouse.py           the humanoid course: walk, pick, bring, speedrun with boxes (Part 9)
    assets/unitree_g1/     g1_mjx.xml + its BSD-3 license
  ml/                      the learning pipeline (plan and reasoning: ../DOC.md)
    expert.py              scripted expert that reads true simulator state (baseline, editor autopilot)
    teacher.py             teacher: MLP actor + critic on privileged state
    bc_teacher.py          bootstrap the teacher: clone the scripted expert (+ a driving teacher), expert DAgger
    train_teacher.py       PPO on the whole course (stage mixing, self-play), from scratch or from the clone
    vla.py                 VLA v1: small transformer, cameras + command words + proprio -> action chunks
    rollout.py             drivers (teacher / VLA / expert) and per-stage evaluation
    distill.py             teacher -> VLA: behavior cloning, then DAgger rounds
    train_grpo.py          GRPO fine-tune of the VLA on episode outcomes
    eval.py                the results table: every model on every stage
    humanoid_policy.py     the humanoid model: one transformer (proprio history + words + camera), state decoder, critic
    train_humanoid.py      PPO on the warehouse course (asymmetric critic, state-decoder loss)
    record_humanoid.py     record the humanoid on a stage (a .simrec file to watch in playback)
  write_scenes.py          writes scenes/course_{n}_{stage}.json (course_1_drive ...) and scenes/warehouse.json
  recording.py             the recording file format (.simrec: scene + meshes + frames, for playback)
  record_games.py          pre-recorded races for the web demo (default teacher vs teacher) (logs/game_{i}.simrec + .mp4)
  live.py                  real-time simulation session for the editor
  api.py                   FastAPI server: scenes, prefabs, recordings, websocket
  scenes/                  scene JSON files the editor opens / saves
front/                     Three.js + Vite editor (edit / simulate / playback)
main.py                    starts back (api) + front (vite) together
logs/{name}/               recordings
back/data/distill/         distillation dataset (gitignored)
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
  "name": "course_4_race",
  "settings": {"dt": 0.0167, "substeps": 4, "gravity": [0, -9.81, 0], "G": 0.0, "max_contacts": 128, "stage": "race"},
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
  "shapes": [{"type": "cylinder", "radius": 0.0508, "height": 0.025, "omni": true,
              "position": [0, 0, 0], "rotation": [1, 0, 0, 0]}],
  "position": [0.05, 0.0508, -0.0866], "rotation": [w, x, y, z],
  "velocity": [0, 0, 0], "angular_velocity": [0, 0, 0],
  "density": 493, "static": false, "collide": true, "group": 1,
  "friction": 1.0, "restitution": 0.0, "color": "#2b2b2b"
}
```

- `shapes`: one or more (MuJoCo's geoms), each at a `position` / `rotation` in the body frame: `box` (`size` = full
  side lengths), `sphere` (`radius`), `cylinder` (`radius`, `height`, axis = local +y, optional `"omni": true` for an
  omni wheel, Part 3), `capsule` (`radius`, `length` between its two cap centers, axis = local +y), `plane` (infinite,
  normal = local +y, must be static and alone). MuJoCo and Bullet use exactly these primitives for robots because
  every pair has an exact, fast collision test. A body may have no shapes (a link that never touches anything).
- the body frame is its center of mass with principal axes (inertia diagonal). mass = volume × `density`, summed over
  the shapes (which must then balance around the origin), or explicit `"mass"` + `"inertia"` `[Ixx, Iyy, Izz]`
  (imported robots, whose links aren't their collision shapes).
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

### Joints: one exact solve per articulation

All locked rows and all motor rows of a mechanism are solved **together, in one linear solve** (Baraff 1996): a
`6J × 6J` system (78 × 78 for LeKiwi, 174 × 174 for the G1), factorized per env by Cholesky once per
step. Separate robots share no rows, so the system is block diagonal: each articulation (bodies connected through
joints) is its own block, batched over envs × articulations. Two robots cost 2 blocks instead of one 4× bigger
matrix. The factorization runs in float64: a humanoid's light links between heavy ones (a 74 g ankle link) give
condition numbers float32 Cholesky fails on. (So TF32 matmuls must stay off: they break the system's symmetry.)

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
- capsules (segments swept by a sphere): plane (the two cap spheres), sphere (closest point on the segment), capsule
  (closest points of the segments + each end against the other segment, so parallel capsules rest on two points),
  box (both ends + the segment's deepest point, found by ternary search: the box's signed distance is convex along
  the segment). Checked against brute-force sampling: depth within 0.06 mm.

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

## Part 5: The Tasks

`robot/env.py` is the base: every tensor has a robot axis R (`obs["rgb"]` (E, R, 6, 64, 64), `obs["proprio"]`
(E, R, 9), actions (E, R, 9), reward (E, R)).

- **observation**: `rgb`: front camera RGB then wrist camera RGB, rendered by ray casting. `proprio`: the 6 arm joint
  angles and 3 wheel speeds, normalized.
- **step(action)**: sets joint targets, runs 6 physics steps (control at 10 Hz), returns `reward, done, info`.
  Envs that finish are reset automatically. Nothing is rendered: call `observe()` (pixels) or `state()` after the
  step when you need them (the teacher never renders).

### The course

`robot/course.py`, `CourseEnv(num_envs, device, stage_probs)`. One arena for everything: walled 2.4 × 1.6 m, two
LeKiwis (orange `lekiwi/`, teal `lekiwi2/`), six pink 3.5 cm blocks, a 40 cm home zone at each end, and a goal
marker per robot (`goal/r`: a pad, plus `goal/r/front`, a dot on the edge the robot must face). Every env samples a
stage at reset from `stage_probs` (5 numbers, STAGES order), so all stages run in one batched world (both robots
exist in every stage; unused markers sit under the floor):

| stage | each robot must | episode | reward |
|---|---|---|---|
| drive | put its base on its goal pad (5 cm, 10°) | 15 s | Φ = -(distance + 0.2 × heading error), +10 on success |
| pick | lift a block 5 cm, held between the jaws | 30 s | Φ = -(grip → nearest block + 2 × missing lift), +10 |
| bring | get a block resting in its home zone | 45 s | Φ = -(grip → block + 2 × block → home), +10 |
| race | have more blocks home than the other robot after 60 s (stealing allowed) | 60 s | 10 × (my score change - theirs) + Φ as in bring |
| speedrun | robot 0 alone (robot 1 parked south of the arena): all 6 blocks home | 120 s | 10 × score change - 0.02 per step + Φ |

**Speedrun subcourses** (reverse curriculum): a speedrun episode can start with k = 0 … 5 blocks already resting in
robot 0's home zone (`env.home_probs` (6,): probability of each k; `info["start_home"]`: an episode's k). The end of a
speedrun can then be practiced before the whole of it.

Rewards are per robot, `r = Φ(s') - Φ(s) + bonus` (potential-based shaping, for PPO). An episode ends when every
active robot succeeded (not in the race), a base flips past 60°, or time runs out. `info`: `success` (E, R) (race:
won), `score` (E, R) blocks home, `stage` (E,).

- **reset**: drive: robots anywhere in the arena, random headings; each goal is an offset from the robot's start on
  the grid the command can say exactly (forward / right in 0.1 m steps up to 1 m, a turn in 15° steps), resampled
  until it lies in the arena 30 cm clear of blocks, robots and the other goal.
  Other stages: each robot in front of its zone (random sideways offset and heading), blocks scattered in the middle
  with random yaw.
- **command(e, r)**: the stage as text, e.g. "move 0.5 m forward and 0.3 m right, then turn 90° right" (drive, from
  the goal relative to the start pose) or "win the race".
- **state()** (E, R, 62): privileged state for the teacher, in the robot's own frame (forward, right): proprio (9),
  base velocity (3), grip point (3), per block position + sin/cos of 4 × relative yaw (30, cubes repeat every 90°;
  blocks sorted nearest first), own and other home (4), other robot position + relative heading (4), stage one-hot
  (5), drive goal (3), time left in the episode 1 → 0 (1).
- **copy_envs(src, dst)**: right after a reset, make envs `dst` exact copies of `src` (GRPO's groups).
- **block_yaw()** (E, B): each block's yaw mod 90°, read off whichever body axis lies most horizontal, so a block that
  tumbled onto a side face reads right too (the quaternion's yaw alone assumes it is upright).

---

## Part 6: Learning (`back/ml/`)

The plan and the reasoning behind it are in `DOC.md`; the PPO explanation in MINDOLOGY `writing/training/PPO.md`.
Four steps, each starting from the previous one's model:

```
0. python -m ml.train_teacher --hours 1       PPO from scratch: learns to drive -> a driving teacher (drive rows)
1. python -m ml.bc_teacher --drive-teacher <driving teacher> --dagger-rounds 3 --dagger-steps 1200 \
       --dagger-probs 0,1,1,1,3 --out checkpoints/teacher_bc.pt    clone the expert, 3 speedrun-weighted DAgger rounds
                                              (keep the best round's checkpoint as checkpoints/teacher.pt)
   python -m ml.train_teacher --init checkpoints/teacher.pt ...      optional PPO fine-tune of the clone
2. python -m ml.distill                       cloning + DAgger        -> checkpoints/vla_sft.pt
3. python -m ml.train_grpo                    GRPO on outcomes        -> checkpoints/vla_grpo.pt
4. python -m ml.eval --vla checkpoints/vla_sft.pt checkpoints/vla_grpo.pt   -> the results table
```

**Scripted expert** (`ml/expert.py`), a baseline row, the teacher's source of demonstrations and DAgger labels, and
an editor autopilot. It cheats: it reads true positions from the simulator. It is memoryless: every step it reads what
to do from the state alone, no phase, timer or last command (with hidden state, DAgger labels followed the expert's
own script instead of the clone's state and every round made the clone worse):
- holding a block (a block's center above 4.2 cm, within 6 cm of the grip point): stow (arm to the carry pose) →
  carry (drive in front of the first free home slot, facing the wall, within 6 cm / 0.25 rad) → place (tool tip over
  the slot, down to 4 cm) → release (once the tip is below 5.5 cm and within 4 cm of the slot).
- empty jaws: raise (a tool tip below 7 cm goes up first) → turn (within 0.6 m of its home, facing more than 100° away
  from the arena and the next block behind it: spin left in place, so a camera-only student sees the blocks left
  instead of guessing which way to turn) → approach (drive to the stopping point of the nearest block
  that isn't home and isn't in the other robot's jaws, so it steals too) → pre (tip above the block) → lower → close
  (once the tip is down) → lift (once the jaw has shut onto the block); a jaw shut on nothing opens again.
Home slots: 6 per zone (2 rows × 3, back row first); the base stops in front of the zone and the arm reaches in, so it
never drives over delivered blocks (driving in kept shoving them into the corners, where the arm couldn't grab them).
Driving: holonomic, kiwi inverse kinematics, each other block within 25 cm pushes the base away (the base rides 10 cm
off the floor; a block it drives at slides under it). Carrying turns at most 0.8 rad/s (faster spins flung blocks out
of the jaws). Arm: top-down inverse kinematics; it aims at a point at most (0.15 rad, 3 cm, 3 cm) past where its tool
tip is now (pure pursuit), so the tip moves in straight lines (joint-space moves swept blocks under the base) and the
label is a clear function of the state. Checks accept 1–2 cm (a clone's error; 5 mm checks left clones waiting
forever). A slot out of the arm's reach (25 cm) is driven to, not reached for; blocks resting in or just outside its
zone don't push the carrying base off its stand. It only fetches blocks, so it idles in the drive stage. Pick / bring:
100 %, speedrun: 99 % in ~560 of 1200 steps.

**0. Bootstrapping the teacher** (`ml/bc_teacher.py`). PPO from scratch learns the drive stage (~90 % in 25M steps)
but random exploration almost never finds a grasp. So the teacher is the expert's clone: (state, expert action) pairs
on the block stages (512 envs × 2400 steps), plus a driving teacher's actions on drive-stage states, behavior-cloned
into the actor (MSE, 80k steps). `--dagger-rounds`: then the clone drives, the expert labels the states it reaches,
and a fresh clone trains on everything (fine-tuning the last clone briefly instead left it fitting even its old data
twice as badly); `--dagger-probs` / `--dagger-steps` weight the stages and make rounds as long as a speedrun episode,
so late-episode states get labeled too; `--model` starts the rounds from an existing clone. Lessons: adding the
previous action to the state made cloning worse (copycat problem); round-to-round results vary (speedrun 88 %, 69 %,
81 % over three rounds), so keep the best round.

**1. Teacher** (`ml/teacher.py`, `ml/train_teacher.py`). The blocks are a set, not slots: each block is a token
(its 5 state numbers, "in my home zone" / "in the other home zone" flags, its offset from the gripper), encoded with
the rest of the state (context, 128 wide), pooled by attention (query = context, 4 heads) and by max over blocks;
[context, attention pool, max pool] → 3 × 512 (ELU) → 9 means + 9 learned log stds (gaussian). The means are residual:
the action holding the current pose (proprio, normalized like the actions, `robot.env.HOLD`) + the network's output
(as absolute targets its ~0.03 errors meant 1–2 cm at the gripper and the arm stalled). The context does not see the
time left (the expert ignores it and finished early, so its clone fell apart on a late clock). A separate critic
of the same shape → 1. Inputs normalized by a running mean / std, pooled over the block slots so block order doesn't
matter. (A plain MLP over the 6 blocks sorted by distance had to work out which slot was still worth fetching; with
blocks already home it mostly didn't: its clone made 0.77 blocks per speedrun, the set encoder's clone 1.0.) PPO: 4096 envs × 2 robots, 32-step rollouts, GAE (γ 0.99, λ 0.95, timeouts are treated as true ends; time
left is in the state, though the networks ignore it), 5 epochs × 4 minibatches, clip 0.2, value 0.5, entropy 0.001, gradient norm 1, reward scaled by
the running std of the discounted return, learning rate adapted to keep the KL near 0.01 (as in legged_gym). The
speedrun's parked robot is left out of the update. Stage mixing: each stage's success rate is an EMA over iterations;
its sampling weight is max(1 − success, 0.1); pick unlocks when drive reaches 80 %, bring after pick, race and
speedrun after bring. Saves `{"model", "opt", "iter", "success", "unlocked", "args"}` every 25 iterations (`--resume`
continues), logs one JSON line per iteration to `logs/teacher.jsonl` (success, mean episode return and sampling
probability per stage, losses, KL, lr, std), stops after `--hours`. With `--init` (the clone): every stage unlocked,
the input normalizer frozen at the clone's statistics, a critic-only warmup (15 iterations), action std 0.01, a
separate actor learning rate starting tiny (a 0.01 std makes the KL very sensitive: one full-size step wrecked the
clone) and the BC loss on the demos kept at weight 0.998^iteration (DAPG). `--home-curriculum`: reverse curriculum over the speedrun
subcourses: only k = 5 (one block left) at first; k's weight is max(1 − its success, 0.1), and k − 1 unlocks when k
succeeds 50 % of the time, down to the full run; the speedrun's own stage weight then follows the full run's success.
This took the full speedrun from 1.0 to ~3 blocks per run in 3 h (it starts from the set-encoder clone).

**2. VLA v1** (`ml/vla.py`), ~11M parameters, trained from scratch. One sequence per robot: 64 front camera patches
+ 64 wrist camera patches (8 × 8 pixels, one linear projection per camera), ≤ 16 command words (word vocabulary of
the command templates, numbers are words), 1 proprio token, 5 action queries; learned position embeddings; 6
pre-norm layers, width 384, 6 heads. Each query → 9 means, residual on the current pose like the teacher's: a chunk
of 5 actions (0.5 s), executed, then the
model plans again. A learned log std (GRPO only) makes it a gaussian policy.

**Distillation** (`ml/distill.py`). Round 0: the teacher drives (its mean action + 0.1 noise), 800k rows. Rounds 1-5
(DAgger): the VLA drives, 200k rows each. Every row is labeled with the teacher's mean action in that state; in the
race robot 1 is always the teacher. After each round the VLA trains on all rows so far (L1 on the next 5 teacher
actions, masked past the episode end; AdamW 3e-4, warmup + cosine, bf16) and is evaluated on every stage. One
collection env persists across rounds, so long episodes are covered past their start. Dataset (`data/distill/`,
preallocated memmaps, one row per active robot per control step): `rgb.npy` (N, 6, 64, 64) uint8, `words.npy`
(N, 16) int16, `proprio.npy` (N, 9), `action.npy` (N, 9) the label, `next.npy` (N,) row of the same robot's next
step (−1 at the episode end; rows of different robots interleave), `meta.json` `{"rows"}`.

**3. GRPO** (`ml/train_grpo.py`). One stage per iteration (round robin), 512 envs in groups of 8 that start from the
same copied state. The VLA samples chunks around its means (std 0.1 at the start, learned); each robot's first
episode gets one reward: drive / pick / bring success (1 / 0), race 1 / 0.5 / 0 for win / tie / loss against the
teacher, speedrun blocks home / 6 + the time left if all 6 are home. Advantage = (reward − group mean) / group std
for every decision of the episode; groups with no spread are dropped (DAPO). Loss: clipped PPO objective per
decision + 0.01 × KL to the frozen distilled VLA (exact, between gaussians). AdamW 1e-6, 2 epochs, stopped early once
the k3 KL estimate between the sampling policy and the current one passes 0.05 (as in SB3). Action std starts at 0.01
(grasping breaks under more noise), which makes log-probs so sensitive that the update runs in fp32: bf16 rounding of
the means alone pushed 80 % of the ratios outside the clip range.

**4. Eval** (`ml/eval.py`). Every model (expert, teacher, each VLA) runs one episode in each of 256 envs per stage:
success rate per stage (race: won against the teacher). Writes `logs/eval.json` and prints a markdown table.

---

## Part 7: Recordings

`recording.py`. One self-contained file per recording: everything a viewer needs, so a recording made on the GPU box
can be copied anywhere and opened in the frontend (playback mode: pick it from the server's list, "Open…", or drop the
file anywhere on the page; normal camera controls).

```
logs/{name}.simrec   a gzip stream, little-endian:
  "SIMREC01"         magic + format version
  uint32             header length
  header             JSON, space-padded to a multiple of 4 bytes:
                       {"scene": the scene, "meshes": [{"name", "vertices": V, "triangles": T}] (blob order),
                        "seconds_per_frame", "frames": F, "info": {...} facts about the run (a game's score, ...)}
  meshes             per mesh: V × 3 float32 positions, then T × 3 uint32 indices
  frames             F × (1 + 7 N) float32: time, then px py pz qw qx qy qz per body in scene order
```

Bodies may carry `"visuals": [{"mesh", "position", "rotation", "color"}]` (imported robots: the G1's real link meshes,
STL files under `robot/assets/`, named by that path). The physics ignores them; the recorder embeds every mesh the
scene uses and the viewer draws them instead of the collision shapes. A G1 recording is ~5 MB for 30 s.

```bash
# on the GPU box, from back/: record a checkpoint (walk: a scripted command sequence, other stages: --seconds of it)
python -m ml.record_humanoid --ckpt checkpoints/warehouse_v1.pt --stage bring --name warehouse_bring
# then on your machine: copy the file and drop it on the page
scp ubuntu@<gpu>:~/physicsim/logs/warehouse_bring.simrec .
```

**Pre-recorded games** (`record_games.py`, run on a GPU): the web demo plays these, so the web server needs no GPU.
Each is one 60 s race with a random layout, both robots driven by the teacher, knocked-over robots respawn. Its info
holds `{"stage": "race", "seconds": 60, "drivers": [orange, teal], "cameras": [camera ids, mosaic order],
"score": [[time, orange, teal], ...]}` (score at t = 0 and at every change), and `logs/game_{i}.mp4` next to it is
the 4 robot cameras at 10 Hz as one 2 × 2 mosaic (row-major in "cameras" order), H.264. The server lists as games the
recordings that have such a video.

Poses are saved at 30 Hz; a game is ~1.2 MB. On a CPU the live sim does not run in real time (rendering the robot
cameras takes 90–200 ms per frame), which is why the demo plays recordings.


---

## Part 8: Server and Editor

`api.py` (FastAPI, port 8080):

```
GET  /scenes                    {"scenes": [names]}
GET  /scenes/{name}             scene JSON
PUT  /scenes/{name}             save scene JSON
GET  /prefabs/lekiwi            LeKiwi bodies / joints / cameras at the origin
GET  /recordings                {"recordings": [names]} every logs/{name}.simrec
GET  /recordings/{name}         the recording file (Part 7)
GET  /games                     {"games": [names]} recordings that are pre-recorded games (have a camera video)
GET  /recordings/{name}/cameras a game's camera mosaic (logs/{name}.mp4)
WS   /ws                        live simulation
```

Websocket messages:

```
client -> server   {"type": "play", "scene": {...}}   pause / resume / step / stop
                   {"type": "drive", "targets": {"lekiwi/shoulder_lift": 0.3, ...}}
                   {"type": "autopilot", "mode": "off" | "expert" | "teacher" | "student" | "expert_both" | "teacher_both" | "student_both"}
server -> client   {"type": "frame", "time", "poses": [[px,py,pz,qw,qx,qy,qz], ...], "joints": {id: value},
                    "cameras": [{"id", "width", "height", "rgb": base64}],
                    "info": {"stage", "commands": [str per robot], "score": [orange, teal], "success": [bool per robot]}}
                    ~30 per second, "info" only in course scenes
                   {"type": "status", "state": "running" | "paused" | "stopped", "autopilot"}
                   {"type": "error", "message"}
```

`live.py` holds the session: one world (physics + cameras as CUDA graphs on GPU) stepped in real time. The autopilot
runs the expert, the teacher (`checkpoints/final/teacher.pt`) or the student VLA (`checkpoints/final/vla_best_v13.pt`)
at 10 Hz. A course scene (settings `"stage"`) runs that stage: you drive orange and the autopilot drives teal; the
"_both" modes drive both. In the speedrun teal is parked, so the autopilot drives orange. A robot knocked over (base
tipped past 60°) for 3 s respawns at its start pose (live sessions only; training ends the episode instead). The panel
shows the stage, your command, the score and whose goal currently holds.

**The frontend** (`front/`) opens in the **simple view**, a demo for people who don't know the project: it plays
the pre-recorded games back to back (teacher vs teacher races; the scoreboard captions come from the recording info's "drivers") with a scoreboard, a countdown, the four robot camera
feeds and the rules in one sentence. One button pauses it; it says the games are pre-recorded because the web server
has no GPU. No panels, no selection, no joint or camera markers (`src/demo.js`).
**Advanced** (toggle top right) is the full editor: CAD-style, scene tree left, properties right, toolbar on top.

- **Edit**: add boxes / spheres / cylinders, insert a LeKiwi, click to select, move/rotate gizmo with snapping, joints
  between two shift-selected bodies, every field editable (rotation shown as XYZ degrees, stored as a quaternion).
- **Simulate**: Play sends the edited scene to the server. Drive LeKiwi with the keyboard:
  W/S forward/back, A/D strafe, Q/E turn, Y/U pan, H/J shoulder, N/M elbow, I/O wrist, K/L wrist roll,
  Space gripper. Or switch the autopilot to expert / teacher / student. Both camera feeds and the joint values are shown live.
- **Playback**: pick a recording, play / pause / speed / scrub.
- **Camera**: left-drag orbits, ctrl+left-drag or right-drag pans (camera and pivot together), wheel zooms at a
  constant speed (pushing the pivot ahead when you pass it).

---

## Part 9: The Humanoid Warehouse Course

A Unitree G1 (29 joints, 1.32 m, 33 kg) learns the job warehouse humanoids are paid for (Agility's Digit moves totes
for GXO): pick up boxes and carry them to a platform. One model is trained end to end and is the model that would be
deployed: no scripted expert, no teacher to distill.

**Robot** (`robot/mjcf.py`, `robot/g1.py`). Imported from MuJoCo Menagerie's `g1_mjx.xml`: masses, inertias, joint
axes / limits / torque limits / PD gains, its simplified collision capsules, and its real link meshes for viewers. The
motors' armature (rotor inertia through the gearbox, e.g. 0.007 kg·m² at an ankle whose link is 1e-5) is added to
each child link about its hinge. Each foot touches the floor with a box. The head camera (RealSense D435 spot, 30°
down, 90° fov, 64 × 64) sees the boxes, pedestals and platform.

**Carry pose.** A search over the arm joints (within their limits, through the model's kinematics) for hands 0.37 m
ahead at 0.725 m height, ±0.142 m apart, palms facing, forearms behind the box. The robot is built in it: actions are
offsets from it, and a 0.26 m box between the hands is squeezed 2.3 cm from the start. Pedestals and the platform are
as tall as a held box's bottom, so boxes are picked and placed without bending.

**Control** (`robot/humanoid.py`). 50 Hz; one world step of 20 ms with 5 substeps (4 ms, like MuJoCo Playground).
Action = joint target offsets from the carry pose, × 0.25 rad, tracked by each motor's PD. Proprio = what the real
robot senses: pelvis gyro and gravity direction (IMU), joint angles and speeds, last action, gait clock, with noise.
Random pushes (≤ 0.5 m/s) every ~5 s.

**Stages** (`robot/warehouse.py`, one arena: four colored 1 kg boxes on pedestals, the platform at the arena's end):

| stage | command | task |
|---|---|---|
| walk | "warehouse walk forward +0.50 left +0.00 turn -0.25" | track the velocity (Playground G1 joystick rewards: tracking, posture, a reference foot-height gait at 1.25 Hz, air time) |
| pick | "warehouse pick up the red box" | lift the named box 10 cm off its pedestal, held between both hands |
| bring | "warehouse bring the red box home" | that box onto the platform; half the episodes start with it in the hands (reverse curriculum) |
| speedrun | "warehouse bring all boxes home" | every box onto the platform, fast |

Rewards: the locomotion terms (posture, gait) in every stage, potential shaping toward the target box (walk up to it,
hands on it, lift it / carry it to the platform), +10 for a success or a delivered box, −10 for dropping a box to the
floor, −25 for falling (more than a success: a fall restarts next to a fresh box, so lunge-grab-fall must not pay).
The command's first word is always the course's name, the game being played.

**Model** (`ml/humanoid_policy.py`). Single-stage asymmetric actor-critic with a state estimator inside the actor
(Ji et al. 2022, "Concurrent training of a control policy and a state estimator"; Gu et al. 2024, "Denoising World
Model Learning"):

```
policy (deployed)  tokens: [act] + 16 steps of proprio + 8 command words + 16 image patches (conv stem)
                   -> 4-layer transformer (width 192) -> z = the [act] token's output
                   z -> action head (gaussian mean, learned per-joint std)
                   z -> state decoder (training only): base velocity, height, feet, boxes, platform, pedestals
                        (regression on the simulator's true values, so z has to carry what the robot can't sense)
critic (training)  MLP on the privileged state (exact, noise-free)
```

Image tokens are always present: blind stages skip rendering and pass only their position embeddings, and the conv
stem's last layer starts at zero, so a model trained blind acts the same when the camera turns on.

**Training** (`ml/train_humanoid.py`). PPO (rsl_rl settings: adaptive learning rate from the KL, clipped value loss,
time-out bootstrapping; 4 optimizer steps per epoch with gradient accumulation), 4096 envs × 24 steps per iteration.
`--init` starts from an earlier run: matching tensors carry over, the vocabulary grows, normalizer statistics of
inputs that never varied restart, and the critic trains alone for a few iterations first. On the A100: physics ~35k
steps/s, the camera 10.6k frames/s at 64 px (rendered at 10 Hz), training ~6–7k steps/s with the camera.

**What we learned on the way** (an earlier sword course, since removed): unitree_rl_gym's walking rewards clipped to
~0 and stalled, Playground's walked in ~900 iterations; TF32 matmuls break the joint solve's symmetry; a stage that
pays more for its goal than a fall costs teaches lunging; zero-sum self-play settled into a standoff until the
shaping rewarded attacking; 80 small optimizer steps per iteration forced the adaptive learning rate down to 1e-5
(gradient accumulation fixed it).

---

## Running Everything

```bash
pip install -r back/requirements.txt
python main.py                       # backend (:8080) + editor (http://localhost:5173)

# learning (run in tmux, from back/)
python -m ml.train_teacher
python -m ml.distill
python -m ml.train_grpo
python -m ml.eval --vla checkpoints/vla_sft.pt checkpoints/vla_grpo.pt
```

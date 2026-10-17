# LeKiwi: course + ML plan

Goal: a **VLA** (vision-language-action model) you can talk to and that drives a LeKiwi robot directly from its
cameras: "drive forward half a meter", "grab the block on your left", "win the race". One transformer that sees,
talks and has **action heads** wired to the robot's joints and wheels.

Status: everything below is built and trained end to end on an A100 (results at the bottom). Code details:
`documentation.md` Part 6.

## The robot (what any policy sees and does)

| | what | shape |
|---|---|---|
| cameras | front camera on the base + wrist camera on the gripper, RGB | 2 × (H, W, 3) |
| proprio | 6 arm joint angles (normalized to [-1, 1]) + 3 wheel speeds | 9 |
| action | 3 wheel speed targets + 6 arm joint targets, in [-1, 1] | 9 |

Actions are raw wheel speeds, not "forward / sideways / turn", so driving a 3-omni-wheel base is a real skill the
model must learn (the three wheels sit at 120°, every motion is a mix of all three). Control runs at 10 Hz.

## Part 1: the course

Built: `back/robot/course.py` (details in `documentation.md`), one editor scene per stage (`course_1_drive` ... `course_5_speedrun`).
One arena (2.4 × 1.6 m, walls, two home zones, two robots, six pink blocks, a goal pad per robot). Every training env
picks a **stage**; the stage decides how the episode starts, what the command is and what earns reward. Same world
in every stage, so all stages batch together in one simulation.

| stage | command (what the robot is told) | episode start | success |
|---|---|---|---|
| 1. drive | go to a goal pose: "move 0.5 m forward and 0.3 m right, then turn 90° right" | anywhere in the arena | base within 5 cm / 10° of the goal |
| 2. pick | "pick up the block nearest you" | blocks scattered | a block lifted 5 cm, held in the gripper |
| 3. bring | "bring a block home" | blocks scattered | a block resting in its home zone |
| 4. race | "win the race" | race start, both robots play | more blocks home than the opponent after 60 s |
| 5. speedrun | "bring all blocks home, fast" | race start, robot 1 parked outside | all 6 blocks home; reward pays per second saved |

In stages 1–3 both robots in the env are learners doing their own command (they share the arena, so they also
learn not to crash into each other). Stage 4 is self-play: both robots are the same policy.

**Stage mixing.** Not "finish stage 1, then switch". Each env samples its stage at reset; a stage's sampling weight
is max(1 − its success rate, 0.1) (the floor keeps mastered skills from being forgotten), and stage k+1 only unlocks
when stage k succeeds 80% of the time (drive → pick → bring → race + speedrun; race "success" is winning
self-play, ~50% at best, so it gates nothing).

**Reward.** Per stage: progress toward the goal (potential shaping) plus
a bonus on success. Race: +10 per block of score difference change (zero-sum). Speedrun: race reward minus a
small time penalty per step.

## Part 2: the models

Four steps, each starting from the previous one's model. The VLA's training mirrors an LLM's: pretraining-like
supervised copying (SFT), then RL on verifiable outcomes (RLVR).

```
1. TEACHER   PPO on simulator state                 -> an MLP that does the course
2. DISTILL   teacher labels rendered frames + DAgger  -> VLA v1 copies it from pixels + text   (= SFT)
3. GRPO      VLA v1 fine-tuned on episode outcomes  -> beats its own SFT version             (= RLVR)
4. EVAL      every model on every stage             -> one results table
```

Why not RL the VLA from scratch: RL from pixels needs 10–100× more samples than RL from state, rendering is the
slowest part of the simulator, and a transformer forward pass per step makes it far worse. Learning the skill from
state, then distilling it into the vision model, is the standard recipe (legged robots at ETH, "Learning by
Cheating", Visual Dexterity). RL comes back at the end, starting from a VLA that already works.

### 0. The scripted expert (`ml/expert.py`)

Reads true positions; it is only the source of demonstrations and labels. Built to be imitable:

- **Memoryless**: every step it reads what to do from the state alone (is a block in the jaws, arm pose, jaw
  opening, where the base stands); no phase, timer or last command. With hidden state, DAgger labels followed the
  expert's own script instead of the clone's state, and every round made the clone worse.
- **Pure-pursuit arm**: the arm aims a few cm past where its tool tip is now, toward the goal, so the tip moves in
  straight lines (joint-space moves swept blocks under the 10 cm-high base and shook held ones loose).
- **Home slots**: the base stops in front of the zone and drops each block on one of 6 slots (12 cm apart, back row
  first); driving into the zone kept shoving delivered blocks into the corners. Driving steers around other blocks.
- **Drop, don't place**: the arm extends smoothly toward the slot as the base nears its stand, and the jaws open once
  the block is over the slot (5 cm), dropping it from the carry height, while the base keeps driving. The old expert
  lowered the arm to the slot once the base stood at an exact spot: the teacher and the student learned to park just
  outside that gate and hold the block over their zone for 25–48 s of a 60 s race (forcing the jaws open there scored
  every time). Expert alone: speedrun 98 % → 100 % (168 s → 80 s for 64 episodes), race unchanged.
- **Camera-friendly**: after placing, with the next block behind it, it spins left until that block is in front.
  Facing the wall, a camera-only student can't see where the remaining blocks are; "turn toward the next block" was a
  guess it averaged into a stall at the zone (its speedrun stuck at ~2.5 blocks).
- **Tolerant**: its "arrived" checks accept 1–2 cm, the error of a learned imitation; 5 mm checks left clones waiting
  forever. Speedrun: 99 % of 384 runs in ~560 of 1200 steps (the old expert: 37 %).

### 1. Teacher: PPO on state (`ml/teacher.py`, `ml/train_teacher.py`)

- **Input**: privileged state, all positions in the robot's own frame (62 numbers, `CourseEnv.state()`):

  | part | numbers |
  |---|---|
  | proprio (arm angles + wheel speeds) | 9 |
  | base velocity (forward, sideways, turn rate) | 3 |
  | grip point relative to base | 3 |
  | 6 blocks, **nearest first**: position (3) + sin, cos of 4 × relative yaw (2; a cube looks the same every 90°) | 30 |
  | own home, other home (x, z) | 4 |
  | other robot: position (x, z) + heading as sin, cos | 4 |
  | command: stage one-hot (5) + drive goal (forward, right, turn; zeros in other stages) | 8 |
  | time left in the episode, 1 → 0 | 1 |

  Blocks are sorted because a fixed slot per block would make the MLP learn every skill once per slot. The network
  ignores the last entry (time left): its training data comes from an expert that ignores time and always finished
  by mid-episode, so a clone that saw a late clock had never seen one and fell apart (speedrun 6 % → 50 % on the
  same weights with the clock hidden).
- **Model**: a set encoder over the 6 blocks (each block a token with "already home" flags, attention + max
  pooling, query = the robot's own state) → MLP 3 × 512 → 9 gaussian means + 9 learned stds; a separate critic of the
  same shape → 1. Only the teacher reads blocks as a set; the VLA sees pixels.
- **Residual actions**: mean action = the action that holds every joint where it is now (read off proprio, which is
  normalized the same way) + the network's output. The expert's arm commands are a few cm ahead of the current pose;
  as absolute targets in [-1, 1] (±1 = a joint's whole range) the network's ~0.03 errors meant 1–2 cm at the gripper
  and the arm stalled. With the residual, pick went 54 % → 76 % on the same data.
- **Speedrun subcourses** (reverse curriculum): episodes may start with k blocks already home; training starts at
  k = 5 (one block left) and unlocks k − 1 when k succeeds half the time, down to the full run.
- **Training**: PPO, 4096 envs × 2 robots, one network for every robot and stage (stage one-hot says which),
  self-play in the race. Rollout 32 steps, GAE (γ 0.99, λ 0.95), clip 0.2, 5 epochs × 4 minibatches, learning rate
  adapted to keep the KL near 0.01. Full explanation: MINDOLOGY `writing/training/PPO.md`.
- **Bootstrapped, not from scratch** (`ml/bc_teacher.py`): PPO from scratch learned to drive but random exploration
  almost never finds a grasp. So the teacher is a behavior clone of the scripted expert (block stages) plus an
  earlier PPO teacher (drive stage), then DAgger: the clone drives, the expert labels every state it reaches, and a
  fresh clone trains on everything (80k steps). Fine-tuning the last clone briefly instead left it fitting even its
  old data twice as badly, and every round got worse. PPO (`ml/train_teacher.py`, DAPG: still cloning the demos at
  a decaying weight, tiny action std 0.01) can fine-tune the clone further.

### 2. VLA v1 + distillation (`ml/vla.py`, `ml/distill.py`)

A VLA that cannot talk, built and trained entirely by us, to prove the pipeline before spending money on a
pretrained backbone.

- **Model**: a plain pre-norm transformer encoder, 6 layers, width 384, ~11M params, no pretrained weights.
- **Input tokens** (one sequence):

  | tokens | count | how |
  |---|---|---|
  | front camera patches | 64 | 96 × 96 image cut into 12 × 12 patches, each linearly projected (ViT style) |
  | wrist camera patches | 64 | same, own projection |
  | command words | ≤ 16 | word vocabulary of the command templates ("move", "forward", "0.5", "m", "°", ...) |
  | proprio | 1 | the 9 joint readings, linearly projected |
  | action queries | 5 | learned tokens, one per future action |

  plus a learned position embedding per slot.
- **Action head**: each action query → linear → 9 numbers: a chunk of 5 actions (0.5 s). Only the first is
  executed and the model plans again every step (receding horizon: 3.6 vs 3.1 speedrun blocks executing whole
  chunks); the rest of the chunk is a training target that makes the model plan ahead. GRPO executes whole chunks. Residual like the teacher's: the numbers are offsets from the action holding the current pose.
  Trained with L1 on the teacher's next 5 actions (as in ACT). For RL the means get a learned std: a gaussian
  policy, the same kind as the teacher's.
- **Commands as text**: drive goals are sampled on the grid the text can say exactly (0.1 m, 15°), so "move 0.5 m
  forward and 0.3 m right, then turn 90° right" is precisely the goal the teacher was given as numbers.
- **Round 0 (behavior cloning)**: the teacher drives, 800k rendered frames labeled with its actions.
- **Rounds 1+ (DAgger)**: the VLA drives, the teacher labels every state the VLA reached, 250k frames per round,
  the VLA retrains on everything. Plain cloning only shows the student states the teacher reached, so its first
  mistake puts it somewhere it has never seen.
- **Pass bar**: the VLA succeeds at each stage nearly as often as the teacher does.

### 3. GRPO: RL fine-tune of the VLA (`ml/train_grpo.py`)

The LLM RLVR recipe (DeepSeek-R1), with a robot. GRPO is PPO without the critic: judge each attempt against its
siblings instead of a learned value.

| | here | LLM equivalent |
|---|---|---|
| starting point | the distilled VLA | the SFT model |
| a "prompt" | one start state, copied into 8 envs (same layout, same command) | one prompt |
| G answers | 8 rollouts, sampling chunks around the means | 8 sampled completions |
| reward | success (drive / pick / bring), win / tie / loss vs the teacher (race), blocks + time left (speedrun) | verifier |
| advantage | (reward − group mean) / group std, for every decision of that episode | same, every token |
| loss | clipped PPO objective + β · KL to the frozen distilled VLA | same |

Why GRPO here but PPO for the teacher: the teacher learns from scratch, where success is rare (all-zero groups give
no signal) and per-step credit from a critic is what makes long episodes learnable. The VLA starts out already
decent, so outcome rewards give mixed groups, like an SFT model on math problems.

### 4. Eval (`ml/eval.py`)

| model | drive | pick | bring | race | speedrun |
|---|---|---|---|---|---|
| scripted expert | | | | | |
| teacher (state) | | | | | |
| VLA after distillation | | | | | |
| VLA after GRPO | | | | | |

Success rate per stage over 256 episodes (race: win rate against the teacher). Each row isolates one idea: what
RL from state gets, how much of it survives the move to pixels + language, and what RL on outcomes adds.

### Student v2: the VLA you can talk to (later)

Same inputs, action head, data and steps 2–4, but the transformer is a pretrained vision-language chat model, so the
same weights answer "what do you see?" and drive. Fine-tuned with LoRA (small trainable adapters, base weights
frozen) and co-trained on ordinary image-text data (as in RT-2) so its language ability survives. Backbone TBD; it
needs larger camera images (~224 px) and runs once per action chunk.

## Results (2026-10-13, A100, 128 episodes per stage, deterministic policies, `logs/eval_v11.json`)

Success rate per stage (race: won against the teacher; speedrun: all 6 blocks home), and blocks home per speedrun.
The first two VLA rows executed whole chunks (before the receding-horizon change).

| model | drive | pick | bring | race | speedrun | speedrun blocks |
|---|---|---|---|---|---|---|
| scripted expert (memoryless, slots) | 0 % | 100 % | 100 % | 76 % | 98 % | 5.94 |
| teacher (state, set encoder, residual, DAgger clone of the expert) | 95 % | 99 % | 100 % | 27 % (vs itself) | 87 % | 5.64 |
| VLA v1 96 px, distilled (cloning + 8 DAgger rounds) | 11 % | 97 % | 95 % | 2 % | 4 % | 3.12 |
| VLA v1 96 px + GRPO (1.5 h) | 8 % | 96 % | 94 % | 4 % | 6 % | 3.02 |
| VLA v1 96 px, + 2 DAgger rounds × 120k steps, replanning every step (`logs/eval_v12.json`) | 9 % | 98 % | 100 % | 6 % | 25 % | 4.09 |
| VLA v1 96 px, + 2 more DAgger rounds (`logs/eval_v13.json`) | 11 % | 100 % | 100 % | 4 % | 44 % | 4.59 |

What moved the numbers: rewriting the scripted expert to be memoryless (no phases or timers, so DAgger labels are
right in any state) with fixed home slots, a stand in front of each slot and pure-pursuit arm motion (the old one
bulldozed delivered blocks and swept them under the base: 37 % speedrun); a residual, time-blind teacher retrained
from scratch on all data every DAgger round (fine-tuning degraded it every round); 96 px cameras for the student (at
64 px a block 30 cm away was ~5 px). The student is the bottleneck: it handles single blocks (pick / bring ~95 %) but
stalls somewhere in a 6-block run, and GRPO on outcome rewards does not move it. Drive from text stays weak (~10 %):
the 5 cm / 10° goal has to be read from numbers in the command.

## Open decisions

- v2: pretrained backbone, and whether the action head moves to flow matching (π0) or action tokens (RT-2, OpenVLA)
  (both handle moments with two valid answers, like going left or right around the opponent; a gaussian averages
  them).
- Hyperparameters that only a GPU run can settle: GRPO learning rate (1e-6, watch the clip fraction) and β.

# SO-101 + chess: simulation, RL, real fine-tuning

Plan for building a simulated SO-101 + chess world, training a policy in it with reinforcement
learning (RL), fine-tuning on a small amount of real data, and using the result for the final
real-arm rollout. Phase 1 is **MuJoCo** (fast iteration, RL). Phase 2 moves the scene to **Isaac Sim**
for realistic rendering (and large-scale RL if needed).

> Status: plan only, nothing implemented. Items marked **(verify)** come from memory of upstream
> projects (asset locations, versions, hardware specs). Check them before relying on them.

## 0. The pipeline

```
 measure real rig ─► sim world ─► validate sim vs real ─► RL in sim ─► real fine-tune ─► real rollout
   (§2)              (§3, §5)        (§6)                  (§7)         (§8)            (§9)
                          │                                  ▲
                          └─ phase 2: Isaac Sim, same task ──┘ (§10)
```

**Cameras (fixed decision):** two only.

| Name | What | Notes |
|---|---|---|
| `wrist` | camera on the gripper | moves with the arm; sees the piece during grasp |
| `side` | fixed side view of the board and arm | must see the whole board and the arm's reach |

Real data and the sim must use exactly these two names, 640×480 at 30 fps (the rig's current
setting). Existing 3-camera datasets are not directly usable: record new real data with the third
camera left out (`scripts/record_so101.sh --exclude-camera extra` or `scene`, whichever is not the side
view), or drop that key at load time.

**Core strategy.** Pixel-based RL from scratch is slow and brittle, so train in two stages inside the sim:

1. **Teacher:** RL with *privileged state* (piece pose, target square, joint state), no images. Fast, stable.
2. **Student:** distill the teacher into an image policy (`wrist` + `side` + joint state → action)
   by rolling out and imitating with DAgger-style relabeling. Optionally add RL fine-tuning on the student.

Then fine-tune the student on real demonstrations, and roll it out. (Why not RL directly on the real
arm: it's slow, risky for the hardware and the pieces, and needs resets. Real data is used where the sim is weakest.)

## 1. Goals and success criteria

- **Task v1:** pick one piece from square A and place it upright on square B, on an otherwise empty board.
- **Task v2:** same with neighbors on the board (full start position), then piece types with different heights.
- **Metrics:** sim success rate (≥ 90 % pawn, ≥ 70 % other types before moving on), and the **real-arm** success rate
  after fine-tuning over a fixed trial set (the metric that matters).
- **Language prompt:** the task text (`"Move the pawn from c3 to d5"`) is generated from the move. A single
  RL policy can be goal-conditioned (target squares as input) first, and the text prompt added when distilling.
- **Non-goals now:** chess rules/engine play, captures, castling (see [§12](#12-later-chess-logic)).

## 2. Things you need to produce a good simulated world

This is the checklist. Each item is a place where sim-to-real usually breaks. Items marked ★ are the ones
RL most often exploits or fails on.

### 2.1 Real-world measurements (do these first)

| Need | Why |
|---|---|
| Board: square pitch, border width, thickness, surface material | defines every target position |
| Each piece type (6): height, base diameter, neck diameter, mass, material | grasp geometry and ★ stability |
| Arm base pose relative to board a1 corner (x, y, yaw) | reachability, camera views |
| `side` camera pose + intrinsics (FOV, resolution); `wrist` camera mount pose + FOV | image match |
| Gripper: max opening, jaw pad shape and material, closing speed | whether a piece fits and holds |
| Servo behavior: step/sine response of each joint (from real `action` vs `observation.state` logs) | actuator model |
| Control timing: real command-to-motion latency, camera latency | ★ delay in sim, or the policy fails on real |
| Joint calibration (our lerobot calibration JSON), joint limits | units/sign/offset mapping |
| Lighting and room photos (table color, light positions, reflections) | rendering and randomization ranges |

### 2.2 Robot model

- Accurate SO-101 kinematics (URDF/MJCF from `TheRobotStudio/SO-ARM100` **(verify)**) and our calibration mapping.
- Inertias/masses plausible, link collision simplified, self-collision filtered.
- ★ **Position actuators fit to the STS3215 servos** (kp, kd, armature, friction loss, force limit), not ideal
  torque motors or perfect position setting. Gripper force limit especially.
- Joint-limit and velocity-limit behavior like the real arm (it saturates and lags, and the policy must learn that).
- Measured (not commanded) joint state returned as the observation.

### 2.3 Chess set and board

- Visual meshes per piece type, textured board; **separate simplified collision shapes** (cylinders/capsules/convex hulls).
- ★ Real mass and friction for pieces, board and fingertips. Weighted bases lower the center of gravity: copy it.
- Stable resting contact: a full board stands still for 10 s untouched.
- One config file for all dimensions, layouts and square→pose mapping, used by the sim, the reward, and the expert.

### 2.4 Physics fidelity

- Contact settings that hold a ~20 g piece without slipping, crushing, jitter or tunneling (timestep, cone, solver iterations, friction with torsion).
- ★ A *grasp that matches reality*: if a grasp works in sim only because of unrealistically sticky contacts,
  RL will find and exploit it, and the real arm will fail. Calibrate by comparing real vs sim on a fixed
  set of grasps (success/slip/topple per piece type).
- Fixed control rate (30 Hz) with substeps; deterministic given a seed.

### 2.5 Sensors

- Two cameras (`wrist`, `side`) with real pose/FOV/resolution, rendered as uint8 RGB.
- Realistic image properties: exposure, white balance, noise, blur, MJPG-like compression.
- Joint state with realistic noise/quantization (servo encoder resolution).
- Camera and control **latency** modeled as a fixed + jittered delay.

### 2.6 Domain randomization (★ for transfer)

Randomize per episode, with ranges taken from real measurements (not arbitrary):

- **Visual:** light direction/intensity/color, table and board color tint, background, camera pose (±1 to 2 cm, ±2°),
  FOV, brightness/contrast/noise/blur.
- **Physics:** piece mass (±20 %), friction (±30 %), actuator gains (±15 %), gripper force, joint friction/damping, latency.
- **Task:** start/target squares, piece type, piece yaw, small piece offset within its square, arm start pose jitter.

### 2.7 Task machinery

- Episode reset with seeded layouts; piece spawn without interpenetration.
- **Success detector:** piece center within ¼ square of target center, upright, at rest, others undisturbed.
- Termination rules: success, timeout, piece fell off the board, arm/board collision, joint-limit violation.
- A **scripted IK expert** that solves the task (also validates the world is solvable, seeds RL, and produces demos).

### 2.8 Infrastructure

- Backend-agnostic `ChessEnv` (Gymnasium-style) so MuJoCo and Isaac share the same task code.
- Vectorized/parallel envs for RL throughput.
- Tests: determinism, static stability, unit mapping, replay-vs-real checks ([§6](#6-validate-the-sim-against-real-before-any-rl)).
- Logging: videos of rollouts, success rate per piece type/square, per randomization setting.
- Experiment tracking (Weights & Biases or TensorBoard) and pinned dependency versions.

### 2.9 Compute

- **MuJoCo phase:** this machine is enough for building the world and state-based RL on CPU (many cores help).
  The GTX 1070 Ti (8 GB) is weak for image-based training. MJX/JAX GPU parallel envs would work but will be slow; budget a better GPU
  or cloud for the student stage.
- **Isaac Sim phase:** needs an **RTX** GPU (RT cores, ≥ 8 GB, 16 GB+ comfortable) **(verify requirements)**. The 1070 Ti
  can't run it. Use an RTX workstation or a cloud instance with an RTX/L4/L40S/A10-class GPU (A100/H100 lack RT cores).

## 3. Architecture and repo layout

```
        RL (teacher, state)  ──►  distillation (student, images)  ──►  real fine-tune ──► rollout/
                    ▲                         ▲
                    └──────── ChessEnv (backend-agnostic) ────────┘
                         reset · step · render · success · randomize
                         units adapter: sim rad ⇄ lerobot deg/%
                 MuJoCoBackend (phase 1)        IsaacBackend (phase 2)
```

```
sim/
  assets/so101/  assets/chess/      # meshes, MJCF, later USD
  config/world.yaml                 # all measured dimensions, poses, ranges (single source of truth)
  calibration.py                    # rad <-> deg/% from the real calibration
  env.py                            # ChessEnv (Gymnasium API)
  mujoco/                           # scene builder + backend
  isaac/                            # phase 2
  expert.py                         # IK pick-and-place
  rl/                               # teacher training, distillation
  record_sim.py                     # sim -> LeRobotDataset v3.0
  tests/
rollout/robots/sim.py               # SimRobot: evaluate served policies in sim via the Robot protocol
```

## 4. Phase 1 setup

```bash
python -m venv .venv-sim && source .venv-sim/bin/activate
pip install mujoco mink gymnasium numpy opencv-python python-chess
pip install stable-baselines3     # or rsl-rl / skrl / your RL library of choice
export MUJOCO_GL=egl              # headless rendering
python -m mujoco.viewer --mjcf sim/mujoco/scene.xml   # primary debugging tool
```

## 5. Build the world (MuJoCo)

Order matters. Do not skip ahead, since each step has a test.

1. **Load the SO-101** (MJCF if available, else URDF → MJCF). Check joint order
   `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`, axis directions, limits.
2. **Calibration adapter.** Per-joint `deg = sign * rad*180/π + offset` and gripper linear map to 0–100 %, built from our calibration JSON.
   Policies see degrees/% exactly like the real robot, so existing normalization works unchanged.
3. **Actuators.** Position actuators; fit kp/kd/armature/frictionloss/forcerange to real step responses by system identification.
   Timestep 0.002 s, control 30 Hz (~16 substeps).
4. **Board and pieces** generated by `build_scene.py` from `world.yaml`. Collision primitives, free joints, measured mass,
   contact defaults to start from: `cone="elliptic"`, `impratio≈10`, `condim=4`, friction ≈ `1 0.02 0.001`. High-friction fingertip pads.
5. **Cameras** `side` and `wrist`. Calibrate `side` by overlaying the sim board corners on a real frame; set `wrist` from the CAD mount.
6. **`ChessEnv`:** `reset(seed, task)`, `step(action_deg)`, observation `{state(6), images{wrist, side}}`, `success()`, randomization hooks, latency buffer.
7. **IK expert:** pre-grasp → grasp → close → lift above tallest piece → transit → pre-place → lower → open → retreat,
   using `mink` with a fixed downward-ish approach (5-DoF arm). Verify success rate on randomized seeds.

## 6. Validate the sim against real before any RL

If this fails, RL will happily optimize a world that doesn't exist. Gates:

1. **Kinematic overlay:** replay real `observation.state` through the inverse adapter, render `side`, compare to the real frame at rest, mid-reach, and fully reached.
2. **Tracking match:** replay real `action` open-loop; sim joint state matches real within a few degrees RMS per joint, with a matching lag.
3. **Static stability:** 32 pieces, arm at rest, 10 s: nothing moves > 1 mm.
4. **Grasp parity:** a fixed set of ~20 real grasps (per piece type) vs the same grasp in sim: same outcome (hold/slip/topple) in ≥ 85 %.
5. **Expert solvable:** IK expert ≥ 90 % pawns, ≥ 70 % other types, over 100 seeds.
6. **Determinism and speed:** identical frames for the same seed; at least 3 to 10× real time with two cameras rendered.
7. **Schema:** a recorded sim dataset loads with `LeRobotDataset` (keys `action`, `observation.state`,
   `observation.images.wrist`, `observation.images.side`, `task`, fps 30). Copy exact names/dtypes from a real dataset's `meta/info.json`.

## 7. RL in the sim

### 7.1 Formulation

- **Action:** delta joint-position targets for the 5 arm joints plus a gripper command, scaled and clipped per step (e.g. ≤ a few degrees at 30 Hz).
  Delta actions on top of the current state transfer better than absolute targets and keep motions smooth. Output is converted to the
  deg/% convention at the boundary so the policy's action matches the real dataset convention. If the student must predict absolute
  commands like the real data, convert at distillation.
- **Teacher observation (privileged):** joint state, gripper state, active piece pose (position + yaw, upright/tilt), target square center,
  phase indicators, previous action.
- **Student observation:** `wrist` + `side` images, joint state, previous action, task goal.
- **Algorithm:** PPO or SAC for the teacher (PPO scales best with many parallel envs). Use asymmetric actor-critic if you RL the student directly.

### 7.2 Reward (staged, dense, then simplified)

```
r = w1 * reach        (-dist gripper → piece grasp point, shaped)
  + w2 * grasp        (both jaws in contact, piece lifted > h)
  + w3 * carry        (-dist piece → target square, while lifted)
  + w4 * place        (+ piece within ε of target, upright, gripper open and clear)
  + w5 * success      (large bonus once, at rest)
  - penalties         (action magnitude and jerk, joint limits, hitting other pieces, dropping, board collision, time)
```

Guards against reward hacking: require *upright and at rest* for success; penalize pieces moving that aren't the target;
penalize pushing the piece rather than grasping it (e.g. piece sliding without both jaws in contact); cap lifted-height reward.

### 7.3 Curriculum

1. Reach and hover above a piece. 2. Grasp and lift. 3. Place at a nearby square. 4. Any square pair. 5. Add neighbors and the full start
position. 6. Add piece types (pawn → rook/bishop → knight → queen/king). Move on at ≥ 80 % success. Initial states from expert
waypoints (reset-to-expert-states) speed up early learning.

### 7.4 Making it transfer (applied during RL, not afterwards)

- Domain randomization from [§2.6](#26-domain-randomization--for-transfer) on **from the start of RL**, ramped up with the curriculum.
- Latency, observation noise and action noise in the env.
- Smoothness penalties and action rate limits so motions are within what the servos track.
- Early-stop and checkpoint on the *grasp parity and sim-vs-real checks*, not only on sim reward.

### 7.5 Distilling into the vision student

1. Roll out the teacher with randomized visuals, store `(wrist, side, state, teacher_action)` at 30 Hz.
2. Train the student (ResNet-ish encoder per camera + MLP/transformer head, or a pretrained VLA backbone) with behavior cloning.
3. **DAgger loop:** run the *student*, let the teacher relabel actions in the states the student visits, retrain. Repeat until the student matches the
   teacher's success within a few points.
4. Optional RL fine-tune of the student with image observations (expensive, so only if the gap remains).
5. Record the final sim dataset as LeRobotDataset v3.0 with the `_sim` suffix so it cannot be mistaken for real data.

## 8. Fine-tuning with real data

Goal: correct what the sim got wrong (visuals, contact, latency) using a small amount of real data.

- **Collect** teleoperated demos with the existing pipeline ([data_collection.md](data_collection.md)), excluding the unused camera so the data has exactly `wrist` + `side`.
  Start with ~30 to 50 episodes over varied squares and piece types, with deliberate piece/target jitter. Include a few recoveries (re-grasp after slip).
- **Fine-tune** the sim-trained student on real data. Options, in order of effort:
  1. BC fine-tune on real only, with the sim-trained weights as initialization (low learning rate, maybe freeze the lower encoder first).
  2. Co-train: mix real and sim batches (e.g. 20 to 50 % real), annealing toward real.
  3. Real DAgger: run the policy on the arm, a human corrects, add the corrections. Slow, but the most data-efficient fix for failure modes.
- **Normalization:** compute statistics from the combined (or real) data, since the policy was normalized per dataset in the FLUX analysis
  ([flux_joint_convention.md](flux_joint_convention.md)). Check that the sim state/action ranges fall inside the real ones after the adapter.
- **Ablate** to see what the sim bought: real-only vs sim-only vs sim → real fine-tune, same real-arm trial set for all.

## 9. Final rollout

- Serve the final policy with the existing rollout path ([inference.md](inference.md)): `python -m rollout run --robot so101 --policy <server URL> --task "<prompt>"`.
- Rollout needs the policy's two-camera input, so the rollout robot config must use only `wrist` and `side`. Check this before the first real run.
- Safety: start with joint-delta clamps, lowered speed, workspace limits and the existing stall guard. Run slow, with a hand on power, on an empty board first.
- Evaluate on a fixed trial set (squares × piece types), report success with trial counts. The same set is used for all ablations.
- `rollout/robots/sim.py` (a `Robot`-protocol adapter over `ChessEnv`) lets the same code evaluate any policy in sim first, so sim and real use one pipeline.

## 10. Phase 2: Isaac Sim

When MuJoCo RL + real fine-tune works but is limited by visual gap or training throughput.

- **Why:** photorealistic lighting/materials for `side` and `wrist`, easy visual randomization, and (via Isaac Lab) thousands of parallel envs on GPU.
- **Port:** URDF/MJCF → USD (SO-101, check for a community Isaac Lab config first **(verify)**); chess visuals as PBR materials; keep the **same simplified collision shapes**;
  emit both MJCF and USD from `build_scene.py` so the backends can't drift.
- **Re-tune, don't assume:** PhysX contact and joint drives differ from MuJoCo. Redo actuator sysid, finger friction, gripper force; TGS solver,
  more position iterations, small dt, CCD for pieces.
- **Cameras:** same names, poses, intrinsics, 640×480. Use the RTX real-time renderer for data generation; add exposure, noise, compression.
- **Wiring:** `IsaacBackend` behind the same `ChessEnv`; Isaac may need its own Python env, so run it as a separate process with a small socket API.
- **Gates:** repeat [§6](#6-validate-the-sim-against-real-before-any-rl); grasp success within ~10 points of MuJoCo; side-by-side real vs sim frames reviewed.
- **Use:** re-render/regenerate the student's training data with Isaac visuals (cheap, the teacher is state-based), or run RL natively in Isaac Lab for scale;
  then repeat [§8](#8-fine-tuning-with-real-data) and compare real-arm results against the MuJoCo-trained policy.

## 11. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Real measurements ([§2.1](#21-real-world-measurements-do-these-first)), SO-101 assets located | table of measurements in `world.yaml` |
| M1 | SO-101 in MuJoCo, calibration adapter | kinematic overlay ([§6](#6-validate-the-sim-against-real-before-any-rl) gate 1) |
| M2 | Actuators fit to real | gate 2 |
| M3 | Board + pieces, stable | gate 3 |
| M4 | Two cameras matched, `ChessEnv`, randomization hooks | seed determinism + overlay |
| M5 | IK expert + grasp parity | gates 4 and 5 |
| M6 | Teacher RL (state) on task v1, then v2 | ≥ 90 % sim success under randomization |
| M7 | Vision student by distillation/DAgger | within a few points of teacher |
| M8 | Real data collection (wrist + side) and fine-tune | real-arm success beats real-only baseline |
| M9 | Final rollout and write-up | fixed trial set reported |
| M10 | Isaac Sim port and comparison (needs RTX GPU) | [§10](#10-phase-2-isaac-sim) gates pass |

M0 to M6 run on this machine (state-based RL on CPU). M7 benefits from a stronger GPU. M10 requires RTX.

## 12. Later: chess logic

Once single moves are reliable: `python-chess` holds game state; a UCI move becomes `(from, to, capture?, special?)`; captures need a
"move the captured piece off the board" sub-task; castling, en passant and promotion are multi-step. The language prompt is generated from the move.

## 13. Common failure modes

| Symptom | Likely cause |
|---|---|
| Sim arm doesn't overlay the real arm | calibration sign/offset wrong |
| RL finds a weird grasp (pinching, flicking, pushing the piece) | contact too sticky or reward hackable; tighten success and grasp-parity checks |
| Policy great in sim, jerky or overshooting on real | actuator model too ideal, no latency, action not rate-limited |
| Pieces slip or are crushed | fingertip friction, gripper force limit |
| Student works on sim frames, fails on real frames | visual gap: widen randomization, calibrate `side` pose better, then Isaac Sim |
| Fine-tuning forgets sim skills | learning rate too high, no co-training with sim batches |
| Isaac Sim won't start | GPU has no RT cores (e.g. the 1070 Ti) or driver too old |

## 14. References

- SO-101 hardware, URDF and sim files: `TheRobotStudio/SO-ARM100` **(verify layout)**.
- MuJoCo: <https://mujoco.readthedocs.io>; `mink` (IK): <https://github.com/kevinzakka/mink>; MuJoCo Playground / MJX for GPU RL **(verify)**.
- LeRobot (dataset format, SO-101): <https://github.com/huggingface/lerobot>.
- Isaac Sim / Isaac Lab: <https://docs.isaacsim.omniverse.nvidia.com>, <https://isaac-sim.github.io/IsaacLab>.
- Local: [data_collection.md](data_collection.md), [flux_joint_convention.md](flux_joint_convention.md), [inference.md](inference.md).

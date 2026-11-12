# Simulation architecture (rough)

Status: **only the environment and a smoke test exist** (TODO step A1). Everything below marked
*(planned)* is the intended design; names may change as we build. Plan and rationale:
[../../docs/simulation.md](../../docs/simulation.md). Step list: [../../docs/simulation_todo.md](../../docs/simulation_todo.md).

## Goal in one line

A MuJoCo world that matches the real table, board, SO-101 and two cameras (`wrist`, `side`) closely
enough that a policy trained with RL in sim can be fine-tuned on a little real data and run on the real arm.

## Data flow

```
 real_setup/  (your photos, measurements, calibration, camera + piece data)
      │  read by humans/me, distilled into numbers
      ▼
 config/world.yaml            ← single source of truth: dimensions, poses, masses, ranges  (planned)
      │
      ▼
 mujoco_env/build_scene.py    → generated scene.xml (table, board, pieces, cameras, lights)  (planned)
      │          + assets/so101/so101.xml (robot, from the SO-ARM100 URDF)                  (planned)
      ▼
 mujoco_env/env.py  ChessEnv  reset · step · render · success · randomize                    (planned)
      │   calibration.py: sim radians ⇄ lerobot degrees/%  (uses real_setup/calibration)     (planned)
      ├──► expert.py          IK pick-and-place (also validates the world is solvable)
      ├──► record_sim.py      sim episodes → LeRobotDataset v3.0 (`_sim`)
      └──► rl/                teacher (state) → student (wrist+side images) → real fine-tune
```

## Layers and rules

| Layer | Lives in | Responsibility | Rule |
|---|---|---|---|
| Real measurements | `real_setup/` | raw evidence: photos, numbers, logs | never edited by code; humans add to it |
| World config | `config/world.yaml` | every number the sim uses, each with a source note | no magic numbers in code; change it here |
| Assets | `assets/` | meshes and robot/piece MJCF | visual mesh and collision shapes are separate |
| Scene builder | `mujoco_env/` | `world.yaml` + assets → MJCF | scene files are generated, not hand-edited |
| Env | `mujoco_env/env.py` | Gymnasium-style API, units adapter | outputs degrees/% like the real dataset |
| Tasks/learning | `expert.py`, `rl/` | experts, RL, recording | talk to the env only through the API |

## Conventions (decide once, keep everywhere)

- **Units:** SI inside MuJoCo (m, kg, rad). Policy-facing values are lerobot degrees for the arm and 0–100 % for the gripper, as in the real data.
- **Joint order:** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- **World frame:** the layout frame of `real_setup/layout.drawio`: origin at the table corner on the robot's edge, +x along that edge (420 mm), +y away from the robot (600 mm), +z up from the table top. The robot base is at y = 0 and faces +y.
- **Board frame (decided 2026-11-12):** a1 is the playing-area corner closest to the origin. Rank numbers 1..8 increase along +x, file letters a..h along +y. `square_center(file, rank)` is defined once, from `setup_measured.yaml`.
- **Cameras:** exactly two, named `wrist` and `side`, 640×480, 30 fps.
- **Control:** 30 Hz policy rate, 500 Hz physics (about 16 substeps).
- **Determinism:** every reset takes a seed; same seed, same frames.

## Backends

MuJoCo first. Isaac Sim later reuses the same `world.yaml`, the same simplified collision shapes and the same `ChessEnv` API, so the task
code, expert and RL code don't change. Isaac needs an RTX GPU (this machine's GTX 1070 Ti can't run it).

## Tests *(planned)*

`tests/` grows with each step: smoke (exists), unit mapping, static stability of the board and pieces, determinism, replay-vs-real error,
expert success rate. A TODO step is done only when its test or comparison exists.

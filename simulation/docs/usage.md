# Usage guide

What you can run **today**, and what the workflow will look like as the steps land. Commands assume the repo root and the venv from
[setup.md](setup.md). Shorthand: `PY=simulation/.venv-sim/bin/python`.

## Available now (A1)

```bash
PY=simulation/.venv-sim/bin/python
$PY simulation/mujoco_env/smoke_test.py                  # load, step physics, render a frame
$PY simulation/mujoco_env/smoke_test.py --out frame.png  # also save the frame
$PY -m mujoco.viewer                                     # empty interactive viewer
```

## Day-to-day workflow

1. **Add real data** to `simulation/real_setup/` (see [setup.md](setup.md) §4).
2. **Tell me what changed**, and I distil it into `config/world.yaml` (every value keeps a note about its source photo/measurement).
3. **Rebuild the scene** and look at it *(planned)*:
   ```bash
   $PY simulation/mujoco_env/build_scene.py            # world.yaml -> scene.xml
   $PY simulation/mujoco_env/view_scene.py             # interactive viewer
   $PY simulation/mujoco_env/render_compare.py --real <photo> --out compare.png   # sim vs real overlay
   ```
4. **Run checks** *(planned)*: `pytest simulation/tests` (stability, unit mapping, determinism) and the replay-vs-real comparison plots.
5. **Commit** small steps, one TODO step per commit where possible.

## Planned commands by TODO step

| Step | Command | Purpose |
|---|---|---|
| A2 | `view_robot.py` | drag SO-101 joints in the viewer, print joint values |
| A4 | `build_scene.py`, `view_scene.py` | table + board + robot scene from `world.yaml` |
| A6 | `calibration.py` + tests | sim radians ⇄ lerobot degrees/%; overlay against reference poses |
| A7 | `replay_real.py <episode>` | replay recorded real actions in sim; per-joint error plots |
| B4 | `calibrate_cameras.py` | intrinsics/extrinsics from checkerboard images; sim camera blocks |
| B5 | `expert.py --piece pawn --from c3 --to d5 --seeds 100` | scripted pick-and-place, success rate, videos |
| B5 | `record_sim.py` | write successful episodes as a LeRobotDataset v3.0 (`_sim`) |
| B6 | `export_trajectory.py` | joint trajectory in real-arm convention for the real follower |

These names are proposals and will be adjusted when the scripts exist; this file will be updated at the same time.

## Conventions when using the sim

- Always pass a **seed**; log it with results.
- Never mix `_sim` and real datasets by accident: sim datasets carry the `_sim` suffix.
- Real-arm runs of sim trajectories: start slow, one piece, empty board, hand on the power switch.
- Don't hand-edit generated `scene.xml`; change `world.yaml` or the builder.

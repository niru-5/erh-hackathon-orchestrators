# Setup guide

Tested on this machine: Ubuntu (Linux 6.8), Python 3.12, `uv` available, GTX 1070 Ti (fine for MuJoCo; no Isaac Sim).

## 1. Create the environment

From the repo root:

```bash
uv venv simulation/.venv-sim --python 3.12
uv pip install --python simulation/.venv-sim/bin/python -r simulation/requirements.txt
```

`requirements.txt` has `mujoco>=4.0` (4.0.1 installed), `mink` (IK), `gymnasium`, `numpy`, `opencv-python`, `pyyaml`, `pytest`.
The venv is git-ignored. Don't install the sim packages into the lerobot environment; the two can pin different versions.

Without `uv`: `python3.12 -m venv simulation/.venv-sim && simulation/.venv-sim/bin/pip install -r simulation/requirements.txt`.

## 2. Verify

```bash
simulation/.venv-sim/bin/python simulation/mujoco_env/smoke_test.py --out /tmp/frame.png
# expected: ok: mujoco 4.0.1, GL=..., box z=0.0199, frame (480, 640, 3)
```

Pytest wrapper. This machine has ROS on `PYTHONPATH`, whose pytest plugins break collection, so isolate it:

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 simulation/.venv-sim/bin/python -m pytest -q simulation/tests
```

## 3. Rendering modes

| Situation | Setting |
|---|---|
| Desktop session (`DISPLAY` set) | default; the interactive viewer works |
| Headless / over SSH / for data generation | `export MUJOCO_GL=egl` (the smoke test sets it automatically if `DISPLAY` is unset) |
| EGL fails | try `MUJOCO_GL=osmesa` (slow, software) and check the NVIDIA driver |

Interactive viewer check: `simulation/.venv-sim/bin/python -m mujoco.viewer`. In MuJoCo 4 this may now be the "studio" viewer.

## 4. Adding your real-world data

Put files under `simulation/real_setup/`, by topic:

| Folder | Content |
|---|---|
| `photos/` | table, board, robot (top-down, front, sides, close-ups with a ruler) |
| `lighting/` | same view under each lighting condition, with a note of the lights |
| `measurements/` | numbers (structured values in `setup_measured.yaml`; free notes in `measurements.yaml`) |
| `calibration/` | follower/leader lerobot calibration JSON, reference poses, motion logs |
| `cameras/` | checkerboard images, intrinsics/extrinsics for `wrist` and `side` |
| `pieces/` | piece photos and measurements |

Tip: for photos, include a ruler and keep the phone as parallel to the table as possible for the top-down shots. Raw phone photos are
3 to 4 MB each; if the folder gets large, say so and we'll move them to Git LFS or out of git.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `pytest` crashes importing `launch`/`lark` | use the `env -u PYTHONPATH ...` command above (ROS plugins) |
| `GLFW`/`DISPLAY` error | `export MUJOCO_GL=egl` |
| `import mujoco` picks the wrong module | don't name a local package `mujoco` (ours is `mujoco_env`) |
| URDF import errors (A2) | usually mesh paths or `meshdir`; we'll handle them in the importer script |

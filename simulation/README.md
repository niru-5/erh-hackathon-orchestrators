# simulation/

MuJoCo (later Isaac Sim) world of the SO-101 and chess. Plan: [docs/simulation.md](../docs/simulation.md),
checklist: [docs/simulation_todo.md](../docs/simulation_todo.md).

Guides: [architecture](docs/architecture.md), [setup](docs/setup.md), [usage](docs/usage.md).

```
simulation/
  requirements.txt     # pinned-ish deps for the sim venv
  mujoco_env/          # MuJoCo scene builder, env, viewers (smoke_test.py for now)
  tools/               # capture_calib.py: Gradio UI to grab checkerboard images (B3)
  assets/              # robot + chess meshes and MJCF (A2, B2)
  config/              # world.yaml: measured dimensions, poses, ranges
  docs/                # architecture, setup guide, usage guide
  tests/
  real_setup/          # YOUR real-world data (photos, measurements, calibration, cameras, pieces)
```

## Setup

```bash
uv venv simulation/.venv-sim --python 3.12
uv pip install --python simulation/.venv-sim/bin/python -r simulation/requirements.txt
```

## Check it works

```bash
simulation/.venv-sim/bin/python simulation/mujoco_env/smoke_test.py            # headless falls back to EGL
simulation/.venv-sim/bin/python simulation/mujoco_env/smoke_test.py --out frame.png
# pytest: this machine has ROS plugins on PYTHONPATH, so isolate them
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 simulation/.venv-sim/bin/python -m pytest -q simulation/tests
```

`mujoco_env/` is deliberately not named `mujoco`, to avoid shadowing the package.

## real_setup/ (where your data goes)

| Folder | What | TODO step |
|---|---|---|
| `photos/` | table, board and robot photos (top-down, front, sides, close-ups, with a ruler in frame) | A3 |
| `lighting/` | the same view under each lighting condition, with a note of which lights were on | A3 |
| `measurements/` | numbers: table height, robot base position, board size, board-to-base offsets, sketch | A3 |
| `calibration/` | follower/leader lerobot calibration JSONs, reference poses, motion logs | A5 |
| `cameras/` | checkerboard images, intrinsics/extrinsics for `wrist` and `side` | B3 |
| `pieces/` | piece photos and a measurements table (mass, height, base) | B1 |

Any format is fine (a `measurements/setup.md` or `.yaml` is easiest to read). Large raw photos and
videos can be committed as-is unless they get big; tell me if you'd rather keep them out of git.

## Capture camera-calibration images (B3)

```bash
simulation/.venv-sim/bin/python simulation/tools/capture_calib.py   # open http://127.0.0.1:7860
```

Pick the camera (listed by product name, e.g. `HD Pro Webcam C920`, `USB2.0_CAM1`), set the interval
(default 2 s) and number of images (default 20), press **Start**. Images are saved to
`real_setup/cameras/<camera-name>/<timestamp>/img_NNN.png`, shown in the gallery, and earlier
sessions can be reopened from the "Saved sessions" dropdown. `--out DIR` changes the root folder.

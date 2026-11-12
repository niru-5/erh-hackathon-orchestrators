# simulation/

MuJoCo (later Isaac Sim) world of the SO-101 and chess. Plan: [docs/simulation.md](../docs/simulation.md),
checklist: [docs/simulation_todo.md](../docs/simulation_todo.md).

Guides: [architecture](docs/architecture.md), [setup](docs/setup.md), [usage](docs/usage.md).

```
simulation/
  requirements.txt     # pinned-ish deps for the sim venv
  mujoco_env/          # MuJoCo scene builder, env, viewers (smoke_test.py for now)
  tools/               # camera calibration: capture UI, AprilGrid detection, OpenCV/Kalibr-style/mrcal (B3/B4)
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

## Camera calibration (B3/B4)

```bash
simulation/.venv-sim/bin/python simulation/tools/capture_calib.py   # open http://127.0.0.1:7860
```

Board: set it in the UI ("Calibration board" panel, saved to `real_setup/cameras/board.yaml`). Three types:
**AprilGrid** (Kalibr's tag grid; default 12x9 tags, 36h11, ~16 mm tags, ~5 mm gaps), plain **checkerboard**
(enter the number of *squares* and the square size, e.g. 10 x 10 squares of 8 mm = 9 x 9 inner corners;
the whole board must be visible), and **ChArUco**. The panel shows what your numbers imply (inner corners, physical
size). The UI has two tabs:

- **Capture**: pick the camera (stable id, below), give it a role (`wrist`, `side`), set interval (default 2 s)
  and count (default 20). The live preview draws the detected tags with a **GOOD/BAD banner** (too few tags/corners, blurry,
  too small), each shot gets a verdict in the status line and the gallery, and "keep going until N GOOD" retries
  bad shots (they go to `_rejected/`). Click a gallery image to **delete** it, or move all BAD ones of a session
  to `_rejected/`. Images: `real_setup/cameras/<role>/<timestamp>/img_NNN.png`.
- **Calibrate**: weeds out bad images (quality gate, then reprojection-outlier rejection), runs OpenCV, a
  Kalibr-style solver and mrcal on the *same* images, and shows a comparison table plus plots (corner coverage,
  per-image error, parameters, distortion, board poses, residuals, undistorted image). *Save calibration* writes
  `real_setup/cameras/<role>/calibration.yaml`.

The same pipeline from a shell: `python simulation/tools/calibrate.py <role> [--move-bad] [--save opencv:5]`.
Findings and how the methods compare: [docs/calibration_comparison.md](docs/calibration_comparison.md).

mrcal is installed without its GUI-only dependency: `uv pip install --python simulation/.venv-sim/bin/python --no-deps mrcal`.

**Identifying identical cameras.** A camera's id is `vid:pid:serial` when it has a unique USB serial,
otherwise `vid:pid@usb-port` (e.g. `046d:08e5@1-5`). Neither camera here has a usable serial, so they are
identified by USB port: **keep each camera in the same port**, or re-save its role after moving it.
Roles live in `real_setup/cameras/cameras.yaml`; calibration is read from
`real_setup/cameras/<role>/calibration.yaml`:

```python
from camera_registry import find_device, load_calibration   # simulation/tools
dev = find_device("wrist")        # /dev/videoN currently holding that role (None if unplugged)
calib = load_calibration("wrist") # dict from calibration.yaml, or None
```

**Camera focus lock.** Intrinsics depend on focus, so a calibrated camera must be used at the focus it was calibrated at.
The C920 autofocuses by default (and forgets a manual setting when unplugged or rebooted), so the controls are stored per
camera in `real_setup/cameras/cameras.yaml` (`controls:`) and **re-applied every time the capture app opens the camera**.
In the Capture tab: *Manual focus* + the slider (0 = far, 250 = near), *Apply*, *Find sharpest* (sweeps and picks the
sharpest value for what the camera sees; hold the board at your working distance first), then *Save for this camera*.
Saving a calibration also records the focus in `calibration.yaml` (`camera_controls`). From code or a shell:

```python
from camera_registry import open_capture
cap = open_capture("side", (1280, 720))   # cv2.VideoCapture with the saved focus lock applied first. Use this for any video capture
```
```bash
python simulation/tools/camera_registry.py list                                   # roles, ids, saved controls
python simulation/tools/camera_registry.py save side focus_automatic_continuous=0 focus_absolute=20
python simulation/tools/camera_registry.py apply side                             # re-apply after a replug
```

Why the lock is re-applied instead of set once: the C920 USB-autosuspends after 2 s idle (`/sys/bus/usb/devices/<port>/power`),
forgets its focus (reads back 0) and then answers `Input/output error` to every `focus_absolute` write until autofocus is flipped
on and off again. `apply_controls` does that flip automatically. To stop the camera suspending at all (needs root):
`echo on | sudo tee /sys/bus/usb/devices/1-5/power/control` (until replug), or a udev rule
`ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="08e5", ATTR{power/control}="on"`
in `/etc/udev/rules.d/99-c920-nosuspend.rules`. While a program streams from the camera it does not suspend.

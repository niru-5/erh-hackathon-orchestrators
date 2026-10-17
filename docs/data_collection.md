# SO-101 data collection (LeRobotDataset v3.0)

We record teleoperated demos with two SO-101 arms (the leader drives the follower) using the
**official LeRobot recording pipeline** (`lerobot-record`, lerobot `0.6.0`). It writes datasets
in the **LeRobotDataset v3.0** format (`meta/info.json` → `"codebase_version": "v3.0"`).

`scripts/record_so101.sh` wraps `lerobot-record` with this rig's ports, calibration and cameras,
so a recording only needs a **task prompt** and an **episode count**.

## Quick start

```bash
# from the repo root
scripts/record_so101.sh --task "Pick up the red cube and place it in the bowl" --episodes 1
```

- `--task` is the language instruction saved with every frame (`dataset.single_task`), so a
  trained policy can be conditioned on it. One prompt per run; use separate runs (or `--resume`)
  for different tasks.
- `--episodes` is how many episodes to record (`dataset.num_episodes`).

The dataset is saved locally to `~/.cache/huggingface/lerobot/<repo-id>`. `lerobot-record` appends
a `_<YYYYmmdd_HHMMSS>` timestamp to the repo id of every **new** dataset, so the default
`<hf_user>/so101` becomes e.g. `niru-5/so101_20261017_130512`. The final id is in the log
(`repo_id=...`) and is the folder name. If you're not logged in to Hugging Face, `<hf_user>` is `local`.

## Workflow

```
 leader arm ──(you move it)──► lerobot-record ──► follower arm mirrors it
                                     │  every frame @30 fps:
                                     │   action            = leader joint positions (6)
                                     │   observation.state = follower joint positions (6)
                                     │   observation.images.{scene,wrist,extra}
                                     │   task              = --task prompt
                                     ▼
          ~/.cache/huggingface/lerobot/<hf_user>/so101_<timestamp>/   (LeRobotDataset v3.0)
                                     │
                                     ▼  (optional) --push, or scripts/push_dataset.py later
              https://huggingface.co/datasets/<hf_user>/so101_<timestamp>
```

1. **Check** (optional): `scripts/check_cameras.py` / `scripts/check_calibration.py`, see
   [Pre-recording checks](#pre-recording-checks).
2. **Record**: `scripts/record_so101.sh --task "<prompt>" --episodes N`. Each episode is
   *record* (up to `--episode-time`) → *reset* (up to `--reset-time`, nothing saved) → next.
   Videos are encoded on the fly (`streaming_encoding`). Each finished episode is saved, and the
   dataset files are finalized on exit. Esc stops cleanly; Ctrl+C also finalizes but drops the
   episode in progress.
3. **Inspect**: the load snippet in [Dataset layout](#dataset-layout-v30) or `lerobot-dataset-viz`.
4. **Add more** (optional): rerun with `--repo-id <full timestamped id> --resume`.
5. **Push**: see [Pushing to the Hugging Face Hub](#pushing-to-the-hugging-face-hub).

### All options

| Flag | Default | Meaning |
|---|---|---|
| `--task "<prompt>"` | required | Task / language instruction |
| `--episodes N` | required | Number of episodes |
| `--repo-id USER/NAME` | `<hf_user>/so101` | Dataset id (lerobot appends `_<timestamp>` to new datasets) |
| `--root DIR` | `~/.cache/huggingface/lerobot/<repo-id>` | Local output dir |
| `--episode-time SEC` | 60 | Max length of an episode |
| `--reset-time SEC` | 30 | Time to reset the scene between episodes |
| `--fps N` | 30 | Recording / camera fps |
| `--push` | off | Upload to the Hugging Face Hub when done |
| `--resume` | off | Append to an existing dataset (needs its full timestamped `--repo-id`) |
| `--no-display` | display on | Disable the Rerun live viewer |
| `--dry-run` | | Print the full `lerobot-record` command and exit |
| `-- <args>` | | Extra args passed straight to `lerobot-record` |

Examples:

```bash
# 20 episodes, pushed to the Hub when done -> niru-5/so101_stack_blocks_<timestamp>
scripts/record_so101.sh --task "Stack the blue block on the red block" --episodes 20 \
  --repo-id niru-5/so101_stack_blocks --push

# add 10 more episodes to that dataset later: use the full id, timestamp included
scripts/record_so101.sh --task "Stack the blue block on the red block" --episodes 10 \
  --repo-id niru-5/so101_stack_blocks_20261017_130512 --resume
```

## Pushing to the Hugging Face Hub

Datasets stay local unless you push them. Two options (both need `.venv/bin/hf auth login`):

- **While recording**: add `--push`. The dataset is uploaded when recording finishes.
- **After recording** (recommended: inspect first, then push):

  ```bash
  # push under the same id
  .venv/bin/python scripts/push_dataset.py niru-5/so101_20261017_130512
  # push under a different name (also how to push a dataset saved under local/)
  .venv/bin/python scripts/push_dataset.py local/so101_20261017_124726 --hub-id niru-5/so101_cube_test
  # add --private for a private Hub repo
  ```

`push_dataset.py` uses lerobot's own `LeRobotDataset.push_to_hub()`. It uploads `meta/`, `data/` and
`videos/`, writes a dataset card and tags the repo with the codebase version (`v3.0`). Anyone can then
load it with `LeRobotDataset("<hf_user>/<name>")` or train on it with `lerobot-train --dataset.repo_id=<hf_user>/<name>`.
Pushing again after `--resume` updates the same Hub repo.

## Keyboard controls during recording

| Key | Action |
|---|---|
| → (right arrow) | End the current episode (or reset period) early and save it |
| ← (left arrow) | Discard and re-record the current episode |
| Esc | Stop recording; episodes recorded so far are saved |

The laptop runs **Wayland**, so lerobot reads these keys from the **terminal** that runs the script,
not globally. Keep that terminal focused when you press them (not the Rerun window).

No reset period follows the last episode.

## Rig configuration

The defaults below are set in the script. Stable `by-id` paths are used so re-plugging USB cables
doesn't swap devices. Override any of them with an env var, e.g. `CAM_EXTRA=/dev/video0 scripts/record_so101.sh ...`.

| Env var | Default | Device |
|---|---|---|
| `FOLLOWER_PORT` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115116-if00` (currently `ttyACM0`) | Follower arm |
| `LEADER_PORT` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116682-if00` (currently `ttyACM1`) | Leader arm |
| `FOLLOWER_ID` / `LEADER_ID` | `so101` | Calibration files in `~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/so101.json` |
| `CAM_SCENE` | Logitech C920, `/dev/video6` (leLab: `right_side`) | → `observation.images.scene`: fixed top view showing marks 0E–8E (training) |
| `CAM_WRIST` | Sonix USB2.0_CAM1, `/dev/video2` (leLab: `ff`) | → `observation.images.wrist`: gripper cam (training) |
| `CAM_EXTRA` | Logitech C920 `2ED4DF6F`, `/dev/video4` (leLab: `left_side`) | → `observation.images.extra`: recorded, not used for training |
| `CAM_WIDTH` / `CAM_HEIGHT` | 640 / 480 | Camera resolution |

The laptop's built-in webcam (`/dev/video0`) is not used. Ports, camera devices and calibration match
the setup done in leLab (`~/.cache/huggingface/lerobot/robots/so101.json`). The cameras are renamed to
`scene` / `wrist` / `extra`, so datasets recorded before 2026-10-17 13:00 (with `left_side` / `right_side` / `ff`)
have different image feature names.

The existing `so101` calibration files are reused as-is; nothing is recalibrated. If `lerobot-record`
asks *"Press ENTER to use provided calibration file..."*, the values stored in the motors differ from the
file. Pressing ENTER writes the file to the motors; typing `c` recalibrates from scratch.

Useful helpers: `lerobot-find-port` (identify an arm's port by unplugging it),
`lerobot-find-cameras opencv` (list cameras and save test frames), and
`lerobot-calibrate` (recalibrate an arm; keep the `so101` id).

## Dataset layout (v3.0)

```
~/.cache/huggingface/lerobot/<repo-id>/
├── meta/
│   ├── info.json          # codebase_version v3.0, fps, features (action, observation.state, images)
│   ├── tasks.parquet      # task prompt(s) -> task_index
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet        # per-frame action / state / timestamps / task_index
└── videos/observation.images.<cam>/chunk-000/file-000.mp4
```

### Dataset spec

| Feature | Shape / type | Content |
|---|---|---|
| fps | 30 Hz | `timestamp` = `frame_index / 30` |
| `observation.state` | 6 × float32 | Follower joint positions: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| `action` | 6 × float32 | **Absolute** leader-arm positions, same order |
| `observation.images.scene` | 480×640×3 video | Fixed top view showing marks 0E–8E (training) |
| `observation.images.wrist` | 480×640×3 video | Wrist/gripper camera (training) |
| `observation.images.extra` | 480×640×3 video | Extra side camera (recorded, not used for training) |
| `task` | string | The `--task` prompt |

Actions and states are stored **raw**: no deltas, and no dataset normalization (`meta/stats.json` is
metadata used only at training time). `action` is the leader's reading (`teleop.get_action()` through
lerobot's default identity processor), not the command clipped for the follower (`max_relative_target`
is unset anyway).

Units: lerobot 0.6.0's SO-101 default `use_degrees=true` converts motor ticks with the calibration
files into **degrees** for the 5 body joints, and **0–100 %** of the calibrated range for the gripper.
These are absolute positions with the same scale for leader and follower, not ticks.

Quick check after a recording:

```bash
.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('<repo-id>')
print(ds.meta.info['codebase_version'], ds.num_episodes, ds.num_frames)
print(ds.meta.tasks)
"
.venv/bin/lerobot-dataset-viz --repo-id <repo-id> --episode-index 0
```

## Pre-recording checks

Both scripts are read-only. They don't move the arms or enable torque.

```bash
.venv/bin/python scripts/check_calibration.py         # calibration files vs. values stored in the motors
.venv/bin/python scripts/check_cameras.py camera_check # saves one JPEG per camera to camera_check/
scripts/record_so101.sh --task "test" --episodes 1 --dry-run   # print the exact lerobot-record command
```

## Setup & verification log (2026-10-17)

Steps done to set up recording on the laptop:

1. **Found the lerobot install.** `./lerobot` is the official repo (`huggingface/lerobot`, commit
   `30da8e6`, version `0.6.0`), installed editable in `./.venv` (Python 3.12) with all `lerobot-*` CLIs.
   The official recording pipeline is `lerobot-record`. It writes LeRobotDataset `v3.0`
   (`CODEBASE_VERSION = "v3.0"`). Hugging Face login: `niru-5`.
2. **Identified the arms.** Two Feetech/CH340 USB serial adapters. Used the port assignment saved by leLab
   (`~/.cache/huggingface/lerobot/robots/so101.json`): follower = `ttyACM0` (serial `5B8E115116`),
   leader = `ttyACM1` (serial `5B8E116682`). The script uses the `/dev/serial/by-id/...` paths so the
   two can't swap. The user is in the `dialout` group.
3. **Identified the cameras.** Mapped `/dev/v4l/by-id` to the 3 cameras configured in leLab:
   `left_side` = C920 `2ED4DF6F` (`video4`), `right_side` = C920 (`video6`), `ff` = Sonix USB2.0_CAM1
   (`video2`, wrist/gripper cam). The built-in laptop webcam (`video0`) is unused. Camera names match
   the earlier leLab datasets.
4. **Checked the calibration.** `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101.json`
   and `.../teleoperators/so_leader/so101.json` each have all 6 joints (`shoulder_pan`…`gripper`,
   motor ids 1–6). `scripts/check_calibration.py` confirmed that both files match the values stored in
   the motors (`motors match file = True`), so recording won't prompt for recalibration. They are used
   via `--robot.id=so101` / `--teleop.id=so101`.
5. **Checked the cameras.** `scripts/check_cameras.py` got a 640×480 frame from all three. `right_side`
   sees the chessboard and arm; `ff` sees the board between the gripper fingers. `left_side` was
   blocked at check time, so check its framing before recording.
6. **Checked the recording settings.** The `--dry-run` command was parsed with lerobot's own
   `RecordConfig` (draccus): `SOFollowerRobotConfig` + `SOLeaderTeleopConfig`, 3 cameras,
   `single_task` = the `--task` prompt, `num_episodes` = `--episodes`, `push_to_hub=false`.
7. **Checked the output format.** An earlier leLab recording (`niru-5/test_2_20261017_120222`) loads
   with `LeRobotDataset` as `v3.0` (2 episodes, 857 frames) with the layout shown above.
8. **Keyboard controls.** The laptop runs Wayland, so lerobot 0.6.0 uses its terminal key listener:
   arrow keys and Esc must be pressed in the recording terminal.
9. **Matched the dataset spec.** Renamed the cameras: `right_side` → `scene` (top view, marks 0E–8E),
   `ff` → `wrist`, `left_side` → `extra`. Checked a test recording
   (`local/so101_20261017_124726_20261017_124733`): `fps` 30 with timestamps spaced exactly 1/30 s,
   `action`/`observation.state` are 6 × float32 in the spec order, and `action` tracks the follower state
   (mean difference under 1° on most joints), confirming absolute leader positions, not deltas.

## Setup (already done on the recording laptop)

lerobot is cloned at `./lerobot` (from https://github.com/huggingface/lerobot, `v0.6.0`) and
installed editable into `./.venv`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -e "./lerobot[feetech,core_scripts]"
.venv/bin/hf auth login     # only needed for --push
```

Your user needs to be in the `dialout` group to access the arms' serial ports.

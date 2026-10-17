# SO-101 data collection (LeRobotDataset v3.0)

We record teleoperated demos with two SO-101 arms (the leader drives the follower) using the
**official LeRobot recording pipeline** (`lerobot-record`, lerobot `0.6.0`). It writes datasets
in the **LeRobotDataset v3.0** format (`meta/info.json` → `"codebase_version": "v3.0"`).

`scripts/record_so101.sh` wraps `lerobot-record` with this rig's ports, calibration and cameras,
so a recording only needs a **task prompt** and an **episode count**.

The recorded training dataset is `niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741`
(31 episodes, 13,229 frames). Its metadata is in [Recorded datasets](#recorded-datasets).

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
| `--resume` | off | Append to an existing dataset (needs its full timestamped `--repo-id`). lerobot's resume needs an explicit root, so the script passes the default `--root` unless you give one |
| `--exclude-camera NAMES` | none | Cameras to leave out, comma-separated or repeated: `scene`, `wrist`, `extra` (see [Camera options](#camera-options)) |
| `--display` | off | Show the Rerun live viewer (can stall recording, see [Recording speed](#recording-speed-30-hz)). `--no-display` is accepted and is the default |
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

# only the two training cameras (scene + wrist)
scripts/record_so101.sh --task "Move the piece forward" --episodes 10 --exclude-camera extra
```

## Camera options

Camera names you can use with `--exclude-camera`. Each becomes `observation.images.<name>` in the dataset.

| Name | Camera | Default device | Device override | Used for training |
|---|---|---|---|---|
| `scene` | Logitech C920, fixed top view showing marks 0E–8E | `/dev/video6` | `CAM_SCENE` | yes |
| `wrist` | Sonix USB2.0_CAM1 on the gripper | `/dev/video2` | `CAM_WRIST` | yes |
| `extra` | Logitech C920 `2ED4DF6F`, left side | `/dev/video4` | `CAM_EXTRA` | no (recorded only) |

```bash
--exclude-camera extra               # record scene + wrist
--exclude-camera extra,wrist         # record scene only (same as: --exclude-camera extra --exclude-camera wrist)
```

- An excluded camera isn't opened at all, so it may be unplugged. Unknown names and excluding all
  three cameras are rejected.
- **A dataset's camera set is fixed when it's created.** With `--resume`, use the same
  `--exclude-camera` as the first session, or lerobot rejects the mismatching features.
- Excluding `extra` also lightens the load (one stream fewer to encode and display).

Camera settings (apply to all recorded cameras):

| Setting | How | Default |
|---|---|---|
| Resolution | env `CAM_WIDTH` / `CAM_HEIGHT` | 640 × 480 |
| Frame rate | `--fps N` (also the dataset fps) | 30 |
| Device | env `CAM_SCENE` / `CAM_WRIST` / `CAM_EXTRA` (a `/dev/videoN` or `/dev/v4l/by-id/...` path) | see table |
| Pixel format | env `CAM_FOURCC` (backend is always V4L2) | `MJPG` (needed for the wrist cam to reach 30 fps) |

Example: `CAM_WIDTH=1280 CAM_HEIGHT=720 scripts/record_so101.sh ...`. Only use a mode every camera
supports at the chosen fps. Check with `lerobot-find-cameras opencv`, or
`v4l2-ctl -d /dev/video6 --list-formats-ext` (`sudo apt install v4l-utils`). Higher resolutions
cost encoding and USB bandwidth, so watch for the speed warning below. lerobot's OpenCV camera also
supports `rotation` (0/90/180/270) and `color_mode`. These aren't exposed by the
script; pass a full `--robot.cameras=...` after `--` if needed.

## Recording speed (30 Hz)

lerobot does **not** store wall-clock time: `timestamp = frame_index / fps`, so timestamps are always
exactly 1/30 s apart even if the loop ran slower. A slow loop means fewer real samples per second, so the
episode plays back too fast and actions are misaligned. **Don't train on such episodes.**

### The timing report (proof of 30 Hz)

`record_so101.sh` runs lerobot-record through `scripts/record_timed.py`. This changes nothing in the
recording; it times every iteration of lerobot's own record loop. At the end it prints:

```
===== Recording timing report: niru-5/so101_20261017_150000 (target 30 Hz) =====
 ep  frames  wall s  real Hz  late  p99 ms  worst ms  1st ms  verdict
  0     612    20.4    30.00     0    33.6      34.1    68.0  OK
RESULT: all episodes at target rate
```

- **real Hz**: frames per real second (frames 2..N). **late**: frames that took over 50 ms (1.5× the budget).
- **1st ms**: the first frame of each episode is where lerobot starts the video encoders. It takes
  about 40–70 ms (≈ 1 frame lost per episode). This is expected and excluded from the verdict.
- **OK** = real Hz ≥ 29.4 and no late frames after the first. Re-record episodes marked
  `NOT 30 Hz`. Discarded (←) attempts aren't counted.
- Saved to `~/.cache/huggingface/lerobot/timing/<repo_id>.json`.

lerobot's own `Record loop is running slower (X Hz)` warning is about **one** frame (X = 1 / that
frame's duration), not an average. One warning at an episode's start is the encoder start above; many
warnings mean a real problem.

### What the script does for 30 Hz (measured on this laptop, 2026-10-17)

| Setting | Why |
|---|---|
| Cameras opened with `backend: V4L2, fourcc: MJPG` | With OpenCV defaults the **wrist cam delivered only 26.7 fps** (scene/extra 30.2). With V4L2 + MJPG all three deliver 30.2 fps. (Camera reads never block the loop; a slow camera gives repeated frames instead.) |
| Rerun viewer **off** by default (`--display` to enable, with compressed images) | With the viewer on, the loop fell to 7.9 Hz and then 0.2 Hz (Rerun hit its 1.5 GiB memory limit and blocked the loop). leLab records at 30 Hz for the same reason: same lerobot loop, no Rerun. |
| `play_sounds=false` | No speech-synthesis process starting at each episode. |
| Streaming video encoding (`encoder_threads=2`) | Encodes during recording. No `Episode finished with N dropped frame(s)` was seen. |

Not a bottleneck: motor bus reads take about 1 ms per arm (worst 4 ms) out of the 33 ms budget.

If episodes still fail: `--exclude-camera extra`, then close leLab / browser tabs (RAM was at 12 of 15 GiB).
`CAM_FOURCC=YUYV` restores the old camera format.

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

## Recorded datasets

### Main dataset: `niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741`

The dataset recorded for training. All metadata below was read from the local copy
(`meta/info.json`, `meta/tasks.parquet`, `meta/episodes/`, `meta/stats.json`, `data/`) and from the
timing report, on 2026-10-17.

| Field | Value |
|---|---|
| Hub | [huggingface.co/datasets/niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741](https://huggingface.co/datasets/niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741), **public**, license `apache-2.0`, tags `LeRobot`, `task_categories:robotics`, git tag `v3.0`, commit `9b6b387` (pushed 15:28 CEST). The Hub copy matches the local one (31 episodes, 13,229 frames) |
| Local path | `~/.cache/huggingface/lerobot/niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741` |
| Format | LeRobotDataset `v3.0`, recorded with lerobot `0.6.0` (`scripts/record_so101.sh`) |
| Robot | `so_follower` (SO-101 follower, teleoperated with the SO-101 leader, calibration id `so101`) |
| Task (`task_index` 0, the only task) | `Move the figure from c3 at 6cm to D5 at 10 cm` (verbatim, including the mixed case) |
| Episodes / frames | **31** episodes, **13,229** frames, split `train` = `0:31` |
| fps | 30 (data and all videos) |
| Total duration | 441.0 s (7 min 21 s) |
| Episode length | mean 426.7 frames (14.2 s), median 406 (13.5 s), min 305 (10.2 s, ep 19), max 598 (19.9 s, ep 1), std 79.7 |
| Cameras | `scene`, `wrist`, `extra`: all three recorded, 640×480, RGB |
| Video encoding | AV1, `yuv420p`, CRF 30, GOP (`g`) 2, preset 12, `pyav` backend, no audio |
| Size | 444 MB: videos 443 MB (scene 160, wrist 154, extra 151), frame data 0.6 MB, meta 0.4 MB |
| Files | 5 data parquet files, 5 episode-meta files, 5 MP4s per camera (one per recording session), `chunks_size` 1000 |

**Recording sessions.** The dataset was created at 14:57:41 and extended 4 times with `--resume`.
Each session wrote its own data/video file:

| Session | Episodes | Frames (global `index`) | File | Saved at (CEST) |
|---|---|---|---|---|
| 1 | 0–2 | 0–1594 (1,595) | `file-000` | 14:59:40 |
| 2 | 3–4 | 1595–2682 (1,088) | `file-001` | 15:08:51 |
| 3 | 5 | 2683–3130 (448) | `file-002` | 15:11:50 |
| 4 | 6–20 | 3131–8744 (5,614) | `file-003` | 15:18:22 |
| 5 | 21–30 | 8745–13228 (4,484) | `file-004` | 15:27:27 |

**Per-episode lengths** (frames / seconds):

| ep | frames | s | ep | frames | s | ep | frames | s | ep | frames | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 464 | 15.5 | 8 | 387 | 12.9 | 16 | 370 | 12.3 | 24 | 454 | 15.1 |
| 1 | 598 | 19.9 | 9 | 368 | 12.3 | 17 | 464 | 15.5 | 25 | 516 | 17.2 |
| 2 | 533 | 17.8 | 10 | 334 | 11.1 | 18 | 338 | 11.3 | 26 | 406 | 13.5 |
| 3 | 557 | 18.6 | 11 | 378 | 12.6 | 19 | 305 | 10.2 | 27 | 521 | 17.4 |
| 4 | 531 | 17.7 | 12 | 308 | 10.3 | 20 | 352 | 11.7 | 28 | 371 | 12.4 |
| 5 | 448 | 14.9 | 13 | 398 | 13.3 | 21 | 534 | 17.8 | 29 | 393 | 13.1 |
| 6 | 482 | 16.1 | 14 | 326 | 10.9 | 22 | 451 | 15.0 | 30 | 388 | 12.9 |
| 7 | 454 | 15.1 | 15 | 350 | 11.7 | 23 | 450 | 15.0 | | | |

**Recording rate (timing report).** `~/.cache/huggingface/lerobot/timing/niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741.json`
covers only session 5 (episodes 21–30), because each `--resume` session overwrites the report for the same repo id.
For those 10 episodes: real rate 29.88–29.91 Hz, **0 late frames** after the first, p99 frame time
33.7–33.9 ms, worst 49.7 ms (ep 30), first frame 38–72 ms (encoder start), no discarded attempts. All 10 are **OK**.
Episodes 0–20 have no saved report. They were recorded with the same settings, but their real rate
can't be checked after the fact.

**Joint statistics** (from `meta/stats.json`; degrees for the 5 body joints, 0–100 % for the gripper):

| Joint | `observation.state` min / max | mean ± std | `action` min / max | mean ± std |
|---|---|---|---|---|
| `shoulder_pan` | −17.1 / 7.8 | −5.2 ± 4.5 | −17.7 / 8.2 | −5.0 ± 4.5 |
| `shoulder_lift` | −97.1 / 41.8 | −24.8 ± 48.0 | −103.6 / 41.3 | −26.5 ± 49.3 |
| `elbow_flex` | −30.4 / 96.7 | 38.9 ± 37.6 | −33.3 / 96.8 | 37.9 ± 38.5 |
| `wrist_flex` | 45.5 / 100.0 | 77.5 ± 10.7 | 44.4 / 100.8 | 77.5 ± 10.8 |
| `wrist_roll` | −106.5 / 67.3 | −74.1 ± 23.2 | −107.2 / 67.4 | −74.0 ± 23.2 |
| `gripper` | 0.8 / 37.1 | 8.5 ± 7.7 | 0.2 / 37.8 | 7.6 ± 8.5 |

- `wrist_roll` stays between −85 and −65 for 98 % of frames (q01 −85.3, q99 −65.5). The max of +67 is a
  rare outlier; check for it before training if normalization uses min/max.
- Mean |`action` − `observation.state`| per joint: pan 0.5°, lift 3.4°, elbow 2.9°, wrist flex 1.0°,
  wrist roll 0.5°, gripper 1.7 %. The follower tracks the leader closely; the larger lift/elbow gap
  is the follower lagging under load.

Image statistics (RGB, 0–1, over 253,996,800 sampled pixels per camera): mean `scene`
(0.553, 0.482, 0.414), `wrist` (0.504, 0.419, 0.425), `extra` (0.404, 0.383, 0.317); std about 0.26–0.29 per channel.

Load it:

```bash
.venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741')
print(ds.num_episodes, ds.num_frames)   # 31 13229
"
```

### Other recordings (2026-10-17)

Test and earlier runs, not meant for training. All are 30 fps, `so_follower`, LeRobotDataset `v3.0`, under
`~/.cache/huggingface/lerobot/` unless noted.

| Repo id | Task | Episodes / frames | Cameras | Where | Notes |
|---|---|---|---|---|---|
| `niru-5/test_2_20261017_120222` | erh_hacakthon | 2 / 857 | `right_side`, `ff`, `left_side` | Hub only | leLab recording, used to check the format |
| `niru-5/test_1_20261017_135757` | Move the horse piece forward | 2 / 928 | `right_side`, `ff`, `left_side` | local + Hub | Old camera names |
| `chess_data/` (repo root, gitignored) | Move the horse piece forward | 2 / 1690 | `right_side`, `ff`, `left_side` | local | leLab-era copy with old camera names |
| `niru-5/chess_piece_forward_20261017_140002` | Move the horse piece forward | 2 / 784 | `scene`, `wrist`, `extra` | local | |
| `niru-5/so101_20261017_141741` | Move the chess piece forward | 1 / 596 | `scene`, `wrist`, `extra` | local | |
| `niru-5/so101_20261017_142009`, `niru-5/so101_20261017_142311` | | 0 / 0 | | local | Empty (only `meta/info.json`) |
| `niru-5/so101_20261017_142326` | Move the chess piece forward | 1 / 431 | `scene`, `wrist`, `extra` | local | Not finalized: no `meta/episodes/`, leftover `tmp*` dirs |
| `niru-5/chess_dataset_20261017_142713` | Move the chess piece forward | 2 / 13,843 | `scene`, `wrist`, `extra` | local (441 MB) | Long episodes (191 s, 272 s); 29.9 Hz with 1 late frame each (worst 134 ms) |

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
   (`local/so101_20261017_124726_20261017_124733`): `fps` 30, and episode 0 has 1789 frames over its
   60 s (about 29.8 Hz real rate; the timestamps don't show this, see [Recording speed](#recording-speed-30-hz)).
   `action`/`observation.state` are 6 × float32 in the spec order, and `action` tracks the follower state
   (mean difference under 1° on most joints), confirming absolute leader positions, not deltas.
10. **Added `--exclude-camera`** after a session with the Rerun viewer on dropped to 7.9 Hz, then 0.2 Hz.
11. **Tuned for 30 Hz.** Compared leLab's recorder (`leLab/lelab/record.py`): it calls the same lerobot
    `record_loop` with `display_data=False`, pins `V4L2`, and turns sounds off. Benchmarked the hardware with the
    3 cameras running at once: default OpenCV gave scene 30.2 / wrist **26.7** / extra 30.2 fps, while V4L2 +
    MJPG gave 30.2 fps on all three. Motor bus reads take about 1 ms per arm. Read lerobot's loop: the slow warning
    is per frame, and the 14.2 / 25.3 Hz warnings of the 14:00 session were single first frames (encoder
    start). Applied V4L2 + MJPG, viewer off by default, sounds off, and added the timing report
    (`scripts/record_timed.py`, tested with a simulated record loop).

## Setup (already done on the recording laptop)

lerobot is cloned at `./lerobot` (from https://github.com/huggingface/lerobot, `v0.6.0`) and
installed editable into `./.venv`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -e "./lerobot[feetech,core_scripts]"
.venv/bin/hf auth login     # only needed for --push
```

Your user needs to be in the `dialout` group to access the arms' serial ports.

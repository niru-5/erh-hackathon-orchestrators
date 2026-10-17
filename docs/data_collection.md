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

The dataset is saved locally to `~/.cache/huggingface/lerobot/<repo-id>`. The default repo id is
`<hf_user>/so101_<YYYYmmdd_HHMMSS>`; the script prints it when it starts.

### All options

| Flag | Default | Meaning |
|---|---|---|
| `--task "<prompt>"` | required | Task / language instruction |
| `--episodes N` | required | Number of episodes |
| `--repo-id USER/NAME` | `<hf_user>/so101_<timestamp>` | Dataset id |
| `--root DIR` | `~/.cache/huggingface/lerobot/<repo-id>` | Local output dir |
| `--episode-time SEC` | 60 | Max length of an episode |
| `--reset-time SEC` | 30 | Time to reset the scene between episodes |
| `--fps N` | 30 | Recording / camera fps |
| `--push` | off | Upload to the Hugging Face Hub when done |
| `--resume` | off | Append to an existing dataset (needs `--repo-id`) |
| `--no-display` | display on | Disable the Rerun live viewer |
| `--dry-run` | | Print the full `lerobot-record` command and exit |
| `-- <args>` | | Extra args passed straight to `lerobot-record` |

Examples:

```bash
# 20 episodes, pushed to the Hub under a fixed name
scripts/record_so101.sh --task "Stack the blue block on the red block" --episodes 20 \
  --repo-id niru-5/so101_stack_blocks --push

# add 10 more episodes to that dataset later
scripts/record_so101.sh --task "Stack the blue block on the red block" --episodes 10 \
  --repo-id niru-5/so101_stack_blocks --resume
```

## Keyboard controls during recording

| Key | Action |
|---|---|
| → (right arrow) | End the current episode (or reset period) early and save it |
| ← (left arrow) | Discard and re-record the current episode |
| Esc | Stop recording; episodes recorded so far are saved |

The laptop runs **Wayland**, so lerobot reads these keys from the **terminal** that runs the script,
not globally. Keep that terminal focused when you press them (not the Rerun window).

Flow per episode: *record* (up to `--episode-time`) → *reset* (up to `--reset-time`, not recorded)
→ next episode. No reset period follows the last episode.

## Rig configuration

The defaults below are set in the script. Stable `by-id` paths are used so re-plugging USB cables
doesn't swap devices. Override any of them with an env var, e.g. `CAM_FF=/dev/video0 scripts/record_so101.sh ...`.

| Env var | Default | Device |
|---|---|---|
| `FOLLOWER_PORT` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115116-if00` (currently `ttyACM0`) | Follower arm |
| `LEADER_PORT` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116682-if00` (currently `ttyACM1`) | Leader arm |
| `FOLLOWER_ID` / `LEADER_ID` | `so101` | Calibration files in `~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/so101.json` |
| `CAM_LEFT` | Logitech C920 (serial `2ED4DF6F`), `/dev/video4` | → `observation.images.left_side` |
| `CAM_RIGHT` | Logitech C920, `/dev/video6` | → `observation.images.right_side` |
| `CAM_FF` | Sonix USB2.0_CAM1 (wrist/gripper cam), `/dev/video2` | → `observation.images.ff` |
| `CAM_WIDTH` / `CAM_HEIGHT` | 640 / 480 | Camera resolution |

The laptop's built-in webcam (`/dev/video0`) is not used. Ports, camera names/indices and calibration
match the setup done in leLab (`~/.cache/huggingface/lerobot/robots/so101.json`), so datasets
recorded here have the same features as the earlier leLab test datasets.

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

Features: `action` (6 leader joint positions), `observation.state` (6 follower joint positions:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`), and one video stream
per camera.

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

## Setup (already done on the recording laptop)

lerobot is cloned at `./lerobot` (from https://github.com/huggingface/lerobot, `v0.6.0`) and
installed editable into `./.venv`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -e "./lerobot[feetech,core_scripts]"
.venv/bin/hf auth login     # only needed for --push
```

Your user needs to be in the `dialout` group to access the arms' serial ports.

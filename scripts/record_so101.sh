#!/usr/bin/env bash
# Record SO-101 teleoperation episodes (leader -> follower) into a LeRobotDataset v3.0
# using the official `lerobot-record` pipeline.
#
# Usage:
#   scripts/record_so101.sh --task "Pick up the red cube and place it in the bowl" --episodes 1
#
# Run with --help for all options. Anything after `--` is passed verbatim to lerobot-record.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/record_so101.sh --task "<prompt>" --episodes <N> [options] [-- <extra lerobot-record args>]

Required:
  --task "<prompt>"       Language instruction stored with every frame (dataset.single_task)
  --episodes N            Number of episodes to record (dataset.num_episodes)

Options:
  --repo-id USER/NAME     Dataset id; lerobot appends _<YYYYmmdd_HHMMSS> to new datasets
                          (default: <hf_user>/so101 -> <hf_user>/so101_<YYYYmmdd_HHMMSS>)
  --root DIR              Local dataset dir (default: ~/.cache/huggingface/lerobot/<repo-id>)
  --episode-time SEC      Max seconds per episode (default: 60)
  --reset-time SEC        Seconds to reset the scene between episodes (default: 30)
  --fps N                 Recording fps (default: 30)
  --push                  Push the dataset to the Hugging Face Hub when done (default: off)
  --resume                Append episodes to an existing dataset (requires --repo-id)
  --exclude-camera NAMES  Comma-separated cameras to leave out: scene, wrist, extra
                          (e.g. --exclude-camera extra). Must match the dataset when resuming.
  --display               Show the Rerun live viewer (off by default: it can stall the record loop)
  --dry-run               Print the lerobot-record command without running it
  -h, --help              Show this help

Cameras (-> observation.images.<name>):
  scene                   Fixed top view, marks 0E-8E   (CAM_SCENE, default /dev/video6)  training
  wrist                   Gripper camera                (CAM_WRIST, default /dev/video2)  training
  extra                   Left side camera              (CAM_EXTRA, default /dev/video4)  recorded only

Hardware overrides (env vars):
  FOLLOWER_PORT, LEADER_PORT, FOLLOWER_ID, LEADER_ID,
  CAM_SCENE, CAM_WRIST, CAM_EXTRA (device paths), CAM_WIDTH, CAM_HEIGHT (default 640x480),
  CAM_FOURCC (default MJPG)
EOF
}

# --- Hardware defaults (stable by-id paths, so USB re-plugging doesn't swap devices) ---
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115116-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116682-if00}"
FOLLOWER_ID="${FOLLOWER_ID:-so101}"   # ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101.json
LEADER_ID="${LEADER_ID:-so101}"       # ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101.json
CAM_EXTRA="${CAM_EXTRA:-/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_2ED4DF6F-video-index0}"
CAM_SCENE="${CAM_SCENE:-/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0}"
CAM_WRIST="${CAM_WRIST:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
CAM_FOURCC="${CAM_FOURCC:-MJPG}"   # MJPG: the wrist cam only reaches ~27 fps with the default format

TASK=""
EPISODES=""
REPO_ID=""
ROOT=""
EPISODE_TIME=60
RESET_TIME=30
FPS=30
PUSH=false
RESUME=false
DISPLAY_DATA=false
DRY_RUN=false
EXCLUDE_CAMERAS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --episode-time) EPISODE_TIME="$2"; shift 2 ;;
    --reset-time) RESET_TIME="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --push) PUSH=true; shift ;;
    --resume) RESUME=true; shift ;;
    --exclude-camera) EXCLUDE_CAMERAS="${EXCLUDE_CAMERAS:+$EXCLUDE_CAMERAS,}$2"; shift 2 ;;
    --display) DISPLAY_DATA=true; shift ;;
    --no-display) DISPLAY_DATA=false; shift ;;  # default, kept for old commands
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "$TASK" ]] || { echo "error: --task is required" >&2; usage >&2; exit 1; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo "error: --episodes must be a positive integer" >&2; exit 1; }
if [[ "$RESUME" == true && -z "$REPO_ID" ]]; then
  echo "error: --resume needs --repo-id of the dataset to append to" >&2; exit 1
fi

# Locate the venv that has lerobot installed: repo .venv (also from inside a git worktree), else PATH.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd || echo "$REPO_ROOT")"
LEROBOT_RECORD=""
for venv in "$REPO_ROOT/.venv" "$MAIN_ROOT/.venv"; do
  if [[ -x "$venv/bin/lerobot-record" ]]; then LEROBOT_RECORD="$venv/bin/lerobot-record"; break; fi
done
LEROBOT_RECORD="${LEROBOT_RECORD:-$(command -v lerobot-record || true)}"
[[ -n "$LEROBOT_RECORD" ]] || { echo "error: lerobot-record not found (see docs/data_collection.md)" >&2; exit 1; }
HF_CLI="$(dirname "$LEROBOT_RECORD")/hf"
PYTHON="$(dirname "$LEROBOT_RECORD")/python"
RECORD_TIMED="$REPO_ROOT/scripts/record_timed.py"

# lerobot-record appends _<YYYYmmdd_HHMMSS> to the repo id of every new dataset (not on --resume).
if [[ -z "$REPO_ID" ]]; then
  HF_USER="$("$HF_CLI" auth whoami --format json 2>/dev/null \
    | "$(dirname "$LEROBOT_RECORD")/python" -c 'import json,sys; print(json.load(sys.stdin).get("user") or "")' 2>/dev/null || true)"
  [[ -n "$HF_USER" ]] || echo "warning: not logged in to Hugging Face, saving under local/ (run: .venv/bin/hf auth login)" >&2
  REPO_ID="${HF_USER:-local}/so101"
fi

# scene = fixed top view (marks 0E-8E), wrist = gripper cam: used for training. extra = recorded only.
declare -A CAM_DEVICES=([scene]="$CAM_SCENE" [wrist]="$CAM_WRIST" [extra]="$CAM_EXTRA")
CAM_NAMES=()
IFS=',' read -ra EXCLUDED <<< "$EXCLUDE_CAMERAS"
for name in "${EXCLUDED[@]}"; do
  [[ -n "${CAM_DEVICES[$name]+x}" ]] || { echo "error: unknown camera '$name' (choose from: scene, wrist, extra)" >&2; exit 1; }
done
for name in scene wrist extra; do
  [[ ",$EXCLUDE_CAMERAS," == *",$name,"* ]] || CAM_NAMES+=("$name")
done
[[ ${#CAM_NAMES[@]} -gt 0 ]] || { echo "error: all cameras excluded" >&2; exit 1; }

# Sanity-check hardware before starting (excluded cameras may be unplugged).
for dev in "$FOLLOWER_PORT" "$LEADER_PORT"; do
  [[ -e "$dev" ]] || { echo "error: device not found: $dev" >&2; exit 1; }
done
for name in "${CAM_NAMES[@]}"; do
  [[ -e "${CAM_DEVICES[$name]}" ]] || { echo "error: $name camera not found: ${CAM_DEVICES[$name]}" >&2; exit 1; }
done

# OpenCV wants the real /dev/videoN node.
cam() { echo "{type: opencv, index_or_path: $(readlink -f "$1"), width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS, backend: V4L2, fourcc: $CAM_FOURCC}"; }
CAMERAS=""
for name in "${CAM_NAMES[@]}"; do
  CAMERAS+="${CAMERAS:+, }$name: $(cam "${CAM_DEVICES[$name]}")"
done
CAMERAS="{$CAMERAS}"

ARGS=(
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_PORT"
  --robot.id="$FOLLOWER_ID"
  --robot.cameras="$CAMERAS"
  --teleop.type=so101_leader
  --teleop.port="$LEADER_PORT"
  --teleop.id="$LEADER_ID"
  --dataset.repo_id="$REPO_ID"
  --dataset.single_task="$TASK"
  --dataset.num_episodes="$EPISODES"
  --dataset.episode_time_s="$EPISODE_TIME"
  --dataset.reset_time_s="$RESET_TIME"
  --dataset.fps="$FPS"
  --dataset.push_to_hub="$PUSH"
  --dataset.streaming_encoding=true
  --dataset.encoder_threads=2
  --display_data="$DISPLAY_DATA"
  --display_compressed_images=true
  --play_sounds=false
  --resume="$RESUME"
)
[[ -n "$ROOT" ]] && ARGS+=(--dataset.root="$ROOT")

if [[ "$RESUME" == true ]]; then echo "Appending $EPISODES episode(s) -> $REPO_ID"; else echo "Recording $EPISODES episode(s) -> ${REPO_ID}_<timestamp> (lerobot adds the timestamp)"; fi
echo "Task: $TASK"
echo "Cameras: ${CAM_NAMES[*]}"
if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' "$PYTHON" "$RECORD_TIMED" "${ARGS[@]}" "${EXTRA_ARGS[@]}"; echo
  exit 0
fi
set -x
# record_timed.py runs lerobot-record unchanged and prints a real-time 30 Hz report at the end.
exec "$PYTHON" "$RECORD_TIMED" "${ARGS[@]}" "${EXTRA_ARGS[@]}"

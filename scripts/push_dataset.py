"""Push a locally recorded LeRobotDataset (v3.0) to the Hugging Face Hub.

Usage:
  .venv/bin/python scripts/push_dataset.py <local repo-id or path> [--hub-id USER/NAME] [--private]

Examples:
  .venv/bin/python scripts/push_dataset.py niru-5/so101_20261017_130000
  .venv/bin/python scripts/push_dataset.py local/so101_20261017_124726_20261017_124733 --hub-id niru-5/so101_cube_test
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dataset", help="Local repo id (under ~/.cache/huggingface/lerobot) or a dataset directory")
parser.add_argument("--hub-id", help="Hub repo id to push to (default: the local repo id)")
parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
args = parser.parse_args()

root = Path(args.dataset).expanduser()
if not (root / "meta" / "info.json").exists():
    root = HF_LEROBOT_HOME / args.dataset
if not (root / "meta" / "info.json").exists():
    raise SystemExit(f"No dataset found at {args.dataset} or {root}")

hub_id = args.hub_id or f"{root.parent.name}/{root.name}"
if hub_id.startswith("local/"):
    raise SystemExit("'local/...' is not a Hub namespace; pass --hub-id <hf_user>/<name>")

ds = LeRobotDataset(hub_id, root=root)
print(f"Pushing {root} ({ds.meta.info['codebase_version']}, {ds.num_episodes} episodes, {ds.num_frames} frames) -> {hub_id}")
ds.push_to_hub(private=args.private)
print(f"Done: https://huggingface.co/datasets/{hub_id}")

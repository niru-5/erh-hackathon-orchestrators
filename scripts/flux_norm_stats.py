"""Compute FLUX 3 Action normalization quantiles for a LeRobot SO-101 dataset.

FLUX 3 Action was pretrained on ~1000 community SO-100/SO-101 datasets, each normalized by its *own*
q01/q99 (state, and per-frame joint deltas a[t] - a[t-1] with the gripper absolute), so it is
calibration-agnostic inside. The released black-forest-labs/flux-3-action-so101 package ships one fixed set
of quantiles, which are exactly those of a single training dataset (VoicAndrei/so100_kitchen, an SO-100 in the
old lerobot calibration: shoulder_lift 85..184 deg). Our calibration puts shoulder_lift at -97..39 deg, so with
the package quantiles our state is far outside anything the model saw. Serving with quantiles computed by
this script normalizes our robot the way pretraining normalized every dataset (see docs/inference.md).

The output has the same layout as the package's dataset_statistics.json; pass it to the model server with
--norm-stats.

Usage:
  .venv/bin/python scripts/flux_norm_stats.py <repo id or dataset dir> [--exclude-episodes 0,1] [-o out.json]

Example:
  .venv/bin/python scripts/flux_norm_stats.py niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 \
    --exclude-episodes 0,1 -o rollout/models/norm_stats/chess_c3_d5.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

GRIPPER = 5  # absolute in FLUX's action representation; the five body joints are deltas

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dataset", help="Hub dataset repo id or a local LeRobot dataset directory")
parser.add_argument("--exclude-episodes", default="", help="comma-separated episode indices to leave out")
parser.add_argument("-o", "--output", type=Path, help="write the JSON here (default: print it)")
args = parser.parse_args()

root = Path(args.dataset).expanduser()
if not (root / "meta" / "info.json").exists():
    root = Path(snapshot_download(args.dataset, repo_type="dataset", allow_patterns=["meta/*", "data/*"]))
frames = pd.concat(pd.read_parquet(f) for f in sorted((root / "data").glob("**/*.parquet")))
excluded = {int(e) for e in args.exclude_episodes.split(",") if e}
frames = frames[~frames["episode_index"].isin(excluded)].sort_values("index")

states = np.stack(frames["observation.state"].to_numpy()).astype(np.float64)
actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
episodes = frames["episode_index"].to_numpy()
within = episodes[1:] == episodes[:-1]  # deltas never cross an episode boundary
deltas = (actions[1:] - actions[:-1])[within]
deltas[:, GRIPPER] = actions[1:][within][:, GRIPPER]


def quantiles(x: np.ndarray) -> dict:
    return {"q01": np.percentile(x, 1, axis=0).round(4).tolist(), "q99": np.percentile(x, 99, axis=0).round(4).tolist()}


stats = {
    "source": {
        "dataset": args.dataset,
        "excluded_episodes": sorted(excluded),
        "episodes": int(len(set(episodes))),
        "frames": int(len(frames)),
    },
    "action": quantiles(deltas),
    "state": quantiles(states),
}
text = json.dumps(stats, indent=2)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(f"wrote {args.output}")
else:
    print(text)

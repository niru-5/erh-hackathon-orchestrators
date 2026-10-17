"""Joint statistics of our chess dataset: ranges, rest pose (first frames) and end pose (last frames)."""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path.home() / ".cache/so101-rollout/flux_convention"
WORK.mkdir(parents=True, exist_ok=True)
ROOT = os.environ.get(
    "DATASET_ROOT",
    str(Path.home() / ".cache/huggingface/lerobot/niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741"),
)
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/data/*/*.parquet"))])
state = np.stack(data["observation.state"].values)
action = np.stack(data["action"].values)
episodes = data["episode_index"].values
first = np.stack([state[episodes == e][0] for e in np.unique(episodes)])
last = np.stack([state[episodes == e][-1] for e in np.unique(episodes)])
rows = []
for j, name in enumerate(NAMES):
    s = state[:, j]
    rows.append(
        {
            "joint": name,
            "min": s.min(),
            "q01": np.quantile(s, 0.01),
            "median": np.median(s),
            "q99": np.quantile(s, 0.99),
            "max": s.max(),
            "rest_mean": first[:, j].mean(),
            "rest_std": first[:, j].std(),
            "end_mean": last[:, j].mean(),
            "end_std": last[:, j].std(),
            "delta_q01": np.quantile(np.diff(action[:, j]), 0.01),
            "delta_q99": np.quantile(np.diff(action[:, j]), 0.99),
        }
    )
table = pd.DataFrame(rows).set_index("joint").round(2)
print(f"{len(np.unique(episodes))} episodes, {len(state)} frames")
print(table.to_string())
json.dump(
    {
        "rest_pose": first.mean(0).tolist(),
        "rest_std": first.std(0).tolist(),
        "table": table.reset_index().to_dict("records"),
    },
    open(WORK / "our_joint_stats.json", "w"),
    indent=1,
)

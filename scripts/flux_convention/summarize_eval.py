"""Summarize teacher-forced dataset-replay runs (python -m rollout run --robot dataset ...).

    python scripts/flux_convention/summarize_eval.py <log dir> [<log dir> ...]

Prints per episode and the frame-weighted mean error per joint (degrees, gripper %), plus the arm mean
(the 5 arm joints), the number the Gawtam/flux3-chess-lora card reports.
"""

import json
import sys
from pathlib import Path

import numpy as np

for directory in sys.argv[1:]:
    run = next(Path(directory).glob("*/summary.json"), None) or Path(directory) / "summary.json"
    summary = json.loads(run.read_text())
    episodes = [e for e in summary.get("episodes", []) if "mae_per_joint" in e]
    if not episodes:
        print(f"{directory}: no finished episodes")
        continue
    names = list(episodes[0]["mae_per_joint"])
    frames = np.array([e["frames"] for e in episodes])
    errors = np.array([[e["mae_per_joint"][n] for n in names] for e in episodes])
    mean = (errors * frames[:, None]).sum(0) / frames.sum()
    print(f"\n{summary['spec']['name']}  ({len(episodes)} episodes, {frames.sum()} frames)")
    for e, row in zip(episodes, errors, strict=True):
        print(f"  ep {e['episode']:2}: " + "  ".join(f"{v:6.2f}" for v in row))
    print(
        "  mean : "
        + "  ".join(f"{v:6.2f}" for v in mean)
        + f"   | arm {mean[:5].mean():.2f} deg, gripper {mean[5]:.2f} %"
    )
    print("         " + "  ".join(f"{n.split('.')[0][:6]:>6}" for n in names))

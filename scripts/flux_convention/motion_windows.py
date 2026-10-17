"""Does a policy predict the demonstrator's motion, or just stay close? Score the teacher-forced plans where
the demonstrator actually moved.

    python scripts/flux_convention/motion_windows.py <log dir> [<log dir> ...]

Each plan covers 32 ticks starting at tick t0 from the recorded command a[t0-1] (teacher forcing). "Hold"
predicts a[t0-1] for all 32 ticks. A window is "moving" when some arm joint's recorded command moves at
least 10 degrees within it. On moving windows:
  - mae / hold mae: mean |predicted - recorded| over the window (arm joints), model vs. hold
  - direction: share of joint-windows with |recorded displacement| >= 5 deg where the model's displacement at
    the end of the window goes the same way and covers at least 10 % of it
  - coverage: projection of the model's end displacement on the recorded one (1 = all of it, 0 = none,
    < 0 = moved the wrong way), median over those joint-windows
"""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path.home() / ".cache/huggingface/lerobot/niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741"
NAMES = ["pan", "lift", "elbow", "wflex", "wroll"]
frames = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{DATASET}/data/*/*.parquet")))
recorded = {
    e: np.stack(g.sort_values("frame_index")["action"].values)[:, :5] for e, g in frames.groupby("episode_index")
}

for directory in sys.argv[1:]:
    direct = Path(directory) / "ticks.jsonl"
    run = direct if direct.exists() else next(Path(directory).glob("*/ticks.jsonl"))
    name = json.loads((run.parent / "summary.json").read_text())["spec"]["name"]
    plans = defaultdict(list)
    for line in run.read_text().splitlines():
        r = json.loads(line)
        plans[(r["episode"], r["plan"])].append(r)
    model_err, hold_err, same_sign, coverage, moving = [], [], [], [], 0
    for (episode, _), rows in plans.items():
        t0 = rows[0]["tick"]
        if t0 == 0 or len(rows) < 32:
            continue
        actions = recorded[episode]
        anchor = actions[t0 - 1]
        truth = actions[t0 : t0 + 32]
        pred = np.array([r["command"][:5] for r in rows[:32]])
        displacement = truth[-1] - anchor
        if np.abs(truth - anchor).max() < 10:
            continue
        moving += 1
        model_err.append(np.abs(pred - truth).mean(0))
        hold_err.append(np.abs(anchor - truth).mean(0))
        for j in range(5):
            if abs(displacement[j]) >= 5:
                moved = pred[-1, j] - anchor[j]
                # right direction = covers at least 10 % of the recorded motion (logs round to 0.001 deg,
                # so a bare sign test would score a model that doesn't move at ~50 %)
                coverage.append(moved / displacement[j])
                same_sign.append(coverage[-1] >= 0.1)
    m, h = np.mean(model_err, 0), np.mean(hold_err, 0)
    print(f"\n{name}: {moving} moving windows of {len(plans)} plans")
    print("  joint      " + "  ".join(f"{n:>6}" for n in NAMES) + "    arm")
    print("  model mae  " + "  ".join(f"{v:6.2f}" for v in m) + f"  {m.mean():6.2f}")
    print("  hold mae   " + "  ".join(f"{v:6.2f}" for v in h) + f"  {h.mean():6.2f}")
    print(
        f"  direction right in {np.mean(same_sign):.0%} of {len(same_sign)} joint-windows that moved >= 5 deg; "
        f"median coverage {np.median(coverage):.2f}"
    )

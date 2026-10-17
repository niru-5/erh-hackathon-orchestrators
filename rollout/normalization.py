"""Per-dataset normalization quantiles, computed the way BFL built the FLUX SO-101 training corpus.

BFL normalized every training dataset with its own quantiles (configs/so101/community_corpus.json:
``x_norm = clip(2 (x - q01) / (q99 - q01) - 1, -6, 6) with the part's own q01/q99``; flux_action
data/lerobot/index.py): q01/q99 = percentiles 1 and 99 over all frames, of the measured state and of the
action targets (joint deltas ``a[t] - a[t-1]`` within each episode, first row 0, gripper absolute).
The released checkpoint instead ships one fixed set (MolmoAct-style, shoulder_lift 85..184 with the arm's
rest pose at the top), which puts our calibration (rest pose at shoulder_lift -97) far out of range.
Normalizing with our own quantiles shows the model our data the way it saw each training dataset.

    .venv/bin/python -m rollout.normalization niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 -o norm.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

GRIPPER = 5  # absolute, not a delta (BFL's absolute_action_dims=(-1,))


def dataset_quantiles(dataset: str, root: str | None = None) -> dict:
    """{"state": {"q01", "q99"}, "action": {"q01", "q99"}, "source": dataset} from a LeRobot v3 dataset."""
    import pandas as pd

    directory = _dataset_dir(dataset, root)
    frames = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{directory}/data/*/*.parquet")))
    frames = frames.sort_values(["episode_index", "frame_index"])
    state = np.stack(frames["observation.state"].values).astype(np.float64)
    action = np.stack(frames["action"].values).astype(np.float64)
    targets = np.zeros_like(action)
    for episode in np.unique(frames["episode_index"].values):
        rows = np.flatnonzero(frames["episode_index"].values == episode)
        targets[rows[1:]] = np.diff(action[rows], axis=0)
    targets[:, GRIPPER] = action[:, GRIPPER]

    def q(values):
        return {
            "q01": np.percentile(values, 1, axis=0).round(6).tolist(),
            "q99": np.percentile(values, 99, axis=0).round(6).tolist(),
        }

    return {"source": dataset, "frames": int(len(state)), "state": q(state), "action": q(targets)}


def load(spec: str | dict | None) -> dict | None:
    """A quantiles dict, from a dict, a JSON string, a JSON file or a dataset (repo id / local dir)."""
    if spec is None or isinstance(spec, dict):
        return spec
    if spec.lstrip().startswith("{"):
        return json.loads(spec)
    if spec.endswith(".json") and Path(spec).expanduser().is_file():
        return json.loads(Path(spec).expanduser().read_text())
    return dataset_quantiles(spec)


def _dataset_dir(dataset: str, root: str | None) -> Path:
    if root:
        return Path(root)
    path = Path(dataset).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(dataset, repo_type="dataset", allow_patterns=["data/*", "meta/info.json"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rollout.normalization",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="LeRobot dataset repo id or local dir")
    parser.add_argument("-o", "--output", type=Path, help="write the quantiles JSON here (default: print)")
    args = parser.parse_args()
    stats = dataset_quantiles(args.dataset)
    text = json.dumps(stats, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

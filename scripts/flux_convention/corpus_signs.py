"""Where does the rest pose sit inside each training part's own [q01, q99] range? (sign check per joint)

For parts of BFL's SO-101 corpus (configs/so101/community_corpus.json), read frame 0 of a few episodes from the
source repo and express it in that part's normalized coordinates, 2 (x - q01) / (q99 - q01) - 1. Compare with
our dataset: if the rest pose sits at the opposite end for a joint, that joint's sign is flipped vs. training.
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

WORK = Path.home() / ".cache/so101-rollout/flux_convention"
CORPUS_URL = "https://raw.githubusercontent.com/black-forest-labs/flux-action/830f1ce178d7aea52f18fde31d8e8e36ce6bae21/configs/so101/community_corpus.json"
CORPUS = WORK / "community_corpus.json"
CACHE = str(WORK / "corpus_src")
NAMES = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
if not CORPUS.exists():
    import urllib.request

    WORK.mkdir(parents=True, exist_ok=True)
    CORPUS.write_bytes(urllib.request.urlopen(CORPUS_URL, timeout=60).read())
corpus = json.load(open(CORPUS))
parts = [p for p in corpus["parts"] if p.get("action_dim") == 6 and p.get("train_windows", 0) > 0]
print(
    f"{len(parts)} training parts; origins {pd.Series([p['origin'] for p in parts]).value_counts().to_dict()}; "
    f"windows {sum(p['train_windows'] for p in parts)}"
)
api = HfApi()


def rest_norm(part):
    repo = part["dataset"]
    try:
        files = [
            f
            for f in api.list_repo_files(repo, repo_type="dataset")
            if f.startswith("data/") and f.endswith(".parquet")
        ]
        path = hf_hub_download(repo, sorted(files)[0], repo_type="dataset", cache_dir=CACHE)
        df = pd.read_parquet(path)
        col = "observation.state" if "observation.state" in df else None
        first = (
            np.stack(df[df["frame_index"] == 0][col].values) if "frame_index" in df else np.stack(df[col].values[:1])
        )
        q01, q99 = np.array(part["state_q01"]), np.array(part["state_q99"])
        return part, np.median(2 * (first - q01) / np.maximum(q99 - q01, 1e-6) - 1, 0), np.median(first, 0)
    except Exception as exc:  # noqa: BLE001
        return part, None, str(exc)[:80]


random.seed(1)
by_origin = {o: [p for p in parts if p["origin"] == o] for o in {p["origin"] for p in parts}}
sample = [p for o, ps in by_origin.items() for p in random.sample(ps, min(30, len(ps)))]
with ThreadPoolExecutor(12) as pool:
    results = list(pool.map(rest_norm, sample))
ok = [(p, n, raw) for p, n, raw in results if n is not None]
print(f"rest pose read for {len(ok)} of {len(sample)} sampled parts")
for origin in by_origin:
    norms = (
        np.stack([n for p, n, _ in ok if p["origin"] == origin])
        if any(p["origin"] == origin for p, _, _ in ok)
        else None
    )
    if norms is None:
        continue
    raws = np.stack([r for p, _, r in ok if p["origin"] == origin])
    print(f"\n{origin}: {len(norms)} parts. Rest pose in the part's normalized coords (-1 = q01 end, +1 = q99 end)")
    print(
        pd.DataFrame(
            {
                "median": np.median(norms, 0),
                "q25": np.quantile(norms, 0.25, 0),
                "q75": np.quantile(norms, 0.75, 0),
                "share<0": (norms < 0).mean(0),
                "raw median": np.median(raws, 0),
            },
            index=NAMES,
        )
        .round(2)
        .to_string()
    )

ours = json.load(open(WORK / "our_joint_stats.json"))  # written by joint_stats.py
t = {r["joint"]: r for r in ours["table"]}
keys = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
q01 = np.array([t[k]["q01"] for k in keys])
q99 = np.array([t[k]["q99"] for k in keys])
print(
    "\nours: rest pose in our normalized coords",
    np.round(2 * (np.array(ours["rest_pose"]) - q01) / (q99 - q01) - 1, 2).tolist(),
)

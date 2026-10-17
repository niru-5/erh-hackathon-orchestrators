"""Per robot_type group in community_dataset_v3: joint names, pooled state ranges and rest pose, to see which
group matches the FLUX SO-101 normalization quantiles."""

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO, REV = "lerobot/community_dataset_v3", "ab92ac3fa4d0c08336e21b8e51cb9dc279535539"
OUT = Path.home() / ".cache/so101-rollout/flux_convention/community"
MODEL_Q01 = [-92.0, 84.8, 61.4, -24.0, -124.5, 0.5]
MODEL_Q99 = [54.1, 184.2, 178.9, 80.6, 60.2, 53.6]

files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
roots = sorted({f.rsplit("/meta/info.json", 1)[0] for f in files if f.endswith("/meta/info.json")})
get = lambda p: hf_hub_download(REPO, p, repo_type="dataset", revision=REV, cache_dir=str(OUT))  # noqa: E731


def load(root):
    try:
        info = json.load(open(get(f"{root}/meta/info.json")))
        stats = json.load(open(get(f"{root}/meta/stats.json")))["observation.state"]
        return root, info, stats
    except Exception:  # noqa: BLE001
        return root, None, None


with ThreadPoolExecutor(16) as pool:
    rows = [r for r in pool.map(load, roots) if r[1]]

groups = {}
for root, info, stats in rows:
    rt = info.get("robot_type")
    if rt not in ("so101", "so101_follower", "so100", "so100_follower"):
        continue
    names = info["features"]["observation.state"].get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()))
    if not names or len(names) != 6:
        continue
    groups.setdefault(rt, []).append((root, tuple(names), stats))

for rt, items in groups.items():
    names = Counter(n for _, n, _ in items).most_common(2)
    lo = np.median([np.array(s.get("q01", s["min"]), float) for _, _, s in items], 0)
    hi = np.median([np.array(s.get("q99", s["max"]), float) for _, _, s in items], 0)
    mean = np.median([np.array(s["mean"], float) for _, _, s in items], 0)
    print(
        f"\n== {rt}: {len(items)} datasets; names {names[0][0]} (x{names[0][1]})"
        + (f", next {names[1]}" if len(names) > 1 else "")
    )
    print(
        pd.DataFrame(
            {"median q01": lo, "median q99": hi, "median mean": mean, "MODEL q01": MODEL_Q01, "MODEL q99": MODEL_Q99},
            index=[n.split(".")[0].replace("main_", "") for n in names[0][0]],
        )
        .round(1)
        .to_string()
    )

    random.seed(0)
    sample = random.sample(items, min(25, len(items)))

    def first(item):
        root = item[0]
        try:
            df = pd.read_parquet(
                get(f"{root}/data/chunk-000/file-000.parquet"), columns=["observation.state", "frame_index"]
            )
            return np.stack(df[df["frame_index"] == 0]["observation.state"].values)
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(8) as pool:
        firsts = [f for f in pool.map(first, sample) if f is not None]
    if firsts:
        rest = np.concatenate(firsts)
        median = np.median(rest, 0).round(1).tolist()
        print(f"   rest pose (frame 0, {len(rest)} episodes of {len(firsts)} datasets): median {median}")

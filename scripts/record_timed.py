"""Run the official `lerobot-record` and measure the real (wall-clock) rate of every recorded frame.

lerobot stores synthetic timestamps (frame_index / fps), so a dataset always *claims* 30 Hz. This wrapper
times each iteration of lerobot's own record loop and prints a per-episode report at the end, so you can
tell whether the recording really ran at the target rate. Nothing about the recording itself changes.

Takes exactly the same arguments as `lerobot-record` (scripts/record_so101.sh calls it).
The report is also saved to ~/.cache/huggingface/lerobot/timing/<repo_id>.json.
"""

import json
import time

import numpy as np

import lerobot.scripts.lerobot_record as lr
from lerobot.utils.constants import HF_LEROBOT_HOME

_real_record_loop = lr.record_loop
_real_precise_sleep = lr.precise_sleep

_state = {"ends": None, "fps": None, "repo_id": None}
_episodes: dict[int, dict] = {}  # episode index -> stats of its last (kept) attempt
_discarded: list[dict] = []


def _timed_precise_sleep(seconds: float) -> None:
    _real_precise_sleep(seconds)
    if _state["ends"] is not None:
        _state["ends"].append(time.perf_counter())


def _timed_record_loop(*args, **kwargs):
    dataset = kwargs.get("dataset")
    if dataset is None:  # reset period: nothing is recorded
        return _real_record_loop(*args, **kwargs)

    fps = kwargs["fps"]
    episode = dataset.num_episodes
    _state.update(fps=fps, repo_id=dataset.repo_id)
    start = time.perf_counter()
    _state["ends"] = ends = []
    try:
        return _real_record_loop(*args, **kwargs)
    finally:
        _state["ends"] = None
        periods_ms = np.diff([start, *ends]) * 1000
        late_ms = 1000 / fps * 1.5  # a frame is "late" if it took 1.5x the budget (50 ms at 30 Hz)
        stats = {
            "episode": episode,
            "frames": len(ends),
            "wall_s": round(ends[-1] - start, 3) if ends else 0.0,
            # rate over frames 2..N: the first frame is where lerobot starts the video encoders
            "real_hz": round((len(ends) - 1) / (ends[-1] - ends[0]), 2) if len(ends) > 1 else 0.0,
            "first_frame_ms": round(float(periods_ms[0]), 1) if len(ends) else 0.0,
            "late_frames_after_first": int((periods_ms[1:] > late_ms).sum()),
            "p99_ms": round(float(np.percentile(periods_ms[1:], 99)), 1) if len(ends) > 1 else 0.0,
            "worst_ms": round(float(periods_ms[1:].max()), 1) if len(ends) > 1 else 0.0,
        }
        if episode in _episodes:  # same index recorded again -> the previous attempt was discarded (left arrow)
            _discarded.append(_episodes[episode])
        _episodes[episode] = stats


def _report() -> None:
    if not _episodes:
        return
    fps = _state["fps"]
    print(f"\n===== Recording timing report: {_state['repo_id']} (target {fps} Hz) =====")
    print(f"{'ep':>3} {'frames':>7} {'wall s':>7} {'real Hz':>8} {'late':>5} {'p99 ms':>7} {'worst ms':>9} {'1st ms':>7}  verdict")
    all_ok = True
    for s in sorted(_episodes.values(), key=lambda s: s["episode"]):
        # Allow the first frame of an episode to be late: lerobot starts the video encoders there.
        ok = s["real_hz"] >= fps * 0.98 and s["late_frames_after_first"] == 0
        all_ok &= ok
        print(
            f"{s['episode']:>3} {s['frames']:>7} {s['wall_s']:>7.1f} {s['real_hz']:>8.2f} {s['late_frames_after_first']:>5} "
            f"{s['p99_ms']:>7.1f} {s['worst_ms']:>9.1f} {s['first_frame_ms']:>7.1f}  {'OK' if ok else 'NOT 30 Hz - re-record'}"
        )
    if _discarded:
        print(f"({len(_discarded)} discarded attempt(s) not shown)")
    print(
        "RESULT: all episodes at target rate"
        if all_ok
        else "RESULT: some episodes ran slower than target (see above)"
    )
    print(
        "real Hz / late / p99 / worst exclude the first frame (lerobot starts the video encoders there, see '1st ms'). "
        f"late = frames over {1500 / fps:.0f} ms."
    )

    out = HF_LEROBOT_HOME / "timing" / f"{_state['repo_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fps": fps, "episodes": list(_episodes.values()), "discarded": _discarded}, indent=2))
    print(f"Saved to {out}")


lr.record_loop = _timed_record_loop
lr.precise_sleep = _timed_precise_sleep

if __name__ == "__main__":
    try:
        lr.main()
    finally:
        _report()

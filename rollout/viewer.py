"""python -m rollout view [run dir]: rollout_logs/<timestamp>/viewer.html, a self-contained page to scrub a run.

Works on any run (joint charts, plan table, findings); with run --debug it also shows the camera frames, the
images each plan was given, and FLUX's imagined future next to what the cameras then saw.
"""

import json
from pathlib import Path

import numpy as np

from .contract import ACTION_NAMES

TEMPLATE = Path(__file__).with_name("viewer_template.html")
JUMP_DEG = 5.0  # first action of a plan this far from the last command sent: the plan doesn't chain on
DRIFT_DEG = 10.0  # the arm this far from the plan's own action at the end of a plan: it didn't follow


def latest_run(log_dir: Path) -> Path:
    """The most recent run under ``log_dir``, also inside group folders (rollout_logs/<group>/<timestamp>)."""
    runs = sorted((p.parent for p in log_dir.rglob("ticks.jsonl")), key=lambda p: (p / "ticks.jsonl").stat().st_mtime)
    if not runs:
        raise SystemExit(f"no runs in {log_dir}")
    return runs[-1]


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    ticks = _jsonl(run / "ticks.jsonl")
    timing = {(p["episode"], p["plan"]): p for p in _jsonl(run / "plans.jsonl")}
    debug_plans = {(p["episode"], p["plan"]): p for p in _jsonl(run / "debug/plans.jsonl")}
    spec = summary.get("spec", {})
    fps = float(spec.get("fps", 30.0))
    results = {r["episode"]: r for r in summary.get("episodes", [])}
    limits = summary.get("limits")

    episodes = []
    for episode in sorted({t["episode"] for t in ticks} | {k[0] for k in timing}):
        rows = [t for t in ticks if t["episode"] == episode]
        plans = []
        for key in sorted(k for k in timing if k[0] == episode):
            plan = {**debug_plans.get(key, {}), **{k: v for k, v in timing[key].items() if k not in ("episode",)}}
            plan.setdefault("timing", {k: v for k, v in timing[key].items() if k.endswith("_ms")})
            plans.append(plan)
        frames_dir = run / "debug" / "frames" / f"ep{episode:03d}"
        cameras = sorted(p.name for p in frames_dir.iterdir()) if frames_dir.is_dir() else []
        episodes.append(
            {
                "episode": episode,
                "ticks": [{k: t[k] for k in ("tick", "plan", "t", "state", "command", "sent") if k in t} for t in rows],
                "plans": plans,
                "frames": {c: f"debug/frames/ep{episode:03d}/{c}/" for c in cameras},
                "result": results.get(episode, {}),
            }
        )
        episodes[-1]["findings"] = findings(episodes[-1], fps, limits)
    return {
        "run": run.name,
        "args": summary.get("args", {}),
        "spec": spec,
        "limits": limits,
        "joints": [n.removesuffix(".pos") for n in ACTION_NAMES],
        "thresholds": {"jump_deg": JUMP_DEG, "drift_deg": DRIFT_DEG},
        "episodes": episodes,
    }


def plan_metrics(episode: dict, limits: dict | None) -> None:
    """Per plan, in place: ``jump`` (first action vs the last command sent before it), ``drift`` (arm vs the
    plan's last executed action), ``clipped`` ticks, ``beyond_limits`` (planned actions outside the joint range)."""
    by_tick = {t["tick"]: t for t in episode["ticks"]}
    for plan in episode["plans"]:
        start = plan["tick"]
        executed = [by_tick[k] for k in sorted(by_tick) if by_tick[k].get("plan") == plan["plan"]]
        plan["executed_ticks"] = len(executed)
        actions = np.asarray(plan["actions"]) if plan.get("actions") else None
        if actions is None and executed:
            actions = np.asarray([t["command"] for t in executed])
        previous = by_tick.get(start - 1)
        if actions is not None and len(actions) and previous:
            gap = np.abs(actions[0] - np.asarray(previous["sent"]))
            plan["jump"] = {"deg": float(gap.max()), "joint": int(gap.argmax())}
        if executed and actions is not None:
            last = executed[-1]
            # the state the arm reaches after the last executed command is the next tick's measurement
            after = by_tick.get(last["tick"] + 1, last)
            gap = np.abs(np.asarray(after["state"]) - actions[len(executed) - 1])
            plan["drift"] = {"deg": float(gap.max()), "joint": int(gap.argmax())}
            plan["clipped"] = int(sum(not np.allclose(t["command"], t["sent"], atol=1e-3) for t in executed))
        if limits and actions is not None:
            low, high = np.asarray(limits["low"]), np.asarray(limits["high"])
            planned = np.asarray(plan["chunk"]) if plan.get("chunk") else actions
            plan["beyond_limits"] = int(((planned < low) | (planned > high)).any(axis=1).sum())


def findings(episode: dict, fps: float, limits: dict | None) -> list[dict]:
    """Plain-language notes on what looks wrong, in tick order; each links to a tick in the viewer."""
    plan_metrics(episode, limits)
    joints = [n.removesuffix(".pos") for n in ACTION_NAMES]
    notes = []

    def note(tick, level, text):
        notes.append({"tick": int(tick), "level": level, "text": f"{tick / fps:.1f} s (tick {tick}): {text}"})

    for plan in episode["plans"]:
        n, tick = plan["plan"], plan["tick"]
        jump, drift = plan.get("jump"), plan.get("drift")
        if jump and jump["deg"] > JUMP_DEG:
            note(
                tick,
                "warn",
                f"plan {n} starts {jump['deg']:.1f}° away from the last command on {joints[jump['joint']]}: "
                "it doesn't continue the motion it was given",
            )
        if drift and drift["deg"] > DRIFT_DEG:
            note(
                tick + max(plan.get("executed_ticks", 1) - 1, 0),
                "warn",
                f"after plan {n} the arm is {drift['deg']:.1f}° from where the plan put {joints[drift['joint']]} "
                "(blocked, capped by max_relative_target, or clamped)",
            )
        if plan.get("clipped"):
            note(
                tick,
                "info",
                f"plan {n}: {plan['clipped']} of {plan['executed_ticks']} commands changed before sending "
                "(joint limits or the per-tick max_relative_target cap)",
            )
        if plan.get("beyond_limits"):
            note(tick, "info", f"plan {n}: {plan['beyond_limits']} predicted actions outside the joint range")
    ticks = episode["ticks"]
    if len(ticks) > 1:
        states = np.asarray([t["state"] for t in ticks])
        moved = np.abs(states - states[0]).max(axis=0)
        still = [joints[i] for i in range(len(joints) - 1) if moved[i] < 2.0]
        if len(still) == len(joints) - 1:
            note(ticks[-1]["tick"], "warn", "the arm never moved more than 2° on any joint")
    stopped = episode["result"].get("stopped_by", "")
    if stopped and stopped not in ("end of data",):
        last = ticks[-1]["tick"] if ticks else 0
        note(last, "stop", f"stopped by {stopped}")
    elif ticks:
        note(ticks[-1]["tick"], "info", f"ran to the end ({stopped or 'episode time'})")
    return sorted(notes, key=lambda x: x["tick"])


def build(run: Path) -> Path:
    data = load_run(run)
    # </ inside the embedded JSON would end the <script> element
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__ROLLOUT_DATA__", payload).replace("__ROLLOUT_TITLE__", f"Rollout {run.name}")
    out = run / "viewer.html"
    out.write_text(html)
    return out

"""Control loop: sync replanning at the model's fps, safety limits and per-run logs.

Every tick: observe, append to the history, ask the backend for a plan when the queue is empty, send
the next command. This is flux's ``select_action`` loop with the model call moved behind a backend,
see docs/inference.md.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backends import Backend
from .contract import ACTION_NAMES, PolicySpec
from .history import History
from .robots import Robot

logger = logging.getLogger(__name__)


class JointLimits:
    """Clamp commands to the joint range seen in a dataset (meta/stats.json min/max +- margin).

    ``abort_margin``: a command more than this many degrees outside the range isn't clamped; the episode
    stops instead (``runaway``). Small overshoots are normal, but a policy commanding shoulder_lift = 844
    degrees has lost track of the arm (2026-10-17, FLUX LoRA step 500), and clamping that just presses the arm
    against the limit.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        abort_margin: float | None = None,
        reach_low: np.ndarray | None = None,
        reach_high: np.ndarray | None = None,
    ):
        self.low, self.high = np.asarray(low, np.float32), np.asarray(high, np.float32)
        self.abort_margin = abort_margin
        # the range the demonstrations' arm actually reached (observation.state), inside the commanded range
        self.reach_low = self.low if reach_low is None else np.asarray(reach_low, np.float32)
        self.reach_high = self.high if reach_high is None else np.asarray(reach_high, np.float32)

    def beyond_reach(self, goal: np.ndarray) -> np.ndarray:
        """Per joint, the reached-range edge the goal lies beyond, else NaN (see StallGuard)."""
        return np.where(
            goal < self.reach_low, self.reach_low, np.where(goal > self.reach_high, self.reach_high, np.nan)
        )

    def runaway(self, action: np.ndarray) -> str | None:
        """The joints whose command is beyond ``abort_margin`` outside the range, or None."""
        if self.abort_margin is None:
            return None
        excess = np.maximum(self.low - action, action - self.high)
        hit = np.flatnonzero(excess > self.abort_margin)
        if not len(hit):
            return None
        return ", ".join(
            f"{ACTION_NAMES[i]} commanded {action[i]:.0f} (range {self.low[i]:.0f}..{self.high[i]:.0f})" for i in hit
        )

    @classmethod
    def from_dataset(cls, dataset: str, margin: float, abort_margin: float | None = None) -> "JointLimits":
        path = Path(dataset).expanduser()
        if path.is_dir():
            stats_path = path / "meta/stats.json"
        else:
            from huggingface_hub import hf_hub_download
            from lerobot.utils.constants import HF_LEROBOT_HOME

            local = HF_LEROBOT_HOME / dataset / "meta/stats.json"
            stats_path = (
                local if local.exists() else Path(hf_hub_download(dataset, "meta/stats.json", repo_type="dataset"))
            )
        stats = json.loads(stats_path.read_text())
        low = np.minimum(stats["observation.state"]["min"], stats["action"]["min"]) - margin
        high = np.maximum(stats["observation.state"]["max"], stats["action"]["max"]) + margin
        gripper = ACTION_NAMES.index("gripper.pos")
        low[gripper], high[gripper] = max(low[gripper], 0.0), min(high[gripper], 100.0)
        reach = stats["observation.state"]
        return cls(low, high, abort_margin, np.array(reach["min"]) - margin, np.array(reach["max"]) + margin)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.low, self.high)

    def describe(self) -> str:
        ranges = zip(ACTION_NAMES, self.low, self.high, strict=True)
        return ", ".join(f"{n.removesuffix('.pos')} [{lo:.0f}, {hi:.0f}]" for n, lo, hi in ranges)


class StallGuard:
    """Detect a blocked arm (collision): a joint far from its goal that has stopped moving.

    Normal tracking lags the goal during fast moves, but the joint keeps moving; a joint pressed against
    the table sits still with the goal held ahead of it (2026-10-17: shoulder_lift stuck at 12.0 with the
    goal 10 degrees beyond for 5 s, until the servos faulted). The gripper is excluded: squeezing a piece
    stalls it on purpose. ``gap_deg`` must stay below lerobot's ``max_relative_target``, which caps how far
    the goal can be from the arm (a 6 degree gap never tripped under the 5 degree cap: step-500 run).

    Not a stall: a goal beyond the range the demonstrations' arm reached, while the arm sits within
    ``range_end_deg`` of that edge. At rest the folded arm stops mechanically about 7 degrees short of the
    leader's command (demos: shoulder_lift commanded down to -103.6, the arm stops at -97), and policies
    reproduce that. A real collision happens mid-range, far from those edges.
    """

    def __init__(self, gap_deg: float = 3.0, still_deg: float = 0.5, ticks: int = 10, range_end_deg: float = 10.0):
        self.gap_deg, self.still_deg, self.ticks, self.range_end_deg = gap_deg, still_deg, ticks, range_end_deg
        self.joints = [i for i, name in enumerate(ACTION_NAMES) if name != "gripper.pos"]
        self.reset()

    def reset(self) -> None:
        self._count = np.zeros(len(ACTION_NAMES), int)
        self._last_state: np.ndarray | None = None
        self._last_goal: np.ndarray | None = None
        self._last_pinned: np.ndarray | None = None

    def update(self, state: np.ndarray, goal: np.ndarray, pinned: np.ndarray | None = None) -> str | None:
        """Call once per tick with the measured state and the goal just sent; returns the stalled joint.

        ``pinned``: per joint, the reached-range edge the goal lies beyond (``JointLimits.beyond_reach``), else NaN.
        """
        stalled = None
        if self._last_goal is not None:
            gap = np.abs(self._last_goal - state)
            still = np.abs(state - self._last_state) < self.still_deg
            at_range_end = np.zeros(len(state), bool)
            if self._last_pinned is not None:
                with np.errstate(invalid="ignore"):
                    at_range_end = np.abs(self._last_pinned - state) <= self.range_end_deg
            blocked = (gap > self.gap_deg) & still & ~at_range_end
            self._count = np.where(blocked, self._count + 1, 0)
            hit = [i for i in self.joints if self._count[i] >= self.ticks]
            if hit:
                stalled = ", ".join(f"{ACTION_NAMES[i]} (gap {gap[i]:.0f} deg)" for i in hit)
        self._last_state, self._last_goal = state.copy(), goal.copy()
        self._last_pinned = None if pinned is None else np.asarray(pinned, np.float32).copy()
        return stalled


@dataclass
class EpisodeStats:
    ticks: int = 0
    plans: int = 0
    wall_s: float = 0.0
    waiting_s: float = 0.0
    clipped_ticks: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    stopped_by: str = ""

    def to_dict(self) -> dict:
        def pct(values, q):
            return round(float(np.percentile(values, q)), 1) if values else None

        return {
            "ticks": self.ticks,
            "plans": self.plans,
            "wall_s": round(self.wall_s, 2),
            "effective_hz": round(self.ticks / self.wall_s, 2) if self.wall_s else None,
            "waiting_fraction": round(self.waiting_s / self.wall_s, 3) if self.wall_s else None,
            "request_ms": {
                "mean": round(float(np.mean(self.latencies_ms)), 1) if self.latencies_ms else None,
                "p50": pct(self.latencies_ms, 50),
                "p95": pct(self.latencies_ms, 95),
                "max": pct(self.latencies_ms, 100),
            },
            "server_inference_ms_p50": pct(self.inference_ms, 50),
            "clipped_ticks": self.clipped_ticks,
            "stopped_by": self.stopped_by,
        }


class RunLog:
    """rollout_logs/<timestamp>/{ticks,plans}.jsonl + summary.json."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self._ticks = (directory / "ticks.jsonl").open("w")
        self._plans = (directory / "plans.jsonl").open("w")

    def tick(self, **record) -> None:
        self._ticks.write(json.dumps(_jsonable(record)) + "\n")

    def plan(self, **record) -> None:
        self._plans.write(json.dumps(_jsonable(record)) + "\n")
        self._plans.flush()

    def summary(self, summary: dict) -> None:
        (self.directory / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    def close(self) -> None:
        self._ticks.close()
        self._plans.close()


def run_episode(
    robot: Robot,
    backend: Backend,
    spec: PolicySpec,
    history: History,
    *,
    task: str,
    episode: int,
    max_ticks: int,
    limits: JointLimits | None = None,
    log: RunLog | None = None,
    events: dict | None = None,
    teacher_forcing: bool = False,
    stall_guard: StallGuard | None = None,
) -> EpisodeStats:
    """One episode with a fresh history (flux's ``policy.reset()``).

    ``teacher_forcing`` (recorded data only): the history records the demonstrator's command for each tick
    instead of the model's, so every plan starts from a consistent state/command history. Without it, a
    replayed recording doesn't follow the model's commands and the history drifts out of distribution.
    """
    if teacher_forcing and not hasattr(robot, "recorded_command"):
        raise ValueError("teacher forcing needs a robot that replays recorded commands")
    period = 1.0 / spec.fps
    history.reset()
    queue: deque[np.ndarray] = deque()
    last_command: np.ndarray | None = None
    last_goal: np.ndarray | None = None
    last_pinned: np.ndarray | None = None  # per joint: the reached-range edge the goal lies beyond, else NaN
    stats = EpisodeStats()
    if stall_guard:
        stall_guard.reset()
    started = next_tick = time.perf_counter()
    while stats.ticks < max_ticks:
        if events and (events["stop_recording"] or events["exit_early"]):
            stats.stopped_by = "stop key" if events["stop_recording"] else "next key"
            break
        observation = robot.observe()
        if observation is None:
            stats.stopped_by = "end of data"
            break
        if stall_guard and last_goal is not None:
            stalled = stall_guard.update(observation.state, last_goal, last_pinned)
            if stalled:
                robot.act(observation.state)  # stop pushing: hold where the arm is
                stats.stopped_by = f"stall: {stalled}"
                logger.warning("episode %d: arm blocked, stopping (%s)", episode, stalled)
                break
        history.append(observation, last_command)
        if not queue:
            request_started = time.perf_counter()
            plan = backend.predict(history.window(task))
            waited = time.perf_counter() - request_started
            if plan.actions.shape != (spec.n_action_steps, len(spec.action_names)):
                raise RuntimeError(f"plan has shape {plan.actions.shape}, expected ({spec.n_action_steps}, 6)")
            queue.extend(plan.actions)
            stats.plans += 1
            stats.waiting_s += waited
            stats.latencies_ms.append(plan.timing.get("client_total_ms", waited * 1e3))
            if "inference_ms" in plan.timing:
                stats.inference_ms.append(plan.timing["inference_ms"])
            if log:
                log.plan(episode=episode, plan=stats.plans, tick=stats.ticks, model=plan.model, **plan.timing)
            logger.info(
                "episode %d plan %d at tick %d: %.0f ms (server inference %s ms)",
                episode,
                stats.plans,
                stats.ticks,
                stats.latencies_ms[-1],
                _fmt(plan.timing.get("inference_ms")),
            )
            next_tick = time.perf_counter()  # the arm held still while waiting; restart the tick clock
        command = queue.popleft()
        runaway = limits.runaway(command) if limits else None
        if runaway:
            robot.act(observation.state)  # don't press toward the limit: hold where the arm is
            stats.stopped_by = f"runaway: {runaway}"
            logger.warning("episode %d: command far outside the joint range, stopping (%s)", episode, runaway)
            break
        goal = limits(command) if limits else command
        if not np.array_equal(goal, command):
            stats.clipped_ticks += 1
        sent = robot.act(goal)
        last_goal = sent
        last_pinned = limits.beyond_reach(goal) if limits else None
        # The history records what the arm was actually told (after our limits and lerobot's per-tick cap),
        # not the raw command: otherwise a clamp leaves a command history the arm never followed, and FLUX,
        # which chains each plan onto that history, runs away (step-500 run: shoulder_lift commanded 844).
        last_command = robot.recorded_command() if teacher_forcing else sent
        if log:
            log.tick(
                episode=episode,
                tick=stats.ticks,
                plan=stats.plans,
                t=time.perf_counter() - started,
                state=observation.state,
                command=command,
                sent=sent,
            )
        stats.ticks += 1
        if robot.realtime:
            next_tick += period
            _sleep_until(next_tick)
    stats.wall_s = time.perf_counter() - started
    return stats


def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0.002:
        time.sleep(remaining - 0.002)
    while time.perf_counter() < deadline:
        pass


def _fmt(value) -> str:
    return f"{value:.0f}" if isinstance(value, (int, float)) else "?"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return np.round(value.astype(float), 3).tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

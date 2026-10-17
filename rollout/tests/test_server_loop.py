"""Server + RemoteBackend over real HTTP, and the control loop, with the mock model (no GPU, no robot)."""

import json
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from rollout.backends.local import LocalBackend
from rollout.backends.remote import RemoteBackend
from rollout.contract import Observation, Plan, PolicySpec
from rollout.history import History
from rollout.loop import JointLimits, RunLog, StallGuard, run_episode
from rollout.models.hold import HoldModel
from rollout.server.app import create_app


@pytest.fixture
def server_url():
    """Serve the hold model on a free localhost port, with an API key."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(HoldModel(), api_key="secret"), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


class FakeRobot:
    """Synthetic observations; the state follows the last goal like a perfect position controller."""

    realtime = False

    def __init__(self, ticks: int):
        self.ticks, self.state, self.sent = ticks, np.array([0, 0, 0, 50, 0, 10], np.float32), []

    def connect(self):
        pass

    def start_episode(self, index):
        self.remaining = self.ticks

    def observe(self):
        if self.remaining == 0:
            return None
        self.remaining -= 1
        image = np.zeros((480, 640, 3), np.uint8)
        return Observation(self.state.copy(), {"scene": image, "wrist": image})

    def act(self, action):
        self.sent.append(action)
        self.state = action.astype(np.float32)
        return action

    def summary(self):
        return {}

    def disconnect(self):
        pass


class RampModel:
    """Moves shoulder_pan +1 per tick from the window's last command (like flux's delta integration)."""

    def spec(self):
        return PolicySpec(name="ramp")

    def predict(self, window):
        steps = np.arange(1, 33, dtype=np.float32)[:, None] * np.eye(6, dtype=np.float32)[0]
        return Plan(window.commands[-1] + steps, {"inference_ms": 1.0}, "ramp")


def test_remote_backend_roundtrip_and_auth(server_url, monkeypatch):
    monkeypatch.setenv("ROLLOUT_API_KEY", "wrong")
    with pytest.raises(RuntimeError, match="401"):
        RemoteBackend(server_url).spec()

    monkeypatch.setenv("ROLLOUT_API_KEY", "secret")
    backend = RemoteBackend(server_url)
    spec = backend.spec()
    assert spec.camera_keys == ("scene", "wrist") and spec.n_action_steps == 32
    history = History(spec)
    image = np.zeros((480, 640, 3), np.uint8)
    history.append(Observation(np.arange(6, dtype=np.float32), {"scene": image, "wrist": image}), None)
    plan = backend.predict(history.window("hold"))
    assert plan.actions.shape == (32, 6)
    np.testing.assert_allclose(plan.actions, np.tile(np.arange(6, dtype=np.float32), (32, 1)))
    assert {"client_total_ms", "server_total_ms", "decode_ms", "request_bytes"} <= set(plan.timing)
    assert plan.timing["request_bytes"] < 30_000  # 4 small JPEGs


def test_server_rejects_bad_window(server_url, monkeypatch):
    import requests

    response = requests.post(
        f"{server_url}/predict",
        json={"task": "x", "states": [[0] * 6], "commands": [[0] * 6], "images": {}},
        headers={"X-Rollout-Key": "secret"},
    )
    assert response.status_code == 422


def test_loop_replans_every_32_ticks_and_chains_commands(tmp_path):
    robot = FakeRobot(ticks=70)
    backend = LocalBackend(RampModel())
    spec = backend.spec()
    log = RunLog(tmp_path / "run")
    robot.start_episode(0)
    stats = run_episode(robot, backend, spec, History(spec), task="t", episode=0, max_ticks=1000, log=log)
    log.close()
    assert stats.ticks == 70 and stats.plans == 3 and stats.stopped_by == "end of data"
    # each plan continues from the previous plan's last command: a continuous +1 ramp
    np.testing.assert_allclose(np.stack(robot.sent)[:, 0], np.arange(1, 71))
    assert len((tmp_path / "run/ticks.jsonl").read_text().splitlines()) == 70
    assert len((tmp_path / "run/plans.jsonl").read_text().splitlines()) == 3


def test_limits_clip_goals_and_the_history_records_the_clipped_goal(tmp_path):
    """The next plan starts from what the arm was told (10), not from the raw command chain (32)."""
    robot = FakeRobot(ticks=40)
    backend = LocalBackend(RampModel())
    spec = backend.spec()
    limits = JointLimits(np.full(6, -100.0), np.array([10.0, 100, 100, 100, 100, 100]))
    log = RunLog(tmp_path / "run")
    robot.start_episode(0)
    stats = run_episode(robot, backend, spec, History(spec), task="t", episode=0, max_ticks=40, limits=limits, log=log)
    log.close()
    sent = np.stack(robot.sent)[:, 0]
    assert sent.max() == 10.0 and stats.clipped_ticks == 30
    commands = [json.loads(line)["command"][0] for line in (tmp_path / "run/ticks.jsonl").read_text().splitlines()]
    assert commands[31] == 32 and commands[32] == 11  # plan 2 continues from the clipped goal


def test_runaway_command_stops_the_episode_and_holds():
    robot = FakeRobot(ticks=100)
    backend = LocalBackend(RampModel())
    spec = backend.spec()
    limits = JointLimits(np.full(6, -100.0), np.array([10.0, 100, 100, 100, 100, 100]), abort_margin=5)
    robot.start_episode(0)
    stats = run_episode(robot, backend, spec, History(spec), task="t", episode=0, max_ticks=100, limits=limits)
    assert stats.stopped_by.startswith("runaway: shoulder_pan.pos commanded 16")
    assert stats.ticks == 15  # commands 1..15 sent (clamped at 10 from 11), 16 is > 10 + 5
    np.testing.assert_allclose(robot.sent[-1], robot.state)  # last goal: hold


def test_default_stall_guard_trips_under_the_default_per_tick_cap():
    """Step-500 run: goal held ~5 degrees ahead (the 5 degree cap) of a joint that doesn't move."""
    guard = StallGuard()
    state = np.zeros(6, np.float32)
    hits = [guard.update(state, state + np.array([0, 0, 5, 0, 0, 0], np.float32)) for _ in range(12)]
    assert hits[-1] is not None and hits[-1].startswith("elbow_flex.pos")


def test_joint_limits_from_dataset_stats(tmp_path):
    (tmp_path / "meta").mkdir()
    stats = {
        k: {"min": [-10, -20, -30, 40, -80, 0.5], "max": [5, 40, 90, 100, -60, 37]}
        for k in ("observation.state", "action")
    }
    (tmp_path / "meta/stats.json").write_text(__import__("json").dumps(stats))
    limits = JointLimits.from_dataset(str(tmp_path), margin=10)
    np.testing.assert_allclose(limits.low, [-20, -30, -40, 30, -90, 0])
    np.testing.assert_allclose(limits.high, [15, 50, 100, 110, -50, 47])


class ReplayRobot(FakeRobot):
    """A recording: the state never follows the goals, and each tick has a recorded command."""

    def observe(self):
        self.tick = getattr(self, "tick", -1) + 1
        return super().observe()

    def act(self, action):
        self.sent.append(action)
        return action

    def recorded_command(self):
        return np.full(6, 100.0 + self.tick, np.float32)


def test_teacher_forcing_anchors_each_plan_on_the_recorded_command():
    robot = ReplayRobot(ticks=70)
    backend = LocalBackend(RampModel())
    spec = backend.spec()
    robot.start_episode(0)
    run_episode(robot, backend, spec, History(spec), task="t", episode=0, max_ticks=1000, teacher_forcing=True)
    sent = np.stack(robot.sent)[:, 0]
    # plan 2 starts at tick 32 from the recorded command of tick 31 (131), not from the model's own chain (32)
    assert sent[31] == 32 and sent[32] == 132 and sent[64] == 100 + 63 + 1
    with pytest.raises(ValueError):
        run_episode(FakeRobot(5), backend, spec, History(spec), task="t", episode=0, max_ticks=5, teacher_forcing=True)


class BlockedRobot(FakeRobot):
    """Follows goals, except shoulder_lift can't pass 5 degrees (a table in the way)."""

    realtime = False

    def act(self, action):
        self.sent.append(action)
        self.state = action.astype(np.float32).copy()
        self.state[1] = min(self.state[1], 5.0)
        return action


class LiftModel(RampModel):
    def predict(self, window):
        steps = np.arange(1, 33, dtype=np.float32)[:, None] * np.eye(6, dtype=np.float32)[1]
        return Plan(window.commands[-1] + steps, {}, "lift")


def test_stall_guard_stops_a_blocked_arm_and_holds_position():
    robot = BlockedRobot(ticks=200)
    backend = LocalBackend(LiftModel())
    spec = backend.spec()
    robot.start_episode(0)
    stats = run_episode(
        robot, backend, spec, History(spec), task="t", episode=0, max_ticks=200, stall_guard=StallGuard()
    )
    assert stats.stopped_by.startswith("stall: shoulder_lift.pos")
    # blocked at 5 from the goal of tick 4; gap > 6 from tick 11, 10 still ticks later -> stop at tick ~21
    assert 15 <= stats.ticks <= 25
    np.testing.assert_allclose(robot.sent[-1][1], 5.0)  # last goal: hold where the arm is


def test_stall_guard_ignores_lag_while_moving():
    guard = StallGuard()
    goal = np.zeros(6, np.float32)
    for _ in range(50):  # the arm trails the goal by 12 degrees but keeps moving 3 degrees per tick
        goal = goal + np.array([0, 3, 0, 0, 0, 0], np.float32)
        assert guard.update(goal - np.array([0, 12, 0, 0, 0, 0], np.float32), goal) is None


def test_stall_guard_ignores_the_folded_arm_at_rest_but_not_mid_range():
    """ACT run: at rest the goal (-103) is below where the arm can go (-97, edge of the demos' reach)."""
    limits = JointLimits(
        np.full(6, -110.0), np.full(6, 110.0), reach_low=np.full(6, -97.0), reach_high=np.full(6, 100.0)
    )
    state, goal = np.full(6, -96.9, np.float32), np.full(6, -102.0, np.float32)
    guard = StallGuard()
    assert all(guard.update(state, goal, limits.beyond_reach(goal)) is None for _ in range(30))
    mid, push = np.zeros(6, np.float32), np.full(6, 5.0, np.float32)  # blocked mid-range (table)
    guard = StallGuard()
    hits = [guard.update(mid, push, limits.beyond_reach(push)) for _ in range(12)]
    assert hits[-1] is not None


def test_stall_guard_ignores_folding_further_at_rest():
    """19:19 run: folded arm (lift -96.9), policy tucks the elbow 87 -> 94; the elbow can't follow. Not a stall."""
    state = np.array([-5.8, -96.9, 87.0, 84.5, -85.8, 1.1], np.float32)
    tuck = state + np.array([0, 0, 5, 0, 0, 0], np.float32)
    guard = StallGuard(fold_lift=-97.05)
    assert all(guard.update(state, tuck) is None for _ in range(30))
    raised = state + np.array([0, 100, 0, 0, 0, 0], np.float32)  # same elbow push with the arm raised: stall
    guard = StallGuard(fold_lift=-97.05)
    assert [guard.update(raised, raised + np.array([0, 0, 5, 0, 0, 0], np.float32)) for _ in range(12)][-1]

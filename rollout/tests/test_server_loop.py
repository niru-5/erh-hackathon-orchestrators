"""Server + RemoteBackend over real HTTP, and the control loop, with the mock model (no GPU, no robot)."""

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
from rollout.loop import JointLimits, RunLog, run_episode
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


def test_limits_clip_goals_but_not_the_command_chain(tmp_path):
    robot = FakeRobot(ticks=40)
    backend = LocalBackend(RampModel())
    spec = backend.spec()
    limits = JointLimits(np.full(6, -100.0), np.array([10.0, 100, 100, 100, 100, 100]))
    robot.start_episode(0)
    stats = run_episode(robot, backend, spec, History(spec), task="t", episode=0, max_ticks=40, limits=limits)
    sent = np.stack(robot.sent)[:, 0]
    assert sent.max() == 10.0 and stats.clipped_ticks == 30


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

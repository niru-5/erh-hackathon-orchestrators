"""run --debug: the model's extra outputs over the wire, the recorder's files and the viewer (no GPU, no robot)."""

import json

import numpy as np
import pytest

from rollout.backends.local import LocalBackend
from rollout.codec import decode_plan, decode_window, encode_plan, encode_window
from rollout.contract import Observation, Plan, PolicySpec, Window
from rollout.debug import DebugRecorder
from rollout.history import History
from rollout.loop import RunLog, run_episode
from rollout.viewer import build, latest_run, load_run


class WorldModel:
    """Mock world model: holds the arm, except plan 2 starts 12 degrees off; imagines a brightening canvas."""

    def __init__(self):
        self._spec = PolicySpec(name="world", image_hw=(32, 48))
        self.plans = 0

    def spec(self):
        return self._spec

    def predict(self, window: Window) -> Plan:
        self.plans += 1
        actions = np.repeat(window.commands[-1][None], self._spec.n_action_steps, axis=0).astype(np.float32)
        if self.plans == 2:
            actions[:, 1] += 12.0
        debug = {}
        if window.debug:
            h, w = self._spec.image_hw
            frames = np.stack([np.full((h, 2 * w, 3), 6 * j, np.uint8) for j in range(41)])
            debug = {"chunk": np.concatenate([actions, actions[-10:]]), "future_frames": frames, "future_fps": 30.0}
        return Plan(actions, {"inference_ms": 1.0}, "world", debug)


class Robot:
    realtime = False

    def __init__(self, ticks):
        self.ticks, self.state = ticks, np.array([0, 0, 0, 50, 0, 10], np.float32)

    def connect(self):
        pass

    def start_episode(self, index):
        self.remaining = self.ticks

    def observe(self):
        if self.remaining == 0:
            return None
        self.remaining -= 1
        image = np.full((480, 640, 3), self.remaining % 255, np.uint8)
        return Observation(self.state.copy(), {"scene": image, "wrist": image})

    def act(self, action):
        self.state = action.astype(np.float32)
        return self.state.copy()


def test_debug_plan_round_trips_the_wire():
    window = Window("t", np.zeros((8, 6), np.float32), np.zeros((8, 6), np.float32), {}, debug=True)
    assert decode_window(json.loads(json.dumps(encode_window(window)))).debug
    assert not decode_window(encode_window(Window("t", window.states, window.commands, {}))).debug
    gradient = np.linspace(0, 250, 32, dtype=np.uint8)[None, :, None]
    frames = np.stack([np.broadcast_to(gradient, (16, 32, 3)) // (j + 1) for j in range(3)])
    plan = Plan(np.ones((32, 6), np.float32), {}, "m", {"chunk": np.ones((42, 6)), "future_frames": frames})
    back = decode_plan(json.loads(json.dumps(encode_plan(plan))))
    assert back.debug["chunk"].shape == (42, 6) and back.debug["future_frames"].shape == (3, 16, 32, 3)
    assert np.abs(back.debug["future_frames"].astype(int) - frames).mean() < 3  # JPEG
    assert decode_plan(encode_plan(Plan(plan.actions))).debug == {}


def test_debug_run_records_everything_and_builds_the_viewer(tmp_path):
    model = WorldModel()
    spec = model.spec()
    log = RunLog(tmp_path / "group" / "run")
    log.summary({"args": {"task": "move"}, "spec": spec.to_dict()})
    debug = DebugRecorder(log.directory, spec)
    robot = Robot(70)
    robot.start_episode(0)
    stats = run_episode(
        robot, LocalBackend(model), spec, History(spec), task="move", episode=0, max_ticks=100, log=log, debug=debug
    )
    log.summary({"args": {"task": "move"}, "spec": spec.to_dict(), "episodes": [{"episode": 0, **stats.to_dict()}]})
    log.close()
    debug.close()
    assert stats.plans == 3

    run = log.directory
    frames = sorted((run / "debug/frames/ep000/scene").iterdir())
    assert len(frames) == 70 and frames[0].name == "00000.jpg"
    import cv2

    assert cv2.imread(str(frames[0])).shape == (32, 48, 3)  # the model's input size
    plans = [json.loads(line) for line in (run / "debug/plans.jsonl").read_text().splitlines()]
    assert [p["tick"] for p in plans] == [0, 32, 64]
    assert len(plans[0]["chunk"]) == 42 and len(plans[0]["states"]) == 8
    assert all((run / f).exists() for f in plans[1]["future"]["frames"]) and len(plans[1]["future"]["frames"]) == 41
    assert (run / plans[0]["inputs"]["wrist"]["7"]).exists()

    data = load_run(run)
    first, second = data["episodes"][0]["plans"][:2]
    assert second["jump"]["deg"] == pytest.approx(12.0, abs=0.01) and second["jump"]["joint"] == 1
    assert "jump" not in first  # nothing was sent before the first plan
    assert any("plan 2 starts 12.0°" in f["text"] for f in data["episodes"][0]["findings"])
    assert latest_run(tmp_path) == run
    html = build(run).read_text()
    assert "__ROLLOUT_DATA__" not in html and '"run":"run"' in html


def test_flux_debug_imagines_future_frames_without_changing_actions():
    torch = pytest.importorskip("torch")
    pytest.importorskip("flux_action")
    import torch.nn.functional as F
    from flux_action.inference import sampling
    from test_flux_equivalence import FakeVideoVAE, observations, tiny_so101_policy

    from rollout.models.flux import FluxModel

    class DecodingVAE(FakeVideoVAE):
        def decode(self, latents):  # (1, 96, t, h, w) -> (1, 3, 1 + 4 (t - 1), 32 h, 32 w) in [-1, 1]
            x = torch.einsum("bcthw,co->bothw", latents.float(), self.proj.float())
            x = x.repeat_interleave(4, dim=2)[:, :, 3:]
            b, c, t, h, w = x.shape
            x = F.interpolate(x.transpose(1, 2).reshape(b * t, c, h, w), scale_factor=32)
            return torch.tanh(x.reshape(b, t, c, 32 * h, 32 * w).transpose(1, 2))

    policy = tiny_so101_policy()
    policy.frozen.video_vae = DecodingVAE()
    model = FluxModel(policy=policy, weights="tiny")
    history = History(model.spec())
    for obs in observations(9):
        history.append(obs, None if len(history) == 0 else obs.state)
    plain = model.predict(history.window("move"))
    window = history.window("move")
    window.debug = True
    solvers = (sampling.euler, sampling.cosmos_unipc_order2)
    traced = model.predict(window)

    assert (sampling.euler, sampling.cosmos_unipc_order2) == solvers  # restored
    np.testing.assert_array_equal(traced.actions, plain.actions)
    assert plain.debug == {}
    frames = traced.debug["future_frames"]
    assert frames.dtype == np.uint8 and frames.shape == (41, 64, 128, 3)  # latent_frames(42) = 11 -> 41 frames
    assert frames.std() > 0
    np.testing.assert_array_equal(traced.debug["chunk"][:32], traced.actions)
    assert traced.debug["chunk"].shape == (42, 6) and "future_decode_ms" in traced.timing

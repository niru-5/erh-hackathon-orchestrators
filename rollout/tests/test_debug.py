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


def test_flux_attention_records_the_conditional_action_rows_and_maps_them_to_the_cameras():
    """Stand-in for flux_action's prepared BF16 DiT (same interface): joint blocks whose ``attention`` module gets
    the normed, rotated q and k; guidance runs an empty-caption pass before each task pass."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    import torch.nn.functional as F

    from rollout.models.flux import FluxModel, _record_attention

    lengths = (3, 4, 2 * 2 * 4, 5, 2)  # text, imagined video, 2 snapshots of a 2x4 latent grid, actions, history
    heads, dim = 2, 8

    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)

    class Dit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.single_blocks = torch.nn.ModuleList(torch.nn.Module() for _ in range(3))
            for block in self.single_blocks:
                block.attention = Attention()

        def forward_prepared(self, request, step, *, video, action):
            q, k, v = request.qkv
            return torch.stack([block.attention(q, k, v) for block in self.single_blocks]).sum(0)

    def request(txt, seed):
        qkv = torch.randn(3, 1, heads, sum(lengths), dim, generator=torch.Generator().manual_seed(seed))
        names = ("txt_len", "video_len", "video_cond_len", "action_len", "action_cond_len")
        return SimpleNamespace(txt=txt, qkv=qkv, **dict(zip(names, lengths)))

    empty = torch.zeros(1)
    dit = Dit()
    policy = SimpleNamespace(dit=dit, _prepared_text_cache={"": SimpleNamespace(txt=empty)})
    negative, positive = request(empty, 0), request(torch.ones(1), 1)

    plain = [dit.forward_prepared(r, None, video=None, action=None) for r in (negative, positive)]
    store = {}
    with _record_attention(policy, store):
        traced = [dit.forward_prepared(r, None, video=None, action=None) for r in (negative, positive)]
    assert "forward_prepared" not in vars(dit) and all(not b.attention._forward_hooks for b in dit.single_blocks)
    for a, b in zip(plain, traced, strict=True):
        torch.testing.assert_close(a, b)  # the outputs are untouched
    assert store["passes"] == 3 and store["blocks"] == 3  # the task pass only, once per block

    start = sum(lengths[:3])
    q, k = positive.qkv[0], positive.qkv[1]
    expected = (q[:, :, start : start + lengths[3]] @ k.transpose(-1, -2) / dim**0.5).softmax(-1).mean(dim=(0, 1, 2))
    torch.testing.assert_close(store["sum"] / store["passes"], expected)

    model = FluxModel.__new__(FluxModel)
    model.policy = SimpleNamespace(config=SimpleNamespace(latent_hw=(2, 4)))
    model._spec = PolicySpec(name="flux", camera_keys=("scene", "wrist"), snapshot_indices=(0, 7))
    attention = model._attention_maps(store)
    history = expected.numpy()[7:23].reshape(2, 2, 4)
    peak = history.max()
    np.testing.assert_allclose(attention["maps"]["wrist"][7], np.round(history[1, :, 2:] / peak, 3))
    np.testing.assert_allclose(attention["maps"]["scene"][0], np.round(history[0, :, :2] / peak, 3))
    assert sum(attention["shares"].values()) == pytest.approx(1.0, abs=1e-3)
    assert attention["shares"]["task text"] == pytest.approx(float(expected[:3].sum()), abs=1e-4)
    assert "3 joint blocks" in attention["about"] and "1 task-conditioned" in attention["about"]


def test_act_attention_maps_its_cameras_and_crosses_the_wire():
    """The goal-conditioned ACT's cross-attention layout: [latent, state, goal, 3 cameras x (h w)]."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from rollout.models.lerobot_policy import LeRobotModel

    model = LeRobotModel.__new__(LeRobotModel)
    model.policy = SimpleNamespace(
        config=SimpleNamespace(robot_state_feature=True, env_state_feature=True, chunk_size=100)
    )
    model._spec = PolicySpec(name="act", camera_keys=("scene", "wrist", "extra"), image_hw=(64, 96))
    weights = torch.zeros(1, 100, 3 + 3 * 2 * 3)  # 64x96 images -> 2x3 feature maps
    weights[..., 0:3] = 0.1  # latent, state, goal
    weights[..., 3 + 6 + 4] = 0.7  # wrist, row 1, column 1
    attention = model._attention_maps([weights])
    assert attention["maps"]["wrist"][0] == [[0, 0, 0], [0, 1.0, 0]]
    assert attention["shares"] == pytest.approx(
        {"scene": 0.0, "wrist": 0.7, "extra": 0.0, "latent": 0.1, "state": 0.1, "goal": 0.1}
    )
    with pytest.raises(RuntimeError, match="don't split"):
        model._attention_maps([torch.zeros(1, 100, 20)])

    plan = Plan(np.zeros((32, 6), np.float32), {}, "act", {"attention": attention})
    assert decode_plan(json.loads(json.dumps(encode_plan(plan)))).debug["attention"]["shares"]["wrist"] == 0.7

"""The stateless chunk protocol gives exactly the actions of flux's own per-tick select_action loop.

Uses BFL's real FluxActionPolicy code with a tiny random DiT and a fake VAE on CPU (as in flux-action's
own tests/conftest.py), so it runs without a GPU or the 7B weights. Skipped when flux_action isn't installed:
    uv pip install --python .venv/bin/python --no-deps --target /tmp/fluxpkg \
        "flux-action @ git+https://github.com/black-forest-labs/flux-action@830f1ce178d7aea52f18fde31d8e8e36ce6bae21"
    PYTHONPATH=/tmp/fluxpkg .venv/bin/python -m pytest rollout/tests
"""

from collections import deque

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flux_action")

import torch.nn.functional as F  # noqa: E402
from flux_action.config import PolicyConfig  # noqa: E402
from flux_action.models.text_encoder import MockTextEncoder  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402

from rollout.contract import Observation  # noqa: E402
from rollout.history import History  # noqa: E402
from rollout.models.flux import FluxModel  # noqa: E402

TINY_DIT = {
    "hidden_size": 64,
    "num_heads": 2,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [8, 8, 8, 8],
    "context_in_dim": 32,
}


class FakeVideoVAE(torch.nn.Module):
    """Shape-only CPU stand-in for the video VAE (same as flux-action's test fixture)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("proj", torch.randn(96, 3, generator=torch.Generator().manual_seed(0)) * 0.1)

    def encode(self, video):
        b, c, t, h, w = video.shape
        x = video.float()[:, :, ::4]
        tp = x.shape[2]
        x = F.avg_pool2d(x.transpose(1, 2).reshape(-1, c, h, w), 32)
        x = torch.einsum("nchw,oc->nohw", x, self.proj.float())
        return x.reshape(b, tp, 96, h // 32, w // 32).transpose(1, 2)

    def encode_task(self, video):
        return self.encode(video)

    def encode_frame(self, frame):
        return self.encode(frame[:, :, None])


def tiny_so101_policy() -> FluxActionPolicy:
    """The SO-101 history profile (as load_policy builds it) with a tiny network."""
    torch.manual_seed(0)
    quantiles = {"q01": [-3.0, -2.0, -2.5, -1.0, -1.5, 0.0], "q99": [3.0, 2.0, 2.5, 1.0, 1.5, 40.0]}
    config = PolicyConfig(
        inference_profile="history",
        n_obs_steps=8,
        history_snapshots=2,
        condition_on_past_actions=True,
        chunk_size=42,
        n_action_steps=32,
        fps=30.0,
        action_dim=6,
        action_modality="action",
        camera_layout="side_by_side",
        camera_keys=("images.scene", "images.wrist"),
        canvas_hw=(64, 128),
        gripper_flip_dims=(),
        action_parameterization="joint_delta",
        absolute_action_dims=(5,),
        action_normalization=quantiles,
        state_normalization={
            "q01": [-90.0, -90.0, -30.0, 40.0, -90.0, 0.0],
            "q99": [10.0, 40.0, 95.0, 100.0, -60.0, 40.0],
        },
        single_frame_encode=True,
        dit_config=TINY_DIT,
        torch_dtype="float32",
        sampler="euler",
        num_inference_steps=2,
        guidance_scale=1.0,
        sampler_shift=5.0,
        augment=False,
    )
    policy = FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))
    return policy.eval()


def observations(n: int, seed: int = 0) -> list[Observation]:
    rng = np.random.default_rng(seed)
    state = np.array([-5.0, -25.0, 40.0, 77.0, -74.0, 8.0], np.float32)
    out = []
    for _ in range(n):
        state = state + rng.normal(0, 1.0, 6).astype(np.float32)
        images = {k: rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for k in ("scene", "wrist")}
        out.append(Observation(state=state.copy(), images=images))
    return out


def test_chunk_protocol_matches_select_action_loop():
    policy = tiny_so101_policy()
    ticks = observations(80)  # 3 plans: ticks 0, 32, 64
    task = "Move the figure from c3 at 6cm to D5 at 10 cm"

    # Reference: flux's own stateful loop, select_action at every tick.
    policy.reset()
    reference = []
    with torch.inference_mode():
        for obs in ticks:
            batch = {f"images.{k}": torch.from_numpy(v).permute(2, 0, 1)[None] for k, v in obs.images.items()}
            batch.update(state=torch.from_numpy(obs.state)[None], task=[task])
            reference.append(policy.select_action(batch)[0].numpy())

    # rollout: history on the robot side, a stateless predict once per plan (positions 1-6 never sent).
    model = FluxModel(policy=policy, weights="tiny")
    spec = model.spec()
    assert spec.snapshot_indices == (0, 7) and spec.image_hw == (64, 64) and spec.camera_keys == ("scene", "wrist")
    history = History(spec)
    queue, last, ours, plans = deque(), None, [], 0
    for obs in ticks:
        history.append(obs, last)
        if not queue:
            queue.extend(model.predict(history.window(task)).actions)
            plans += 1
        last = queue.popleft()
        ours.append(last)

    assert plans == 3
    np.testing.assert_allclose(np.stack(ours), np.stack(reference), rtol=0, atol=1e-4)
    # sanity: the model actually moves (the comparison isn't trivially between constants)
    assert np.abs(np.diff(np.stack(reference)[:, 0])).max() > 1e-3


def test_norm_stats_override_equals_policy_built_with_them(tmp_path):
    """--norm-stats gives the same plans as a package that shipped those quantiles."""
    import json

    ours = {
        "state": {"q01": [-14.0, -97.0, -24.0, 56.0, -105.0, 1.0], "q99": [4.0, 39.0, 96.0, 100.0, -62.0, 36.0]},
        "action": {"q01": [-0.8, -3.5, -2.5, -1.5, -0.8, 0.7], "q99": [0.6, 3.8, 3.8, 1.3, 0.7, 36.0]},
    }
    path = tmp_path / "ours.json"
    path.write_text(json.dumps(ours))

    def plan(model):
        history = History(model.spec())
        for obs in observations(8, seed=1):
            history.append(obs, obs.state + 0.5 if len(history) else None)
        return model.predict(history.window("Move the figure from c3 at 6cm to D5 at 10 cm")).actions

    built = tiny_so101_policy()
    built.config.state_normalization, built.config.action_normalization = ours["state"], ours["action"]
    expected = plan(FluxModel(policy=built, weights="tiny"))
    overridden = FluxModel(policy=tiny_so101_policy(), weights="tiny", norm_stats=str(path))
    np.testing.assert_allclose(plan(overridden), expected, rtol=0, atol=1e-5)
    assert overridden.spec().name == f"tiny+{path}"
    # the package's own quantiles give a different plan, so the override is what made them equal
    assert np.abs(plan(FluxModel(policy=tiny_so101_policy(), weights="tiny")) - expected).max() > 1e-3


def test_bundled_norm_stats_load():
    from rollout.models.flux import load_norm_stats

    stats = load_norm_stats("chess_c3_d5")
    assert stats["state"]["q01"][1] < -90 < stats["state"]["q99"][1]  # shoulder_lift in our calibration

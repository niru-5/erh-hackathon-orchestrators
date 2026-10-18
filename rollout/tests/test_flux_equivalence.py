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


def write_adapter(directory, tensors, r=4, alpha=8):
    import json

    from safetensors.torch import save_file

    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha}))
    save_file(tensors, str(directory / "adapter_model.safetensors"))


def test_lora_merge_matches_peft_formula(tmp_path):
    """LeRobot adapter names (base_model.model.dit.*) land on flux_action weights: W + alpha/r * B @ A."""
    from rollout.models.lora import merge_lora

    policy = tiny_so101_policy()
    target, head = "dit.content_mode_blocks.action.0.q_proj.weight", "dit.emb_in.action.weight"
    before = {k: policy.state_dict()[k].clone() for k in (target, head)}
    out_features, in_features = before[target].shape
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(4, in_features, generator=g), torch.randn(out_features, 4, generator=g)
    new_head = torch.randn(before[head].shape, generator=g)
    module = target.removesuffix(".weight")
    write_adapter(
        tmp_path / "adapter",
        {
            f"base_model.model.{module}.lora_A.weight": a,
            f"base_model.model.{module}.lora_B.weight": b,
            f"base_model.model.{head}": new_head,
        },
    )
    summary = merge_lora(policy, str(tmp_path / "adapter"))
    assert summary["lora_layers"] == 1 and summary["replaced"] == 1 and summary["scale"] == 2.0
    after = policy.state_dict()
    torch.testing.assert_close(after[target], before[target] + 2.0 * b @ a)
    torch.testing.assert_close(after[head], new_head)


def test_lora_merge_rejects_unknown_modules(tmp_path):
    from rollout.models.lora import merge_lora

    policy = tiny_so101_policy()
    write_adapter(
        tmp_path / "adapter",
        {
            "base_model.model.dit.no_such_layer.lora_A.weight": torch.zeros(4, 8),
            "base_model.model.dit.no_such_layer.lora_B.weight": torch.zeros(8, 4),
        },
    )
    with pytest.raises(KeyError):
        merge_lora(policy, str(tmp_path / "adapter"))


def test_dataset_normalization_replaces_checkpoint_quantiles():
    """FluxModel(normalization=...) changes what the model sees; the checkpoint's own quantiles change nothing."""
    policy = tiny_so101_policy()
    own = {"state": dict(policy.config.state_normalization), "action": dict(policy.config.action_normalization)}
    ticks = observations(9)

    def plan(model):
        history = History(model.spec())
        for obs in ticks:
            history.append(obs, None if len(history) == 0 else obs.state)
        return model.predict(history.window("move")).actions

    baseline = plan(FluxModel(policy=policy, weights="tiny"))
    np.testing.assert_allclose(plan(FluxModel(policy=policy, weights="tiny", normalization=own)), baseline, atol=1e-6)
    ours = {
        "state": {"q01": [-15, -97, -24, 55, -105, 1], "q99": [4, 39, 97, 100, 46, 36]},
        "action": {"q01": [-0.8, -3.5, -2.6, -1.5, -1.0, 0.6], "q99": [0.6, 3.8, 3.7, 1.3, 0.8, 36.0]},
        "source": "ours",
    }
    model = FluxModel(policy=policy, weights="tiny", normalization=ours)
    assert policy.config.state_normalization["q01"][1] == -97 and model.spec().name == "tiny [norm: ours]"
    assert np.abs(plan(model) - baseline).max() > 1e-3


def test_adapter_in_a_repo_subfolder():
    from rollout.models.lora import split_adapter

    assert split_adapter("Gawtam/flux3-chess-lora") == ("Gawtam/flux3-chess-lora", "")
    assert split_adapter("Gawtam/flux3-chess-lora-28moves/checkpoints/002000/pretrained_model_ema") == (
        "Gawtam/flux3-chess-lora-28moves",
        "checkpoints/002000/pretrained_model_ema",
    )


def test_fewer_action_steps_per_plan_execute_the_same_chunk_prefix():
    policy = tiny_so101_policy()
    ticks = observations(9)

    def plan(model):
        history = History(model.spec())
        for obs in ticks:
            history.append(obs, None if len(history) == 0 else obs.state)
        return model.predict(history.window("move")).actions

    full = plan(FluxModel(policy=policy, weights="tiny"))
    model = FluxModel(policy=policy, weights="tiny", n_action_steps=10)
    assert model.spec().n_action_steps == 10 and model.spec().name == "tiny [10 steps/plan]"
    np.testing.assert_array_equal(plan(model), full[:10])
    with pytest.raises(ValueError):
        FluxModel(policy=policy, weights="tiny", n_action_steps=43)

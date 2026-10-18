"""FLUX 3 Action SO-101 (black-forest-labs/flux-3-action-so101) through BFL's ``flux_action`` package.

Needs a CUDA GPU with about 32 GB and the flux-action environment (torch 2.10 + cu128, transformers,
NATTEN), see rollout/server/Dockerfile. Only imported when a flux model is requested.
"""

import time
from contextlib import contextmanager

import numpy as np
import torch

from ..contract import ACTION_NAMES, Plan, PolicySpec, Window

DEFAULT_WEIGHTS = "black-forest-labs/flux-3-action-so101"


class FluxModel:
    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        *,
        revision: str | None = None,
        device: str = "cuda",
        compile_dit: bool = False,
        offload_text_encoder: bool = False,
        adapter: str | None = None,  # LeRobot PEFT LoRA merged into the weights, e.g. Gawtam/flux3-chess-lora
        adapter_revision: str | None = None,
        normalization: dict | str | None = None,  # our dataset's quantiles instead of the checkpoint's, see
        # rollout/normalization.py (a dict, JSON string/file or dataset id)
        policy=None,  # an already loaded FluxActionPolicy (tests)
    ):
        from flux_action.processing.history import snapshot_indices

        if policy is None:
            from flux_action.inference.precision import prepare_for_serving
            from flux_action.inference.so101 import load_policy

            policy = load_policy(weights, revision=revision)
            if adapter:
                from .lora import merge_lora

                merge_lora(policy, adapter, adapter_revision)  # on CPU, before the GPU/serving preparation
            prepare_for_serving(
                policy, device=device, compile_dit=compile_dit, offload_text_encoder=offload_text_encoder
            )
        self.policy = policy
        cfg = policy.config
        norm_source = None
        if normalization is not None:
            from ..normalization import load

            stats = load(normalization)
            for key in ("state", "action"):
                if any(len(stats[key][q]) != cfg.action_dim for q in ("q01", "q99")):
                    raise ValueError(f"normalization {key} quantiles must have {cfg.action_dim} values")
            # read on every call by flux_action (normalization.normalize/denormalize), nothing is cached
            cfg.state_normalization = {q: list(map(float, stats["state"][q])) for q in ("q01", "q99")}
            cfg.action_normalization = {q: list(map(float, stats["action"][q])) for q in ("q01", "q99")}
            norm_source = stats.get("source", "custom")
        if cfg.inference_profile != "history" or cfg.camera_layout != "side_by_side":
            raise ValueError("FluxModel supports the SO-101 history package (side_by_side cameras)")
        canvas_h, canvas_w = cfg.canvas_hw
        self._keys = tuple(cfg.camera_keys)  # "images.scene", "images.wrist"
        self._spec = PolicySpec(
            name=(weights if isinstance(weights, str) else "flux")
            + (f"+{adapter}" if adapter else "")
            + (f" [norm: {norm_source}]" if norm_source else ""),
            camera_keys=tuple(key.removeprefix("images.") for key in self._keys),
            n_obs_steps=cfg.n_obs_steps,
            snapshot_indices=tuple(snapshot_indices(cfg)),
            chunk_size=cfg.chunk_size,
            n_action_steps=cfg.n_action_steps,
            fps=float(cfg.fps),
            image_hw=(canvas_h, canvas_w // 2),
            action_names=ACTION_NAMES,
        )

    def spec(self) -> PolicySpec:
        return self._spec

    def predict(self, window: Window) -> Plan:
        """The explicit-window form of ``predict_action_chunk`` (flux docs: "offline history window"):
        equal to what ``select_action`` would execute after this window, see docs/inference.md."""
        spec, device = self._spec, self.policy.device
        n = spec.n_obs_steps
        if window.states.shape != (n, len(spec.action_names)) or window.commands.shape != window.states.shape:
            raise ValueError(f"window needs {n} states and commands")
        started = time.perf_counter()
        batch = {
            "task": [window.task],
            "state": torch.from_numpy(window.states).float()[None].to(device),
            "command_history": torch.from_numpy(window.commands).float()[None].to(device),
        }
        frames = {camera: self._fill(window.images[camera]) for camera in spec.camera_keys}
        if len({f.shape for f in frames.values()}) != 1:  # the side_by_side layout needs equal camera sizes
            frames = {c: self._resize(f, spec.image_hw) for c, f in frames.items()}
        for camera, key in zip(spec.camera_keys, self._keys, strict=True):
            # uint8 (1, n, 3, H, W): flux normalizes uint8 on CPU, as its reference loader expects.
            batch[key] = torch.from_numpy(np.ascontiguousarray(frames[camera].transpose(0, 3, 1, 2)))[None].to(device)
        solver_out: dict = {}
        with torch.inference_mode(), _keep_solver_output(solver_out, enabled=window.debug):
            chunk = self.policy.predict_action_chunk(batch)[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing = {"inference_ms": (time.perf_counter() - started) * 1e3}
        chunk = chunk.float().cpu().numpy()
        actions = chunk[: spec.n_action_steps]
        if not np.isfinite(actions).all():
            raise RuntimeError("policy returned non-finite actions")
        debug = {}
        if window.debug:
            decode_started = time.perf_counter()
            future = self._decode_future(solver_out["x_video"])
            debug = {"chunk": chunk, "future_frames": future, "future_fps": spec.fps}
            timing["future_decode_ms"] = (time.perf_counter() - decode_started) * 1e3
        return Plan(actions, timing, spec.name, debug)

    def _decode_future(self, tokens: torch.Tensor) -> np.ndarray:
        """The video tokens the sampler denoised alongside the actions -> (T, H, W, 3) uint8 canvas frames.

        flux_action encodes the window's future frames as one clip of their own (history.encode_windows), so the
        predicted latents decode on their own: frame j is the observation after action j of the chunk.
        """
        from einops import rearrange
        from flux_action.processing import packing

        cfg, device = self.policy.config, self.policy.device
        (h, w), t = cfg.latent_hw, packing.latent_frames(cfg.chunk_size)
        latents = rearrange(tokens[0].float(), "(t h w) c -> c t h w", t=t, h=h, w=w)[None].to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            video = self.policy.video_vae.decode(latents)  # (1, 3, T, H, W) in [-1, 1]
        content_h, content_w = packing.content_hw(cfg.camera_layout, cfg.canvas_hw)
        video = video[0, :, :, :content_h, :content_w].float().clamp(-1, 1)
        return ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()

    def _fill(self, snapshots: dict[int, np.ndarray]) -> np.ndarray:
        """(n, H, W, 3): the model reads only the snapshot positions; fill the rest with the nearest one."""
        missing = set(self._spec.snapshot_indices) - set(snapshots)
        if missing:
            raise ValueError(f"window lacks snapshot images at positions {sorted(missing)}")
        known = sorted(snapshots)
        return np.stack([snapshots[min(known, key=lambda k: abs(k - i))] for i in range(self._spec.n_obs_steps)])

    @staticmethod
    def _resize(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        import cv2

        return np.stack([cv2.resize(f, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for f in frames])


@contextmanager
def _keep_solver_output(store: dict, enabled: bool = True):
    """Copy what flux_action's ODE solver returns into ``store``: ``FluxActionPolicy._sample`` keeps only the
    action stream and drops the video stream it denoised with it. Wraps the solver functions for the duration;
    their results are passed through untouched, so the actions don't change."""
    if not enabled:
        yield
        return
    from flux_action.inference import sampling

    originals = {name: getattr(sampling, name) for name in ("cosmos_unipc_order2", "euler")}

    def keeping(solver):
        def solve(*args, **kwargs):
            out = solver(*args, **kwargs)
            store.update(out)
            return out

        return solve

    for name, solver in originals.items():
        setattr(sampling, name, keeping(solver))
    try:
        yield
    finally:
        for name, solver in originals.items():
            setattr(sampling, name, solver)

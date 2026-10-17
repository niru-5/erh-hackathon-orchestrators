"""FLUX 3 Action SO-101 (black-forest-labs/flux-3-action-so101) through BFL's ``flux_action`` package.

Needs a CUDA GPU with about 32 GB and the flux-action environment (torch 2.10 + cu128, transformers,
NATTEN), see rollout/server/Dockerfile. Only imported when a flux model is requested.
"""

import time

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
        policy=None,  # an already loaded FluxActionPolicy (tests)
    ):
        from flux_action.processing.history import snapshot_indices

        if policy is None:
            from flux_action.inference.precision import prepare_for_serving
            from flux_action.inference.so101 import load_policy

            policy = load_policy(weights, revision=revision)
            prepare_for_serving(
                policy, device=device, compile_dit=compile_dit, offload_text_encoder=offload_text_encoder
            )
        self.policy = policy
        cfg = policy.config
        if cfg.inference_profile != "history" or cfg.camera_layout != "side_by_side":
            raise ValueError("FluxModel supports the SO-101 history package (side_by_side cameras)")
        canvas_h, canvas_w = cfg.canvas_hw
        self._keys = tuple(cfg.camera_keys)  # "images.scene", "images.wrist"
        self._spec = PolicySpec(
            name=weights if isinstance(weights, str) else "flux",
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
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)[0, : spec.n_action_steps]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        actions = actions.float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise RuntimeError("policy returned non-finite actions")
        return Plan(actions, {"inference_ms": (time.perf_counter() - started) * 1e3}, spec.name)

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

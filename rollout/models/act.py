"""lerobot ACT checkpoints (e.g. rtrtwrw/act_chess_c3_to_d5_step2000) with their saved pre/post processors.

ACT reads one tick (n_obs_steps=1): the measured state and a full-resolution frame per camera. It is a
ResNet18 + small transformer, so it runs on a CPU too. Only imported when an act model is requested.
"""

import time

import numpy as np
import torch

from ..contract import ACTION_NAMES, Plan, PolicySpec, Window


class ACTModel:
    def __init__(
        self,
        weights: str,
        *,
        revision: str | None = None,
        device: str = "cuda",
        n_action_steps: int | None = None,
        fps: float = 30.0,
    ):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(weights, revision=revision)
        self.policy.config.device = device
        self.policy.to(self.device).eval()
        cfg = self.policy.config
        self.preprocess, self.postprocess = make_pre_post_processors(
            cfg,
            pretrained_path=weights,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        if cfg.n_obs_steps != 1:
            raise ValueError("ACTModel supports n_obs_steps=1 checkpoints")
        cameras = [key.removeprefix("observation.images.") for key in cfg.image_features]
        heights_widths = {tuple(cfg.input_features[key].shape[1:]) for key in cfg.image_features}
        if len(heights_widths) != 1:
            raise ValueError(f"ACTModel needs equal camera sizes, got {heights_widths}")
        self._spec = PolicySpec(
            name=weights,
            camera_keys=tuple(cameras),
            n_obs_steps=1,
            snapshot_indices=(0,),
            chunk_size=cfg.chunk_size,
            n_action_steps=min(n_action_steps or cfg.n_action_steps, cfg.chunk_size),
            fps=fps,
            image_hw=heights_widths.pop(),
            action_names=ACTION_NAMES,
        )

    def spec(self) -> PolicySpec:
        return self._spec

    def predict(self, window: Window) -> Plan:
        spec = self._spec
        started = time.perf_counter()
        observation = {"observation.state": torch.from_numpy(window.states[-1]).float()}
        for camera in spec.camera_keys:
            # float CHW in [0, 1], as LeRobotDataset yields frames during training.
            frame = torch.from_numpy(np.ascontiguousarray(window.images[camera][0])).permute(2, 0, 1)
            observation[f"observation.images.{camera}"] = frame.float() / 255.0
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocess(observation))
            actions = self.postprocess(chunk)[0, : spec.n_action_steps]
        actions = actions.float().cpu().numpy()
        if not np.isfinite(actions).all():
            raise RuntimeError("policy returned non-finite actions")
        return Plan(actions, {"inference_ms": (time.perf_counter() - started) * 1e3}, spec.name)

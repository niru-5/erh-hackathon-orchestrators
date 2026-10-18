"""LeRobot policies that act on one observation and predict a chunk of absolute actions (e.g. ACT).

Small enough to run in-process on the recording laptop's CPU (rtrtwrw/act_chess_c3_to_d5: 51.6M params,
~0.6 s per 100-action chunk). The checkpoint owns the contract: cameras and resolution from its
``input_features``, chunk length and actions per plan from its config, normalization from its saved
processors. This is what LeRobot's own ``select_action`` does, minus the queue (rollout.loop keeps it).

Goal-conditioned chess checkpoints (rtrtwrw/act_chess_goalcond_*) also take
``observation.environment_state``: the move, encoded from the two squares in the task as their
``chess_goal.py`` does (``from_x, from_y, to_x, to_y, dx, dy``, squares mapped to [-1, 1]).
"""

import re
import time

import numpy as np
import torch

from ..contract import ACTION_NAMES, Plan, PolicySpec, Window

IMAGE_PREFIX = "observation.images."
GOAL_KEY = "observation.environment_state"


def chess_goal(task: str) -> np.ndarray:
    """The 6-value move encoding of rtrtwrw's chess_goal.py, from a task naming exactly two squares."""
    squares = re.findall(r"\b([a-h][1-8])\b", task.lower())
    if len(squares) != 2:
        raise ValueError(f"the task must name exactly two squares (from, to), got {squares}: {task!r}")
    (fx, fy), (tx, ty) = ((("abcdefgh".index(s[0])) / 7 * 2 - 1, (int(s[1]) - 1) / 7 * 2 - 1) for s in squares)
    return np.array([fx, fy, tx, ty, (tx - fx) / 2, (ty - fy) / 2], dtype=np.float32)


class LeRobotModel:
    def __init__(
        self,
        weights: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        n_action_steps: int | None = None,
        fps: float = 30.0,
    ):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config = PreTrainedConfig.from_pretrained(weights, revision=revision)
        config.device = device
        if config.n_obs_steps != 1:
            raise ValueError(
                f"{weights}: only single-observation policies are supported (n_obs_steps={config.n_obs_steps})"
            )
        self.policy = get_policy_class(config.type).from_pretrained(weights, config=config, revision=revision)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=weights,
            pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        images = [key for key in config.input_features if key.startswith(IMAGE_PREFIX)]
        shapes = {tuple(config.input_features[key].shape) for key in images}
        if len(shapes) != 1:
            raise ValueError(f"{weights}: cameras must share one resolution, got {shapes}")
        _, height, width = shapes.pop()
        chunk = getattr(config, "chunk_size", None) or config.n_action_steps
        steps = n_action_steps or config.n_action_steps
        if not 1 <= steps <= chunk:
            raise ValueError(f"n_action_steps must be within 1..{chunk}")
        self.device = device
        self.goal_conditioned = GOAL_KEY in config.input_features
        self._spec = PolicySpec(
            name=weights,
            camera_keys=tuple(key.removeprefix(IMAGE_PREFIX) for key in images),
            n_obs_steps=1,
            snapshot_indices=(0,),
            chunk_size=chunk,
            n_action_steps=steps,
            fps=fps,
            image_hw=(height, width),
            action_names=ACTION_NAMES,
        )

    def spec(self) -> PolicySpec:
        return self._spec

    def predict(self, window: Window) -> Plan:
        started = time.perf_counter()
        observation = {"observation.state": torch.from_numpy(window.states[-1]).float(), "task": window.task}
        for camera in self._spec.camera_keys:
            image = window.images[camera][0]  # RGB uint8 HWC -> float CHW in [0, 1], as in the dataset
            observation[IMAGE_PREFIX + camera] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        if self.goal_conditioned:
            observation[GOAL_KEY] = torch.from_numpy(chess_goal(window.task))
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocessor(observation))
            actions = self.postprocessor(chunk)[0, : self._spec.n_action_steps]
        actions = actions.float().cpu().numpy()
        if actions.shape != (self._spec.n_action_steps, len(ACTION_NAMES)) or not np.isfinite(actions).all():
            raise RuntimeError(f"policy returned invalid actions of shape {actions.shape}")
        return Plan(actions, {"inference_ms": (time.perf_counter() - started) * 1e3}, self._spec.name)

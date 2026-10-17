"""Rolling observation history kept on the robot side.

Mirrors ``flux_action.processing.history.ObservationHistory``: every tick appends the measured
observation and the command issued before it (the measured state on an episode's first tick), and the
first tick of an episode fills the whole window with copies of itself. Because the robot side keeps
this, the model server can be stateless and is called once per plan instead of once per tick.
"""

from collections import deque

import cv2
import numpy as np

from .contract import Observation, PolicySpec, Window


class History:
    def __init__(self, spec: PolicySpec, *, resize_images: bool = True):
        self.spec = spec
        self.resize_images = resize_images
        self._observations: deque[Observation] = deque(maxlen=spec.n_obs_steps)
        self._commands: deque[np.ndarray] = deque(maxlen=spec.n_obs_steps)

    def reset(self) -> None:
        self._observations.clear()
        self._commands.clear()

    def __len__(self) -> int:
        return len(self._observations)

    def append(self, observation: Observation, last_command: np.ndarray | None) -> None:
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (len(self.spec.action_names),):
            raise ValueError(f"expected state of shape ({len(self.spec.action_names)},), got {state.shape}")
        missing = [key for key in self.spec.camera_keys if key not in observation.images]
        if missing:
            raise KeyError(f"observation lacks camera(s) {missing}; the model needs {list(self.spec.camera_keys)}")
        command = state if last_command is None else np.asarray(last_command, dtype=np.float32)
        observation = Observation(state=state, images={k: observation.images[k] for k in self.spec.camera_keys})
        repeats = self.spec.n_obs_steps if not self._observations else 1
        for _ in range(repeats):
            self._observations.append(observation)
            self._commands.append(command.copy())

    def window(self, task: str) -> Window:
        if len(self._observations) != self.spec.n_obs_steps:
            raise RuntimeError("history is empty; append an observation first")
        images = {
            key: {i: self._prepare(self._observations[i].images[key]) for i in self.spec.snapshot_indices}
            for key in self.spec.camera_keys
        }
        return Window(
            task=task,
            states=np.stack([o.state for o in self._observations]),
            commands=np.stack(list(self._commands)),
            images=images,
        )

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("cameras must be RGB uint8 HWC frames")
        h, w = self.spec.image_hw
        if not self.resize_images or image.shape[:2] == (h, w):
            return image
        # The model squashes each camera to image_hw (bilinear, antialiased); INTER_AREA is the cv2 match.
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

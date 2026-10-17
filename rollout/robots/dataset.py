"""Replay a recorded LeRobotDataset episode as observations (open-loop evaluation, no hardware).

Observations come from the recording, not from the policy's actions, so this checks the pipeline and
how close the model's commands are to the demonstrator's, not whether it would succeed.
"""

import numpy as np

from ..contract import ACTION_NAMES, Observation


class DatasetRobot:
    realtime = False

    def __init__(self, repo_id: str, camera_keys: tuple[str, ...], *, root: str | None = None):
        self.repo_id = repo_id
        self.root = root
        self.camera_keys = camera_keys
        self._dataset = None
        self._episode = None
        self._episodes: dict[int, dict] = {}

    def connect(self) -> None:
        pass

    def start_episode(self, index: int) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # pyav bundles its own FFmpeg; torchcodec needs system FFmpeg libraries this laptop doesn't have.
        self._dataset = LeRobotDataset(self.repo_id, root=self.root, episodes=[index], video_backend="pyav")
        names = self._dataset.meta.features["action"]["names"]
        if tuple(names) != ACTION_NAMES:
            raise ValueError(f"dataset joints {names} differ from {ACTION_NAMES}")
        self._episode = index
        self._frame = 0
        self._predicted: list[np.ndarray] = []
        self._recorded: list[np.ndarray] = []

    def observe(self) -> Observation | None:
        if self._frame >= len(self._dataset):
            return None
        item = self._dataset[self._frame]
        self._frame += 1
        self._current_action = item["action"].numpy().astype(np.float32)
        images = {
            key: (item[f"observation.images.{key}"].permute(1, 2, 0) * 255).round().clamp(0, 255).byte().numpy()
            for key in self.camera_keys
        }
        return Observation(state=item["observation.state"].numpy().astype(np.float32), images=images)

    def recorded_command(self) -> np.ndarray:
        """The demonstrator's command at the current tick (for teacher-forced evaluation)."""
        return self._current_action

    def act(self, action: np.ndarray) -> np.ndarray:
        self._predicted.append(np.asarray(action, dtype=np.float32))
        self._recorded.append(self._current_action)
        return action

    def summary(self) -> dict:
        """Per-joint mean absolute error between predicted commands and the recorded actions."""
        if not self._predicted:
            return {}
        error = np.abs(np.stack(self._predicted) - np.stack(self._recorded))

        def per_joint(values: np.ndarray) -> dict:
            return {name: round(float(v), 2) for name, v in zip(ACTION_NAMES, values, strict=True)}

        result = {
            "episode": self._episode,
            "frames": len(error),
            "mae_per_joint": per_joint(error.mean(0)),
            "mae_first_plan": per_joint(error[:32].mean(0)),
        }
        self._episodes[self._episode] = result
        return result

    def disconnect(self) -> None:
        pass

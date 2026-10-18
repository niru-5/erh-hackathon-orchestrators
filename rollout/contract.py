"""Types shared by the robot side, the backends and the models (numpy only, no torch)."""

from dataclasses import asdict, dataclass, field

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTION_NAMES = tuple(f"{joint}.pos" for joint in JOINT_NAMES)


@dataclass(frozen=True)
class PolicySpec:
    """What a model needs from the robot side. The model owns it; the client sizes its history from it."""

    name: str
    camera_keys: tuple[str, ...] = ("scene", "wrist")
    n_obs_steps: int = 8
    snapshot_indices: tuple[int, ...] = (0, 7)  # window positions whose images the model reads
    chunk_size: int = 42
    n_action_steps: int = 32
    fps: float = 30.0
    image_hw: tuple[int, int] = (256, 256)  # each camera is resized to this by the model
    action_names: tuple[str, ...] = ACTION_NAMES

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PolicySpec":
        tuples = ("camera_keys", "snapshot_indices", "image_hw", "action_names")
        return cls(**{k: tuple(v) if k in tuples else v for k, v in data.items()})


@dataclass
class Observation:
    """One tick: measured joint positions and one RGB uint8 HWC frame per camera."""

    state: np.ndarray  # (6,) float32, degrees / gripper %
    images: dict[str, np.ndarray]


@dataclass
class Window:
    """The model input for one plan: the last n_obs_steps ticks, oldest first."""

    task: str
    states: np.ndarray  # (n_obs_steps, 6) measured states
    commands: np.ndarray  # (n_obs_steps, 6) command issued before each state
    images: dict[str, dict[int, np.ndarray]]  # camera -> window position -> RGB uint8 HWC (snapshots only)
    debug: bool = False  # ask the model for Plan.debug (run --debug)


@dataclass
class Plan:
    """The model output: absolute joint targets to execute in order, one per tick."""

    actions: np.ndarray  # (n_action_steps, 6) float32
    timing: dict[str, float] = field(default_factory=dict)
    model: str = ""
    # Only when Window.debug asked for it, all optional: "chunk" (chunk_size, 6) every predicted action, executed
    # or not; "future_frames" (T, H, W, 3) uint8 the frames a world model imagines after this window (FLUX: the
    # side_by_side canvas, frame j = the observation after action j); "future_fps"; "future_error" if decoding failed;
    # "attention" where the action tokens attend in each input image (rollout/attention.py), "attention_error".
    debug: dict = field(default_factory=dict)

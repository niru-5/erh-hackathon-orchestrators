"""The real SO-101 follower arm and its cameras, through lerobot (same rig defaults as scripts/record_so101.sh).

Only the follower is used: the policy replaces the leader arm. Override devices with the same env vars
as the recording script (FOLLOWER_PORT, FOLLOWER_ID, CAM_SCENE, CAM_WRIST, CAM_EXTRA, CAM_WIDTH, CAM_HEIGHT,
CAM_FOURCC).
"""

import logging
import os
from pathlib import Path

import numpy as np

from ..contract import ACTION_NAMES, Observation

logger = logging.getLogger(__name__)

FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115116-if00"
CAMERAS = {  # name -> default device; the env var is CAM_<NAME>
    "scene": "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0",
    "extra": "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_2ED4DF6F-video-index0",
}


class SO101Robot:
    realtime = True

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        *,
        fps: int = 30,
        max_relative_target: float | None = 10.0,
        dry_run: bool = False,
    ):
        from lerobot.cameras.configs import Cv2Backends
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        unknown = set(camera_keys) - set(CAMERAS)
        if unknown:
            raise ValueError(f"the model wants cameras {sorted(unknown)}, this rig has {sorted(CAMERAS)}")
        port = os.environ.get("FOLLOWER_PORT", FOLLOWER_PORT)
        devices = {key: os.environ.get(f"CAM_{key.upper()}", CAMERAS[key]) for key in camera_keys}
        for device in (port, *devices.values()):
            if not Path(device).exists():
                raise FileNotFoundError(f"device not found: {device}")
        cameras = {
            key: OpenCVCameraConfig(
                index_or_path=Path(os.path.realpath(device)),  # OpenCV wants the real /dev/videoN node
                width=int(os.environ.get("CAM_WIDTH", 640)),
                height=int(os.environ.get("CAM_HEIGHT", 480)),
                fps=fps,
                backend=Cv2Backends.V4L2,
                fourcc=os.environ.get("CAM_FOURCC", "MJPG"),  # the wrist cam needs MJPG for 30 fps
            )
            for key, device in devices.items()
        }
        config = SO101FollowerConfig(
            port=port,
            id=os.environ.get("FOLLOWER_ID", "so101"),
            cameras=cameras,
            max_relative_target=max_relative_target,
        )
        self.camera_keys = camera_keys
        self.dry_run = dry_run
        self.robot = SO101Follower(config)

    def connect(self) -> None:
        # calibrate=False never starts an interactive calibration; a mismatch raises instead.
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            raise RuntimeError("follower calibration doesn't match the motors; see docs/data_collection.md")

    def start_episode(self, index: int) -> None:
        pass

    def observe(self) -> Observation | None:
        obs = self.robot.get_observation()
        state = np.array([obs[name] for name in ACTION_NAMES], dtype=np.float32)
        return Observation(state=state, images={key: obs[key] for key in self.camera_keys})

    def act(self, action: np.ndarray) -> np.ndarray:
        if self.dry_run:
            return action
        goal = {name: float(value) for name, value in zip(ACTION_NAMES, action, strict=True)}
        sent = self.robot.send_action(goal)
        return np.array([sent[name] for name in ACTION_NAMES], dtype=np.float32)

    def summary(self) -> dict:
        return {}

    def disconnect(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()  # disables torque (disable_torque_on_disconnect)

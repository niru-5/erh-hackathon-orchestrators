"""Grab one frame from each recording camera and save it as a JPEG, to check the views before recording.

Usage: .venv/bin/python scripts/check_cameras.py [output_dir]   (default: ./camera_check)
"""

import os
import sys
from pathlib import Path

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

CAMERAS = {
    "left_side": os.environ.get("CAM_LEFT", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_2ED4DF6F-video-index0"),
    "right_side": os.environ.get("CAM_RIGHT", "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"),
    "ff": os.environ.get("CAM_FF", "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0"),
}

out = Path(sys.argv[1] if len(sys.argv) > 1 else "camera_check")
out.mkdir(parents=True, exist_ok=True)
for name, path in CAMERAS.items():
    path = os.path.realpath(path)  # OpenCV wants the real /dev/videoN node
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=path, width=640, height=480, fps=30))
    cam.connect()
    try:
        frame = cam.read()
        cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"{name}: {path} {frame.shape} -> {out / f'{name}.jpg'}")
    finally:
        cam.disconnect()

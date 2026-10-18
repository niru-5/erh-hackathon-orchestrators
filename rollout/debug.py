"""run --debug: everything needed to replay a rollout afterwards, under <run dir>/debug/ (see docs/inference.md).

    frames/ep000/<camera>/00042.jpg          every tick's camera frame, at the model's input size
    plans/ep000_0003/<camera>_<pos>.jpg      the snapshot images the model got for plan 3
    plans/ep000_0003/future/000.jpg          FLUX: the frames it imagined (frame j = after action j)
    plans.jsonl                              per plan: tick, task, state/command history, executed actions,
                                             the full predicted chunk, timing, the image paths above and
                                             where the action tokens attended in each input image
                                             (FLUX and ACT, rollout/attention.py)

Images are written on a background thread so the control loop keeps its tick rate.
``python -m rollout view <run dir>`` turns the run into viewer.html.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .contract import Observation, Plan, PolicySpec, Window

logger = logging.getLogger(__name__)


class DebugRecorder:
    def __init__(self, run_directory: Path, spec: PolicySpec, jpeg_quality: int = 85):
        self.directory = run_directory / "debug"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.spec, self.jpeg_quality = spec, jpeg_quality
        self._plans = (self.directory / "plans.jsonl").open("w")
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="debug-writer")
        self._errors = 0

    def tick(self, episode: int, tick: int, observation: Observation) -> None:
        for camera in self.spec.camera_keys:
            self._write(f"frames/ep{episode:03d}/{camera}/{tick:05d}.jpg", observation.images[camera], resize=True)

    def plan(self, episode: int, number: int, tick: int, window: Window, plan: Plan) -> None:
        base = f"plans/ep{episode:03d}_{number:04d}"
        inputs = {}
        for camera, frames in window.images.items():
            inputs[camera] = {}
            for position, image in frames.items():
                inputs[camera][position] = self._write(f"{base}/{camera}_{position}.jpg", image)
        record = {
            "episode": episode,
            "plan": number,
            "tick": tick,
            "model": plan.model,
            "task": window.task,
            "states": window.states,
            "commands": window.commands,
            "actions": plan.actions,
            "chunk": plan.debug.get("chunk"),
            "timing": plan.timing,
            "inputs": inputs,
        }
        frames = plan.debug.get("future_frames")
        if frames is not None:
            record["future"] = {
                "frames": [self._write(f"{base}/future/{j:03d}.jpg", frame) for j, frame in enumerate(frames)],
                "fps": plan.debug.get("future_fps", self.spec.fps),
            }
        if "future_error" in plan.debug:
            record["future_error"] = plan.debug["future_error"]
            logger.warning("plan %d: no imagined frames: %s", number, plan.debug["future_error"])
        if "attention" in plan.debug:
            record["attention"] = plan.debug["attention"]
        if "attention_error" in plan.debug:
            record["attention_error"] = plan.debug["attention_error"]
            logger.warning("plan %d: no attention maps: %s", number, plan.debug["attention_error"])
        self._plans.write(json.dumps(_jsonable(record)) + "\n")
        self._plans.flush()

    def close(self) -> None:
        self._writer.shutdown(wait=True)
        self._plans.close()
        if self._errors:
            logger.warning("debug: %d images could not be written", self._errors)

    def _write(self, relative: str, image: np.ndarray, resize: bool = False) -> str:
        """Queue one RGB frame as a JPEG; returns its path relative to the run directory."""
        path = self.directory / relative

        def write():
            try:
                frame = image
                h, w = self.spec.image_hw
                if resize and frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)  # what the model sees
                path.parent.mkdir(parents=True, exist_ok=True)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                    raise OSError(f"cv2.imwrite failed for {path}")
            except Exception:  # never let a debug image stop the arm
                self._errors += 1
                logger.exception("debug: writing %s failed", path)

        self._writer.submit(write)
        return str(Path("debug") / relative)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return np.round(value.astype(float), 3).tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

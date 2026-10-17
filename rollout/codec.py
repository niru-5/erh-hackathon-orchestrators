"""JSON wire format for windows and plans; images travel as base64 JPEG."""

import base64

import cv2
import numpy as np

from .contract import Plan, Window

DEFAULT_JPEG_QUALITY = 90


def encode_image(image: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> str:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_image(data: str) -> np.ndarray:
    buffer = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid JPEG image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_window(window: Window, quality: int = DEFAULT_JPEG_QUALITY) -> dict:
    return {
        "task": window.task,
        "states": window.states.astype(np.float32).tolist(),
        "commands": window.commands.astype(np.float32).tolist(),
        "images": {
            key: {str(i): encode_image(image, quality) for i, image in frames.items()}
            for key, frames in window.images.items()
        },
    }


def decode_window(data: dict) -> Window:
    states = np.asarray(data["states"], dtype=np.float32)
    commands = np.asarray(data["commands"], dtype=np.float32)
    if states.ndim != 2 or states.shape != commands.shape:
        raise ValueError(f"states {states.shape} and commands {commands.shape} must be matching (T, D) arrays")
    return Window(
        task=str(data["task"]),
        states=states,
        commands=commands,
        images={
            key: {int(i): decode_image(image) for i, image in frames.items()} for key, frames in data["images"].items()
        },
    )


def encode_plan(plan: Plan) -> dict:
    return {"actions": plan.actions.astype(np.float32).tolist(), "timing": plan.timing, "model": plan.model}


def decode_plan(data: dict) -> Plan:
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError("plan actions must be a finite (N, D) array")
    return Plan(actions=actions, timing=dict(data.get("timing", {})), model=data.get("model", ""))

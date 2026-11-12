"""Stable camera identity + role registry, so calibration files follow the *physical* camera.

Identity (`id`), in order of preference:
  1. `vid:pid:serial`  when the USB serial is unique among the connected cameras
  2. `vid:pid@port`    (USB port path, e.g. 1-5) when there is no serial, or the "serial" is just
                       the product string / shared by identical units (true for the cheap USB2.0_CAM1)
Option 2 stays valid only while the camera is plugged into the same USB port.

Roles ("wrist", "side") are mapped to ids in real_setup/cameras/cameras.yaml.
Calibration for a role lives in real_setup/cameras/<role>/calibration.yaml (written by B4).

Library use:
    from camera_registry import find_device, load_calibration
    dev = find_device("wrist")          # '/dev/video0' (or index) of the camera currently holding that role
    calib = load_calibration("wrist")   # dict from calibration.yaml, or None
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

CAM_ROOT = Path(__file__).resolve().parents[1] / "real_setup" / "cameras"
REGISTRY = "cameras.yaml"


def _read(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def discover() -> list[dict]:
    """Connected capture cameras: {device, name, vid, pid, serial, port, id, label}."""
    cams: list[dict] = []
    sysfs = Path("/sys/class/video4linux")
    if sysfs.exists():
        for node in sorted(sysfs.glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            if _read(node / "index") != "0":  # one capture stream per physical camera
                continue
            usb = (node / "device").resolve().parent  # the USB device dir, e.g. .../usb1/1-5
            cams.append({
                "device": f"/dev/{node.name}",
                "name": _read(node / "name").split(":")[0],
                "vid": _read(usb / "idVendor"), "pid": _read(usb / "idProduct"),
                "serial": _read(usb / "serial"), "port": usb.name,
            })
    else:  # macOS / Windows: probe indices, no hardware identity available
        import cv2
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append({"device": i, "name": f"Camera {i}", "vid": "", "pid": "", "serial": "", "port": f"index{i}"})
            cap.release()

    for c in cams:
        key = (c["vid"], c["pid"], c["serial"])
        serial_ok = (c["serial"] and c["serial"] not in c["name"]
                     and sum((o["vid"], o["pid"], o["serial"]) == key for o in cams) == 1)
        c["id"] = f"{c['vid']}:{c['pid']}:{c['serial']}" if serial_ok else f"{c['vid']}:{c['pid']}@{c['port']}"
    roles = {v: k for k, v in load_registry().items()}
    for c in cams:
        c["role"] = roles.get(c["id"], "")
        c["label"] = f"{c['role'] + ' - ' if c['role'] else ''}{c['name']} [{c['id']}] ({c['device']})"
    return cams


def load_registry(root: Path | None = None) -> dict[str, str]:
    """role -> camera id."""
    p = (root or CAM_ROOT) / REGISTRY
    return (yaml.safe_load(p.read_text()) or {}).get("roles", {}) if p.exists() else {}


def save_role(role: str, cam_id: str, root: Path | None = None) -> None:
    root = root or CAM_ROOT
    roles = {r: i for r, i in load_registry(root).items() if i != cam_id and r != role}  # one id per role
    roles[role] = cam_id
    root.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY).write_text(yaml.safe_dump({"roles": dict(sorted(roles.items()))}, sort_keys=False))


def find_device(role: str, root: Path | None = None):
    """Device path/index of the connected camera assigned to `role` (None if it isn't plugged in)."""
    cam_id = load_registry(root).get(role)
    return next((c["device"] for c in discover() if c["id"] == cam_id), None)


def calibration_path(role: str, root: Path | None = None) -> Path:
    return (root or CAM_ROOT) / role / "calibration.yaml"


def load_calibration(role: str, root: Path | None = None) -> dict | None:
    p = calibration_path(role, root)
    return yaml.safe_load(p.read_text()) if p.exists() else None


if __name__ == "__main__":  # quick listing
    for c in discover():
        print(f"{c['role'] or '(no role)':10} {c['id']:28} {c['device']}  {c['name']}")

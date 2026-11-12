"""Stable camera identity + role registry, so calibration files follow the *physical* camera.

Identity (`id`), in order of preference:
  1. `vid:pid:serial`  when the USB serial is unique among the connected cameras
  2. `vid:pid@port`    (USB port path, e.g. 1-5) when there is no serial, or the "serial" is just
                       the product string / shared by identical units (true for the cheap USB2.0_CAM1)
Option 2 stays valid only while the camera is plugged into the same USB port.

Roles ("wrist", "side") are mapped to ids in real_setup/cameras/cameras.yaml, which also stores per-camera
V4L2 **controls** (e.g. manual focus). Intrinsics depend on focus, so a camera that is calibrated at a given focus
must be used at that focus: the controls are re-applied every time the capture app opens the camera, and
`open_capture()` does the same for any other code. Calibration for a role lives in
real_setup/cameras/<role>/calibration.yaml (written by B4, with the controls recorded next to the intrinsics).

    cameras.yaml:
      roles:    {side: '046d:08e5@1-5'}
      controls: {'046d:08e5@1-5': {focus_automatic_continuous: 0, focus_absolute: 20}}

Library use:
    from camera_registry import find_device, load_calibration, open_capture
    dev = find_device("wrist")          # '/dev/video0' (or index) of the camera currently holding that role
    calib = load_calibration("wrist")   # dict from calibration.yaml, or None
    cap = open_capture("wrist")         # cv2.VideoCapture with the saved focus etc. already applied

CLI:  python camera_registry.py [list | read T | apply T | save T name=value ...]   (T = role, camera id or /dev/videoN)
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

CAM_ROOT = Path(__file__).resolve().parents[1] / "real_setup" / "cameras"
REGISTRY = "cameras.yaml"

# V4L2 controls we store/apply. Order matters: a mode switch (autofocus / auto exposure) must be written before the
# manual value it unlocks, otherwise the driver refuses the write.
CONTROL_ORDER = ["focus_automatic_continuous", "focus_absolute", "auto_exposure", "exposure_time_absolute",
                 "exposure_dynamic_framerate", "white_balance_automatic", "white_balance_temperature"]
# manual value -> (mode switch, its "auto" value, its "manual" value)
MODE_SWITCH = {"focus_absolute": ("focus_automatic_continuous", 1, 0),
               "exposure_time_absolute": ("auto_exposure", 3, 1),
               "white_balance_temperature": ("white_balance_automatic", 1, 0)}


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


# --------------------------------------------------------------------------- registry file
def _load_doc(root: Path | None = None) -> dict:
    p = (root or CAM_ROOT) / REGISTRY
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def _save_doc(doc: dict, root: Path | None = None) -> None:
    root = root or CAM_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY).write_text(yaml.safe_dump(doc, sort_keys=False))


def load_registry(root: Path | None = None) -> dict[str, str]:
    """role -> camera id."""
    return _load_doc(root).get("roles", {})


def save_role(role: str, cam_id: str, root: Path | None = None) -> None:
    doc = _load_doc(root)
    roles = {r: i for r, i in doc.get("roles", {}).items() if i != cam_id and r != role}  # one id per role
    roles[role] = cam_id
    doc["roles"] = dict(sorted(roles.items()))
    _save_doc(doc, root)  # keeps the other sections (controls)


def load_controls(cam_id: str, root: Path | None = None) -> dict[str, int]:
    """Saved V4L2 controls for a camera id (empty if none)."""
    return dict(_load_doc(root).get("controls", {}).get(cam_id, {}))


def save_controls(cam_id: str, controls: dict[str, int], root: Path | None = None) -> None:
    unknown = set(controls) - set(CONTROL_ORDER)
    if unknown:
        raise ValueError(f"unsupported controls {sorted(unknown)}; supported: {CONTROL_ORDER}")
    doc = _load_doc(root)
    ordered = {k: int(controls[k]) for k in CONTROL_ORDER if k in controls}
    doc.setdefault("controls", {})[cam_id] = ordered
    _save_doc(doc, root)


def find_device(role: str, root: Path | None = None):
    """Device path/index of the connected camera assigned to `role` (None if it isn't plugged in)."""
    cam_id = load_registry(root).get(role)
    return next((c["device"] for c in discover() if c["id"] == cam_id), None)


def calibration_path(role: str, root: Path | None = None) -> Path:
    return (root or CAM_ROOT) / role / "calibration.yaml"


def load_calibration(role: str, root: Path | None = None) -> dict | None:
    p = calibration_path(role, root)
    return yaml.safe_load(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- V4L2 controls (Linux, v4l2-ctl)
def _v4l2(device, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["v4l2-ctl", "-d", str(device), *args], capture_output=True, text=True)


def read_controls(device, names=CONTROL_ORDER) -> dict[str, int]:
    """Current value of each control the camera has."""
    out = {}
    for n in names:
        r = _v4l2(device, "--get-ctrl", n)
        m = re.search(r":\s*(-?\d+)", r.stdout)
        if r.returncode == 0 and m:
            out[n] = int(m.group(1))
    return out


def apply_controls(device, controls: dict[str, int], retries: int = 3, settle: float = 0.5) -> list[str]:
    """Write controls one at a time in a safe order, skipping values already set, retrying transient I/O errors.

    (Writing autofocus-off and the focus value in ONE v4l2-ctl call fails; the camera can also answer
    'Input/output error' while its lens is moving, hence the retries.) Returns a list of problems, empty on success."""
    if not isinstance(device, str) or not device.startswith("/dev/"):
        return [f"controls need a /dev/videoN device, got {device!r}"]
    problems = []
    current = read_controls(device, list(controls))

    def write(name, want, tries=retries) -> subprocess.CompletedProcess:
        for attempt in range(tries):
            r = _v4l2(device, "-c", f"{name}={want}")
            if r.returncode == 0:
                time.sleep(settle)
                return r
            time.sleep(1.0 + attempt)
        return r

    for name in [n for n in CONTROL_ORDER if n in controls]:
        want = int(controls[name])
        if current.get(name) == want:
            continue
        r = write(name, want, 1 if name in MODE_SWITCH else retries)  # a stuck camera never recovers by waiting
        if r.returncode != 0 and name in MODE_SWITCH:
            # Seen on the C920 after a process using it dies: the mode switch READS as manual but the camera is stuck,
            # and every manual-value write returns EIO. Flipping the switch to auto and back clears it.
            switch, auto_value, manual_default = MODE_SWITCH[name]
            if _v4l2(device, "-c", f"{switch}={auto_value}").returncode == 0:
                time.sleep(1.0)
                _v4l2(device, "-c", f"{switch}={int(controls.get(switch, manual_default))}")
                time.sleep(1.0)
                r = write(name, want)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout).strip().splitlines()
            problems.append(f"{name}={want}: {msg[-1] if msg else 'failed'}")
    return problems


def apply_saved(device, cam_id: str, root: Path | None = None) -> list[str]:
    ctrl = load_controls(cam_id, root)
    return apply_controls(device, ctrl) if ctrl else []


def open_capture(target: str | int, size: tuple[int, int] | None = None, root: Path | None = None):
    """cv2.VideoCapture for a role (or /dev/videoN) with the camera's saved controls (focus lock...) applied first.

    Use this instead of cv2.VideoCapture(...) anywhere a camera stream is read, so you always see the calibrated focus."""
    import cv2
    cam = resolve(target, root)
    if cam is None:
        raise RuntimeError(f"camera {target!r} not found (plugged in? role saved?)")
    problems = apply_saved(cam["device"], cam["id"], root)
    if problems:
        print(f"[camera_registry] could not apply saved controls to {cam['device']}: {problems}", file=sys.stderr)
    cap = cv2.VideoCapture(cam["device"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def resolve(target, root: Path | None = None) -> dict | None:
    """Connected camera dict for a role name, camera id or device path."""
    for c in discover():
        if target in (c["role"], c["id"], c["device"]):
            return c
    return None


def describe_controls(controls: dict[str, int]) -> str:
    if not controls:
        return "none saved"
    af = controls.get("focus_automatic_continuous")
    parts = []
    if af is not None:
        parts.append("autofocus on" if af else f"manual focus @{controls.get('focus_absolute', '?')}")
    elif "focus_absolute" in controls:
        parts.append(f"focus @{controls['focus_absolute']}")
    parts += [f"{k}={v}" for k, v in controls.items() if not k.startswith("focus")]
    return ", ".join(parts)


def _cli(argv: list[str]) -> int:
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        for c in discover():
            print(f"{c['role'] or '(no role)':10} {c['id']:20} {c['device']:12} {c['name']:22} "
                  f"saved: {describe_controls(load_controls(c['id']))}")
        return 0
    cam = resolve(argv[1]) if len(argv) > 1 else None
    if cam is None:
        print("usage: camera_registry.py [list | read T | apply T | save T name=value ...]  (T = role, id or /dev/videoN)")
        return 2
    if cmd == "read":
        print(read_controls(cam["device"]))
    elif cmd == "apply":
        problems = apply_saved(cam["device"], cam["id"])
        print("applied" if not problems else problems, describe_controls(load_controls(cam["id"])))
        return 1 if problems else 0
    elif cmd == "save":
        kv = {k: int(v) for k, v in (a.split("=", 1) for a in argv[2:])} if len(argv) > 2 else read_controls(
            cam["device"], ["focus_automatic_continuous", "focus_absolute"])
        save_controls(cam["id"], kv)
        print(f"saved for {cam['role'] or cam['id']}: {describe_controls(kv)}")
    else:
        print("unknown command", cmd)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))

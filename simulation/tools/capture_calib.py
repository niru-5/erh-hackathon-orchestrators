"""Gradio UI to capture checkerboard images for camera calibration (TODO step B3).

    simulation/.venv-sim/bin/python simulation/tools/capture_calib.py
    # then open http://127.0.0.1:7860

- Detects connected cameras with a stable identity (USB serial, else USB port; see camera_registry.py)
  so two identical cameras are told apart. Give each a role name (wrist/side); calibration files are
  then found by role: camera_registry.load_calibration('wrist').
- Live preview, then "Start" saves one frame every `interval` seconds, `count` times (both configurable).
- Images land in real_setup/cameras/<role-or-camera-id>/<session>/ and are shown in a gallery;
  past sessions can be reloaded and viewed.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import camera_registry as reg
from camera_registry import discover

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "real_setup" / "cameras"
DEFAULT_INTERVAL_S = 2.0
DEFAULT_COUNT = 20
RESOLUTIONS = ["Camera default", "640x480", "1280x720", "1920x1080"]

OUT_ROOT = DEFAULT_OUT


# --------------------------------------------------------------------------- cameras
def list_cameras() -> dict[str, dict]:
    """label -> camera info (see camera_registry.discover)."""
    return {c["label"]: c for c in discover()}


class Camera:
    """Keeps one device open with a reader thread so preview and capture share the latest frame."""

    def __init__(self) -> None:
        self.device: str | int | None = None
        self.res: str = ""
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._lock = threading.Lock()
        self._run = False
        self._thread: threading.Thread | None = None

    def open(self, device: str | int, res: str) -> None:
        if self._cap is not None and (device, res) == (self.device, self.res) and self._run:
            return
        self.close()
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if isinstance(device, str) else cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {device} (is another program using it?)")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for HD over USB2
        if res in RESOLUTIONS[1:]:
            w, h = map(int, res.split("x"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap, self.device, self.res, self._run = cap, device, res, True
        self._frame = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cap = self._cap
        while self._run and cap is not None:
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_frame(self, timeout: float = 3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.latest()
            if f is not None:
                return f
            time.sleep(0.05)
        return None

    def close(self) -> None:
        self._run = False
        if self._thread:
            self._thread.join(timeout=1)
        if self._cap is not None:
            self._cap.release()
        self._cap, self._thread, self._frame = None, None, None


CAM = Camera()


# --------------------------------------------------------------------------- helpers
def folder(label: str) -> str:
    """Output folder: the camera's role if assigned, else name + USB id (identical cameras never mix)."""
    c = list_cameras().get(label)
    if c is None:
        return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    if c["role"]:
        return c["role"]
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{c['name']}_{c['id']}").strip("_").lower()


def session_dirs(label: str) -> list[str]:
    base = OUT_ROOT / folder(label)
    return sorted((p.name for p in base.glob("*") if p.is_dir()), reverse=True) if base.exists() else []


def images_in(label: str, session: str) -> list[str]:
    return [str(p) for p in sorted((OUT_ROOT / folder(label) / session).glob("*.png"))]


def open_cam(label: str, res: str):
    cams = list_cameras()
    if label not in cams:
        raise gr.Error("Camera not found, press Refresh.")
    try:
        CAM.open(cams[label]["device"], res)
    except RuntimeError as e:
        raise gr.Error(str(e))


# --------------------------------------------------------------------------- callbacks
def cam_info(label: str):
    c = list_cameras().get(label)
    if not c:
        return "", ""
    calib = reg.calibration_path(c["role"]).exists() if c["role"] else False
    return c["role"], (f"id: {c['id']}\nusb port: {c['port']}   serial: {c['serial'] or 'none'}\n"
                       f"calibration: {'found' if calib else 'not yet'}")


def save_role(label: str, role: str):
    c = list_cameras().get(label)
    role = re.sub(r"[^A-Za-z0-9_-]+", "_", (role or "").strip())
    if not c or not role:
        raise gr.Error("Select a camera and type a role name (e.g. wrist, side).")
    reg.save_role(role, c["id"], OUT_ROOT)
    cams = list_cameras()
    new = next(l for l, x in cams.items() if x["id"] == c["id"])
    return gr.Dropdown(choices=list(cams), value=new), f"Saved: {role} = {c['id']}"


def refresh_cameras():
    labels = list(list_cameras())
    return gr.Dropdown(choices=labels, value=labels[0] if labels else None)


def preview(label: str, res: str):
    if not label:
        return None
    try:
        open_cam(label, res)
    except gr.Error:
        return None
    f = CAM.latest()
    return None if f is None else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def capture(label: str, res: str, interval: float, count: float):
    """Generator: yields (status, gallery, session dropdown) after each saved image."""
    if not label:
        raise gr.Error("No camera selected.")
    interval, count = max(0.2, float(interval)), max(1, int(count))
    open_cam(label, res)
    if CAM.wait_frame() is None:
        raise gr.Error("Camera opened but gave no frames.")

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / folder(label) / session
    out.mkdir(parents=True, exist_ok=True)
    shots: list[str] = []
    for i in range(1, count + 1):
        deadline = time.time() + interval
        while (left := deadline - time.time()) > 0:  # countdown, so you can move the board
            yield f"Image {i}/{count} in {left:0.1f}s: move the board", shots, gr.skip()
            time.sleep(min(0.2, left))
        frame = CAM.latest()
        if frame is None:
            continue
        path = out / f"img_{i:03d}.png"
        cv2.imwrite(str(path), frame)
        shots.append(str(path))
        yield f"Saved {i}/{count}", shots, gr.skip()
    yield (f"Done: {len(shots)} images in {out}", shots,
           gr.Dropdown(choices=session_dirs(label), value=session))


def snap_one(label: str, res: str):
    open_cam(label, res)
    f = CAM.wait_frame()
    if f is None:
        raise gr.Error("No frame from camera.")
    out = OUT_ROOT / folder(label) / "manual"
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.png"))) + 1
    cv2.imwrite(str(out / f"img_{n:03d}.png"), f)
    return f"Saved manual/img_{n:03d}.png", images_in(label, "manual"), gr.Dropdown(
        choices=session_dirs(label), value="manual")


def load_session(label: str, session: str):
    return images_in(label, session) if label and session else []


def on_camera_change(label: str):
    ses = session_dirs(label) if label else []
    return (gr.Dropdown(choices=ses, value=ses[0] if ses else None), *cam_info(label))


def release():
    CAM.close()
    return "Camera released", None


# --------------------------------------------------------------------------- UI
def build() -> gr.Blocks:
    labels = list(list_cameras())
    first = labels[0] if labels else None
    with gr.Blocks(title="Camera calibration capture") as demo:
        gr.Markdown("## Camera calibration capture\nHold the checkerboard in view, press **Start**, "
                    "and tilt/move it a little between shots (different positions, angles, distances).")
        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Dropdown(labels, value=first, label="Camera")
                refresh = gr.Button("Refresh camera list", size="sm")
                role = gr.Textbox(label="Role name for this camera (e.g. wrist, side)", value=cam_info(first)[0])
                save = gr.Button("Save role for this camera", size="sm")
                ident = gr.Textbox(label="Camera identity", lines=3, interactive=False, value=cam_info(first)[1])
                res = gr.Dropdown(RESOLUTIONS, value="1280x720", label="Resolution")
                interval = gr.Number(DEFAULT_INTERVAL_S, label="Interval between images (s)", minimum=0.2)
                count = gr.Number(DEFAULT_COUNT, label="Number of images", precision=0, minimum=1, maximum=200)
                with gr.Row():
                    start = gr.Button("Start", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                with gr.Row():
                    snap = gr.Button("Snap one", size="sm")
                    rel = gr.Button("Release camera", size="sm")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                live = gr.Image(label="Live preview", height=420, interactive=False)
        with gr.Row():
            session = gr.Dropdown(session_dirs(first) if first else [], label="Saved sessions (view earlier captures)")
        gallery = gr.Gallery(label="Captured images", columns=5, height=360, object_fit="contain")

        timer = gr.Timer(0.2)
        timer.tick(preview, [cam, res], live, show_progress="hidden")
        refresh.click(refresh_cameras, None, cam)
        cam.change(on_camera_change, cam, [session, role, ident])
        save.click(save_role, [cam, role], [cam, status])
        session.change(load_session, [cam, session], gallery)
        run = start.click(capture, [cam, res, interval, count], [status, gallery, session])
        stop.click(lambda: "Stopped", None, status, cancels=[run])
        snap.click(snap_one, [cam, res], [status, gallery, session])
        rel.click(release, None, [status, live])
    return demo


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="root folder for captured images")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = args.out.resolve()
    reg.CAM_ROOT = OUT_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("cameras:", list(list_cameras()) or "none found", file=sys.stderr)
    build().launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
                   allowed_paths=[str(OUT_ROOT)])


if __name__ == "__main__":
    main()

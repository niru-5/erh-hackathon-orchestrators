"""Gradio UI for camera calibration (TODO steps B3/B4): capture board images, judge them live, calibrate, compare.

Board types: AprilGrid, plain checkerboard, ChArUco (set in the "Calibration board" panel, saved to board.yaml).

    simulation/.venv-sim/bin/python simulation/tools/capture_calib.py     # http://127.0.0.1:7860

Tab "Capture"   pick camera (stable id, see camera_registry.py), live preview with the detected tags and a
                GOOD/BAD banner, timed burst (interval + count configurable), per-image verdict, delete images.
Tab "Calibrate" analyse sessions, weed bad images, run OpenCV / Kalibr-style / mrcal on the same images,
                compare numbers + plots, save the chosen one as real_setup/cameras/<role>/calibration.yaml.
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

import board as ag
import calib_methods as cm
import calib_viz as viz
import calibrate as cal
import camera_registry as reg
from camera_registry import discover

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "real_setup" / "cameras"
DEFAULT_INTERVAL_S = 2.0
DEFAULT_COUNT = 20
RESOLUTIONS = ["Camera default", "640x480", "1280x720", "1920x1080"]

OUT_ROOT = DEFAULT_OUT
SPEC = ag.BoardSpec()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    if not base.exists():
        return []
    return sorted((p.name for p in base.glob("*") if p.is_dir() and not p.name.startswith("_") and p.name != "calibration"),
                  reverse=True)


def session_paths(label: str, session: str) -> list[Path]:
    return cal.list_images(OUT_ROOT / folder(label) / session)


def thresholds(min_tags, min_sharp, min_px) -> ag.QualityThresholds:
    return ag.QualityThresholds(int(min_tags), float(min_sharp), float(min_px))


def captioned(label: str, session: str, th: ag.QualityThresholds):
    """[(image path, 'GOOD 14 tags' / 'BAD n tags: reason')] for the gallery."""
    out = []
    for p in session_paths(label, session):
        r = cal.analyze_image(p, SPEC, th)
        out.append((str(p), f"{r.verdict} {r.det.n} {SPEC.unit}" + ("" if r.quality.ok else f": {r.quality.reasons[0]}")))
    return out


def open_cam(label: str, res: str):
    cams = list_cameras()
    if label not in cams:
        raise gr.Error("Camera not found, press Refresh.")
    try:
        CAM.open(cams[label]["device"], res)
    except RuntimeError as e:
        raise gr.Error(str(e))


# --------------------------------------------------------------------------- board callbacks
BOARD_LABELS = {  # (cols label, rows label, size label)
    "aprilgrid": ("Tags across", "Tags down", "Tag size (mm)"),
    "checkerboard": ("Squares across", "Squares down", "Square size (mm)"),
    "charuco": ("Squares across", "Squares down", "Square size (mm)"),
}


def _spec_from(kind, family, cols, rows, size_mm, gap_mm, ratio) -> ag.BoardSpec:
    try:
        spec = ag.BoardSpec(kind=kind, family=family, cols=int(cols), rows=int(rows), tag_size=float(size_mm) / 1000,
                            spacing=float(gap_mm or 0) / 1000, marker_ratio=float(ratio))
    except (ValueError, TypeError) as e:
        raise gr.Error(str(e))
    if spec.tag_size <= 0 or spec.cols < 2 or spec.rows < 2 or (kind == "aprilgrid" and spec.spacing < 0):
        raise gr.Error("Sizes must be positive and the board at least 2 x 2.")
    return spec


def board_fields(kind):
    c, r, z = BOARD_LABELS[kind]
    fams = list(ag.FAMILIES)
    return (gr.Dropdown(choices=fams, visible=kind != "checkerboard"), gr.Number(visible=kind == "aprilgrid"),
            gr.Number(visible=kind == "charuco"), gr.Number(label=c), gr.Number(label=r), gr.Number(label=z))


def board_preview(*vals):
    try:
        return "Not applied yet: " + _spec_from(*vals).describe()
    except gr.Error as e:
        return f"Invalid board: {e}"


def apply_board(*vals):
    global SPEC
    SPEC = _spec_from(*vals)
    ag.save_board(SPEC, OUT_ROOT)
    return f"**Active board:** {SPEC.describe()}", gr.Slider(value=ag.default_thresholds(SPEC).min_tags), \
        f"Board saved to {ag.board_path(OUT_ROOT)}. Re-run Analyze / Compute to re-judge existing images."


# --------------------------------------------------------------------------- capture callbacks
def cam_info(label: str):
    c = list_cameras().get(label)
    if not c:
        return "", ""
    calib = reg.calibration_path(c["role"], OUT_ROOT).exists() if c["role"] else False
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


def preview(label: str, res: str, overlay: bool, min_tags, min_sharp, min_px):
    if not label:
        return None, ""
    try:
        open_cam(label, res)
    except gr.Error:
        return None, "camera unavailable"
    f = CAM.latest()
    if f is None:
        return None, ""
    if not overlay:
        return rgb(f), ""
    det = ag.detect(f, SPEC)
    q = ag.assess(det, SPEC, thresholds(min_tags, min_sharp, min_px))
    return rgb(ag.annotate(f, det, q)), q.text()


def capture(label, res, interval, count, only_good, min_tags, min_sharp, min_px):
    """Generator: yields (status, gallery, session dropdown) after each shot."""
    if not label:
        raise gr.Error("No camera selected.")
    interval, count = max(0.2, float(interval)), max(1, int(count))
    th = thresholds(min_tags, min_sharp, min_px)
    open_cam(label, res)
    if CAM.wait_frame() is None:
        raise gr.Error("Camera opened but gave no frames.")

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / folder(label) / session
    out.mkdir(parents=True, exist_ok=True)
    shots: list[tuple[str, str]] = []
    good = bad = 0
    max_attempts = count * 3 if only_good else count
    attempt = 0
    while attempt < max_attempts and (good < count if only_good else True):
        attempt += 1
        deadline = time.time() + interval
        while (left := deadline - time.time()) > 0:  # countdown, so you can move the board
            yield (f"Shot {attempt}/{max_attempts if only_good else count} in {left:0.1f}s: move the board "
                   f"(good so far: {good}/{count})"), shots, gr.skip(), gr.skip()
            time.sleep(min(0.2, left))
        frame = CAM.latest()
        if frame is None:
            continue
        det = ag.detect(frame, SPEC)
        q = ag.assess(det, SPEC, th)
        if q.ok:
            good += 1
        else:
            bad += 1
        if only_good and not q.ok:
            rej = out / "_rejected"
            rej.mkdir(exist_ok=True)
            cv2.imwrite(str(rej / f"img_{attempt:03d}.png"), frame)
            yield f"Rejected shot {attempt}: {q.text()}", shots, gr.skip(), gr.skip()
            continue
        path = out / f"img_{attempt:03d}.png"
        cv2.imwrite(str(path), frame)
        shots.append((str(path), f"{q.verdict} {det.n} {SPEC.unit}" + ("" if q.ok else f": {q.reasons[0]}")))
        yield f"Saved {path.name}: {q.text()}", shots, gr.skip(), gr.skip()
    ses = session_dirs(label)
    yield (f"Done: {good} good, {bad} bad in {out}", shots,
           gr.Dropdown(choices=ses, value=session), gr.CheckboxGroup(choices=ses, value=ses))


def snap_one(label, res, min_tags, min_sharp, min_px):
    open_cam(label, res)
    f = CAM.wait_frame()
    if f is None:
        raise gr.Error("No frame from camera.")
    out = OUT_ROOT / folder(label) / "manual"
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.png"))) + 1
    cv2.imwrite(str(out / f"img_{n:03d}.png"), f)
    th = thresholds(min_tags, min_sharp, min_px)
    q = ag.assess(ag.detect(f, SPEC), SPEC, th)
    return (f"Saved manual/img_{n:03d}.png: {q.text()}", captioned(label, "manual", th),
            gr.Dropdown(choices=session_dirs(label), value="manual"), gr.CheckboxGroup(choices=session_dirs(label), value=session_dirs(label)))


def load_session(label, session, min_tags, min_sharp, min_px):
    if not (label and session):
        return []
    return captioned(label, session, thresholds(min_tags, min_sharp, min_px))


def on_camera_change(label):
    ses = session_dirs(label) if label else []
    return (gr.Dropdown(choices=ses, value=ses[0] if ses else None), *cam_info(label),
            gr.CheckboxGroup(choices=ses, value=ses))


def remember_selection(label, session, evt: gr.SelectData):
    paths = session_paths(label, session)
    return str(paths[evt.index]) if evt.index is not None and evt.index < len(paths) else ""


def delete_selected(label, session, selected, min_tags, min_sharp, min_px):
    if not selected:
        raise gr.Error("Click an image in the gallery first.")
    p = Path(selected)
    if OUT_ROOT not in p.resolve().parents:
        raise gr.Error("Refusing to delete outside the capture folder.")
    p.unlink(missing_ok=True)
    return f"Deleted {p.parent.name}/{p.name}", captioned(label, session, thresholds(min_tags, min_sharp, min_px)), ""


def move_bad_images(label, session, min_tags, min_sharp, min_px):
    th = thresholds(min_tags, min_sharp, min_px)
    reps = cal.analyze(session_paths(label, session), SPEC, th)
    moved = cal.move_bad([r for r in reps if not r.quality.ok])
    return f"Moved {len(moved)} BAD images to {session}/_rejected/ (recoverable)", captioned(label, session, th)


def release():
    CAM.close()
    return "Camera released", None


# --------------------------------------------------------------------------- calibrate callbacks
def _paths(label, sessions):
    return [p for s in (sessions or []) for p in session_paths(label, s)]


def _quality_rows(reports):
    return [[r.name, r.verdict, r.det.n, round(r.quality.sharpness), round(r.quality.tag_px),
             "" if r.reproj is None else round(r.reproj, 2), r.reason] for r in reports]


def _annotated(reports):
    out = []
    for r in reports:
        img = cv2.imread(str(r.path))
        out.append((rgb(ag.annotate(img, r.det, r.quality)), f"{r.verdict}: {r.name.split('/')[-1]} {r.reason}"))
    return out


def do_analyze(label, sessions, min_tags, min_sharp, min_px):
    paths = _paths(label, sessions)
    if not paths:
        raise gr.Error("No images: capture some first or select sessions.")
    reps = cal.analyze(paths, SPEC, thresholds(min_tags, min_sharp, min_px))
    good = sum(r.quality.ok for r in reps)
    return f"{len(reps)} images, {good} good, {len(reps) - good} bad", _quality_rows(reps), _annotated(reps)


def do_calibrate(label, sessions, min_tags, min_sharp, min_px, methods, weed_on, k):
    paths = _paths(label, sessions)
    if not paths:
        raise gr.Error("No images: capture some first or select sessions.")
    if not methods:
        raise gr.Error("Select at least one method.")
    run = cal.run(paths, SPEC, thresholds(min_tags, min_sharp, min_px), list(methods), bool(weed_on), float(k))
    rows = cal.comparison_rows(run)
    cols = ["method", "images", "own RMS px", "common RMS px", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "fx ±", "time s"]
    table = [[r.get(c, r.get("status", "") if c == "images" else "") for c in cols] for r in rows]
    ok = [m for m, r in run.results.items() if r.ok]
    notes = [f"[{m}] {n}" for m, r in run.results.items() for n in r.notes]
    notes += [f"[{m}] FAILED: {r.error}" for m, r in run.results.items() if not r.ok]
    log = "\n".join(run.log + notes)
    figs = viz.all_figures(run, ok[0] if ok else None)
    return (run, log, table, _quality_rows(run.reports), _annotated(run.reports),
            gr.Dropdown(choices=ok, value=ok[0] if ok else None), gr.Dropdown(choices=ok, value=ok[0] if ok else None),
            figs["coverage"], figs["errors"], figs["parameters"], figs["distortion"],
            figs["poses"], figs["residuals"], figs["undistorted"])


def inspect_method(run, method):
    if run is None or not method:
        return None, None, None
    return viz.poses(run, method), viz.residuals(run, method), viz.undistorted(run, method)


def save_calibration(label, run, method):
    if run is None or not method:
        raise gr.Error("Compute a calibration first.")
    out = cal.save_calibration(run, method, OUT_ROOT / folder(label))
    return f"Saved {method} -> {out}"


# --------------------------------------------------------------------------- UI
def build() -> gr.Blocks:
    labels = list(list_cameras())
    first = labels[0] if labels else None
    ses0 = session_dirs(first) if first else []
    with gr.Blocks(title="Camera calibration") as demo:
        gr.Markdown("## Camera calibration")
        c_lab, r_lab, z_lab = BOARD_LABELS[SPEC.kind]
        with gr.Accordion("Calibration board: what you hold in front of the camera", open=False):
            with gr.Row():
                b_kind = gr.Dropdown([(v, k) for k, v in ag.KINDS.items()], value=SPEC.kind, label="Board type", scale=2)
                b_family = gr.Dropdown(list(ag.FAMILIES), value=SPEC.family, label="Marker family",
                                       visible=SPEC.kind != "checkerboard", scale=1)
            with gr.Row():
                b_cols = gr.Number(SPEC.cols, label=c_lab, precision=0, minimum=2)
                b_rows = gr.Number(SPEC.rows, label=r_lab, precision=0, minimum=2)
                b_size = gr.Number(round(SPEC.tag_size * 1000, 3), label=z_lab, minimum=0.1)
                b_gap = gr.Number(round(SPEC.spacing * 1000, 3), label="Gap between tags (mm)", minimum=0,
                                  visible=SPEC.kind == "aprilgrid")
                b_ratio = gr.Number(SPEC.marker_ratio, label="Marker size / square size", minimum=0.1, maximum=0.95,
                                    visible=SPEC.kind == "charuco")
            b_note = gr.Markdown("Checkerboard and ChArUco: enter the number of **squares** (a board with 9 x 9 inner "
                                 "corners has 10 x 10 squares). A plain checkerboard must be fully visible in every shot.")
            b_info = gr.Markdown(f"**Active board:** {SPEC.describe()}")
            b_apply = gr.Button("Apply & save board", variant="primary", size="sm")
        b_all = [b_kind, b_family, b_cols, b_rows, b_size, b_gap, b_ratio]
        with gr.Row():
            cam = gr.Dropdown(labels, value=first, label="Camera", scale=3)
            refresh = gr.Button("Refresh camera list", size="sm", scale=1)
            role = gr.Textbox(label="Role (e.g. wrist, side)", value=cam_info(first)[0], scale=2)
            save = gr.Button("Save role", size="sm", scale=1)
        ident = gr.Textbox(label="Camera identity", lines=2, interactive=False, value=cam_info(first)[1])
        with gr.Accordion("Image quality thresholds (used by the live check and by calibration)", open=False):
            with gr.Row():
                min_tags = gr.Slider(4, 100, ag.default_thresholds(SPEC).min_tags, step=1, label="min tags / corners seen")
                min_sharp = gr.Slider(0, 400, ag.QualityThresholds.min_sharpness, step=1, label="min sharpness")
                min_px = gr.Slider(5, 60, ag.QualityThresholds.min_tag_px, step=1, label="min tag size (px)")
        th = [min_tags, min_sharp, min_px]
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            # ------------------------------------------------------------- capture
            with gr.Tab("Capture"):
                with gr.Row():
                    with gr.Column(scale=1):
                        res = gr.Dropdown(RESOLUTIONS, value="1280x720", label="Resolution")
                        interval = gr.Number(DEFAULT_INTERVAL_S, label="Interval between images (s)", minimum=0.2)
                        count = gr.Number(DEFAULT_COUNT, label="Number of images", precision=0, minimum=1, maximum=200)
                        only_good = gr.Checkbox(False, label="Keep going until N GOOD images (bad ones go to _rejected/)")
                        overlay = gr.Checkbox(True, label="Live check: draw detected tags + GOOD/BAD")
                        with gr.Row():
                            start = gr.Button("Start", variant="primary")
                            stop = gr.Button("Stop", variant="stop")
                        with gr.Row():
                            snap = gr.Button("Snap one", size="sm")
                            rel = gr.Button("Release camera", size="sm")
                    with gr.Column(scale=2):
                        live = gr.Image(label="Live preview", height=420, interactive=False)
                        verdict = gr.Textbox(label="Live verdict", interactive=False)
                session = gr.Dropdown(ses0, value=ses0[0] if ses0 else None, label="Saved sessions (view earlier captures)",
                                  allow_custom_value=True)
                gallery = gr.Gallery(label="Captured images (click one to select it)", columns=5, height=360, object_fit="contain")
                selected = gr.State("")
                with gr.Row():
                    del_sel = gr.Button("Delete selected image", variant="stop", size="sm")
                    mv_bad = gr.Button("Move all BAD images in this session to _rejected/", size="sm")

            # ------------------------------------------------------------- calibrate
            with gr.Tab("Calibrate"):
                with gr.Row():
                    sessions = gr.CheckboxGroup(ses0, value=ses0, label="Sessions to use")
                with gr.Row():
                    methods = gr.CheckboxGroup([(f"{v}  [{k}]", k) for k, v in cm.METHODS.items()], value=cal.DEFAULT_METHODS,
                                               label="Methods (all run on the same images)")
                with gr.Row():
                    weed_on = gr.Checkbox(True, label="Weed reprojection outliers")
                    k = gr.Slider(1.5, 6, 2.5, step=0.1, label="outlier threshold (x median error)")
                with gr.Row():
                    analyze_btn = gr.Button("Analyze images (quality only)")
                    calib_btn = gr.Button("Compute calibration", variant="primary")
                log = gr.Textbox(label="Log", lines=6, interactive=False)
                run_state = gr.State(None)
                result_df = gr.Dataframe(headers=["method", "images", "own RMS px", "common RMS px", "fx", "fy", "cx", "cy",
                                                  "k1", "k2", "p1", "p2", "k3", "fx ±", "time s"],
                                         label="Method comparison ('common RMS' = identical scoring for every method)",
                                         interactive=False)
                with gr.Row():
                    insp = gr.Dropdown([], label="Inspect method (poses / residuals / undistorted)")
                    pick = gr.Dropdown([], label="Save as calibration.yaml")
                    save_cal = gr.Button("Save calibration")
                with gr.Tabs():
                    with gr.Tab("Coverage"):
                        p_cov = gr.Plot()
                    with gr.Tab("Per-image error"):
                        p_err = gr.Plot()
                    with gr.Tab("Parameters"):
                        p_par = gr.Plot()
                    with gr.Tab("Distortion"):
                        p_dis = gr.Plot()
                    with gr.Tab("Board poses"):
                        p_pos = gr.Plot()
                    with gr.Tab("Residuals"):
                        p_res = gr.Plot()
                    with gr.Tab("Undistorted"):
                        p_und = gr.Plot()
                    with gr.Tab("Image verdicts"):
                        qual_df = gr.Dataframe(headers=["image", "verdict", "tags/corners", "sharp", "tag/square px", "err px", "reason"],
                                               interactive=False)
                        ann = gr.Gallery(label="Annotated images", columns=4, height=420, object_fit="contain")

        # wiring: board
        b_kind.change(board_fields, b_kind, [b_family, b_gap, b_ratio, b_cols, b_rows, b_size])
        for comp in b_all:
            comp.change(board_preview, b_all, b_info, show_progress="hidden")
        b_apply.click(apply_board, b_all, [b_info, min_tags, status])
        # wiring: camera / role
        timer = gr.Timer(0.3)
        timer.tick(preview, [cam, res, overlay, *th], [live, verdict], show_progress="hidden")
        refresh.click(refresh_cameras, None, cam)
        cam.change(on_camera_change, cam, [session, role, ident, sessions])
        save.click(save_role, [cam, role], [cam, status])
        session.change(load_session, [cam, session, *th], gallery)
        # capture
        run = start.click(capture, [cam, res, interval, count, only_good, *th], [status, gallery, session, sessions])
        stop.click(lambda: "Stopped", None, status, cancels=[run])
        snap.click(snap_one, [cam, res, *th], [status, gallery, session, sessions])
        rel.click(release, None, [status, live])
        gallery.select(remember_selection, [cam, session], selected)
        del_sel.click(delete_selected, [cam, session, selected, *th], [status, gallery, selected])
        mv_bad.click(move_bad_images, [cam, session, *th], [status, gallery])
        # calibrate
        analyze_btn.click(do_analyze, [cam, sessions, *th], [log, qual_df, ann])
        calib_btn.click(do_calibrate, [cam, sessions, *th, methods, weed_on, k],
                        [run_state, log, result_df, qual_df, ann, insp, pick,
                         p_cov, p_err, p_par, p_dis, p_pos, p_res, p_und])
        insp.change(inspect_method, [run_state, insp], [p_pos, p_res, p_und])
        save_cal.click(save_calibration, [cam, run_state, pick], status)
    return demo


def main() -> None:
    global OUT_ROOT, SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="root folder for captured images")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    OUT_ROOT = args.out.resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reg.CAM_ROOT = OUT_ROOT
    ag.CAM_ROOT = cal.CAM_ROOT = OUT_ROOT
    SPEC = ag.load_board(OUT_ROOT)
    print("cameras:", list(list_cameras()) or "none found", file=sys.stderr)
    build().launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
                   allowed_paths=[str(OUT_ROOT)])


if __name__ == "__main__":
    main()

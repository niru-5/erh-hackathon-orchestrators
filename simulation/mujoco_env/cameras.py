"""Side and wrist cameras for the MuJoCo scene (see simulation/docs/cameras.md).

MuJoCo cameras are ideal pinholes. We feed them the calibrated fx, fy, cx, cy (focalpixel /
principalpixel) and reproduce the lens distortion afterwards with apply_real_distortion().
The scene must set <visual><global offwidth="1280" offheight="720"/></visual> to render at native size.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

SIM_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = SIM_DIR / "config" / "cameras_sim.yaml"
ASSET_DIR = SIM_DIR / "assets" / "so101" / "assets"
MOUNT_STL = ASSET_DIR / "wrist_camera_mount_so101_v1.stl"   # millimetres
CAMERA_STL = ASSET_DIR / "wrist_camera_so101_v1.stl"        # metres


# ----------------------------------------------------------------------------- config
def load_camera_config() -> dict:
    """cameras_sim.yaml with, for each camera, an added 'intrinsics' dict from its calibration.yaml:
    camera_matrix (3x3 ndarray), dist_coeffs (ndarray), size (w, h)."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    for name in ("side", "wrist"):
        cal = yaml.safe_load((SIM_DIR / cfg[name]["calibration"]).read_text())
        cfg[name]["intrinsics"] = {
            "camera_matrix": np.array(cal["camera_matrix"], dtype=float),
            "dist_coeffs": np.array(cal["dist_coeffs"], dtype=float),
            "size": tuple(int(v) for v in cal["image_size"]),
            "method": cal.get("method"),
            "rms_px": cal.get("rms_px"),
        }
    return cfg


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return load_camera_config()


def render_size(name: str) -> tuple[int, int]:
    """(width, height) in pixels = the calibration size (native)."""
    return _cfg()[name]["intrinsics"]["size"]


# ----------------------------------------------------------------------------- intrinsics
def rescale_intrinsics(K, src_size, dst_size, mode: str = "crop") -> np.ndarray:
    """Convert a camera matrix between image sizes.
    mode='scale'  : same field of view, uniform scale (requires equal aspect ratios).
    mode='crop'   : centre-crop src to dst's aspect ratio, then uniformly scale (e.g. 16:9 -> 4:3
                    keeps the vertical FOV and drops the sides).
    mode='stretch': independent x/y scale (anamorphic; only if the camera really squeezes).
    Assumes pixel centres at integer+0.5 are NOT used: cx, cy are in the OpenCV convention."""
    K = np.array(K, dtype=float).copy()
    sw, sh = src_size
    dw, dh = dst_size
    if mode == "stretch":
        sx, sy, ox, oy = dw / sw, dh / sh, 0.0, 0.0
    elif mode in ("scale", "crop"):
        if mode == "scale" and abs(sw / sh - dw / dh) > 1e-3:
            raise ValueError("aspect ratios differ; use mode='crop' or 'stretch'")
        if dw / dh < sw / sh:       # dst is narrower: crop width, keep height
            cw, ch = sh * dw / dh, sh
        else:                       # dst is wider: crop height
            cw, ch = sw, sw * dh / dw
        ox, oy = (sw - cw) / 2, (sh - ch) / 2
        sx = sy = dw / cw
    else:
        raise ValueError(mode)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - ox) * sx
    K[1, 2] = (K[1, 2] - oy) * sy
    return K


def camera_matrix(name: str, size=None, mode: str = "crop") -> np.ndarray:
    """OpenCV camera matrix of camera `name`, rescaled to `size` (w, h) if given (default native)."""
    i = _cfg()[name]["intrinsics"]
    if size is None or tuple(size) == i["size"]:
        return i["camera_matrix"].copy()
    return rescale_intrinsics(i["camera_matrix"], i["size"], size, mode)


def dist_coeffs(name: str) -> np.ndarray:
    return _cfg()[name]["intrinsics"]["dist_coeffs"].copy()


# ----------------------------------------------------------------------------- poses
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def look_at_rotation_cv(pos, target, up, roll_deg: float = 0.0) -> np.ndarray:
    """Rotation whose columns are the OpenCV camera axes (right, down, forward) in the parent frame."""
    f = _unit(np.asarray(target, float) - np.asarray(pos, float))
    r = _unit(np.cross(f, up))
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=1)
    if roll_deg:
        a = np.radians(roll_deg)
        c, s = np.cos(a), np.sin(a)
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # roll about the optical axis
    return R


def _fmt(v, n=6):
    return " ".join(f"{x:.{n}g}" for x in v)


def _camera_xml(name: str, pos, R_cv) -> str:
    """<camera> element. MuJoCo camera: looks along -z, y up, x right  =>  x_mj = x_cv, y_mj = -y_cv, z_mj = -z_cv.
    xyaxes = (x_mj, y_mj) expressed in the parent frame."""
    cfg = _cfg()[name]
    w, h = cfg["intrinsics"]["size"]
    K = cfg["intrinsics"]["camera_matrix"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # principalpixel is the offset of the principal point from the image centre, x towards the left, y towards
    # the top (verified by rendering: pp = (W/2 - cx, H/2 - cy)). sensorsize is nominal (1 um pixels).
    return (
        f'<camera name="{name}" mode="fixed" pos="{_fmt(pos, 8)}" '
        f'xyaxes="{_fmt(R_cv[:, 0], 8)} {_fmt(-R_cv[:, 1], 8)}" '
        f'resolution="{w} {h}" sensorsize="{w * 1e-6:.6g} {h * 1e-6:.6g}" '
        f'focalpixel="{fx:.6f} {fy:.6f}" principalpixel="{w / 2 - cx:.6f} {h / 2 - cy:.6f}"/>'
    )


def side_pose_cv() -> tuple[np.ndarray, np.ndarray]:
    """(position, R_cv) of the side camera in the world frame; R columns = OpenCV right/down/forward."""
    c = _cfg()["side"]
    pos = np.array(c["position_m"], float)
    o = c["orientation"]
    if o["mode"] == "xyaxes_cv":
        return pos, np.array(o["R_cv"], float)
    return pos, look_at_rotation_cv(pos, o["look_at_m"], o["up_world"], o.get("roll_deg", 0.0))


# Wrist camera mount + camera module, placed in the 'gripper' body exactly as in the upstream SO-ARM100 model
# (Simulation/SO101/so101_new_calib_camera.xml, commit a758567). The camera STL is a 32x32 mm UVC board with the
# lens barrel along its +z axis (barrel z 2..26 mm), so the optical axis is +z of the 'wrist_camera' body.
MOUNT_POS = np.array([-0.0150, 0.0240, -0.0315])
MOUNT_QUAT = np.array([0.4997981226682881, -0.5001999599912927, 0.4998017973423795, -0.5001999599912927])
CAM_POS = np.array([-0.04941, -0.03813, 0.0175])          # in the mount frame
CAM_QUAT = np.array([0.37978693649389456, -0.5964584309361936, -0.37978833153177, -0.596456240026065])


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrist_cad_frame() -> tuple[np.ndarray, np.ndarray]:
    """(position of the camera body origin, rotation) in the 'gripper' frame; rotation columns = camera-body axes."""
    Rm = _quat_to_mat(MOUNT_QUAT)
    Rc = _quat_to_mat(CAM_QUAT)
    return MOUNT_POS + Rm @ CAM_POS, Rm @ Rc


def wrist_pose_cv() -> tuple[np.ndarray, np.ndarray]:
    """(position, R_cv) of the wrist camera in the 'gripper' body frame."""
    c = _cfg()["wrist"]
    if c.get("mode") == "cad":
        p0, R = wrist_cad_frame()
        pos = p0 + R @ np.array([0.0, 0.0, c["lens_z_m"]])
        z = R[:, 2]                                   # optical axis = lens axis of the module
        x0, y0 = R[:, 0], R[:, 1]
        a = np.radians(c.get("roll_deg", 0.0))        # roll of the image about the optical axis
        x = np.cos(a) * x0 + np.sin(a) * y0
        y = np.cross(z, x)
        return pos, np.stack([x, y, z], axis=1)
    pos = np.array(c["position_m"], float)
    if c.get("rotation_cv_override") is not None:
        return pos, np.array(c["rotation_cv_override"], float)
    return pos, look_at_rotation_cv(pos, c["look_at_m"], c["up_gripper"], c.get("roll_deg", 0.0))


def side_camera_xml() -> str:
    """One <camera name="side"/> for <worldbody> (world coordinates, metres)."""
    return _camera_xml("side", *side_pose_cv())


def wrist_camera_xml() -> str:
    """One <camera name="wrist"/> to insert inside the 'gripper' body of so101.xml."""
    return _camera_xml("wrist", *wrist_pose_cv())


def wrist_mount_assets_xml() -> str:
    """<mesh> elements for <asset>; '' when the mount visual is disabled in cameras_sim.yaml."""
    if not _cfg()["wrist"].get("show_mount_visual"):
        return ""
    return (f'<mesh name="wrist_cam_mount" file="{MOUNT_STL}" scale="0.001 0.001 0.001"/>'
            f'<mesh name="wrist_cam_body" file="{CAMERA_STL}"/>')


def wrist_mount_visual_xml() -> str:
    """The camera mount + camera module as two fixed child bodies of 'gripper' (visual only, masses as upstream)."""
    if not _cfg()["wrist"].get("show_mount_visual"):
        return ""
    f = lambda v: " ".join(f"{x:.10g}" for x in v)
    return (
        f'<body name="wrist_camera_mount" pos="{f(MOUNT_POS)}" quat="{f(MOUNT_QUAT)}">'
        '<inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>'
        '<geom type="mesh" mesh="wrist_cam_mount" contype="0" conaffinity="0" group="2" mass="0" rgba="0.1 0.7 0.8 1"/>'
        f'<body name="wrist_camera_module" pos="{f(CAM_POS)}" quat="{f(CAM_QUAT)}">'
        '<inertial pos="0 0 0" mass="0.015" diaginertia="1e-5 1e-5 1e-5"/>'
        '<geom type="mesh" mesh="wrist_cam_body" contype="0" conaffinity="0" group="2" mass="0" rgba="0.08 0.08 0.08 1"/>'
        '</body></body>')


# ----------------------------------------------------------------------------- lens distortion
@lru_cache(maxsize=16)
def _distort_maps(name: str, w: int, h: int):
    import cv2
    K = camera_matrix(name, (w, h), "scale" if abs(w / h - 16 / 9) < 1e-3 else "crop")
    D = dist_coeffs(name)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([u, v], -1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, K, D, P=K).reshape(h, w, 2)   # where each real pixel looks in the ideal image
    return ideal[..., 0].astype(np.float32), ideal[..., 1].astype(np.float32)


def apply_real_distortion(img_sim: np.ndarray, name: str) -> np.ndarray:
    """Make an ideal-pinhole sim render look like the real lens: output pixel p_d samples the sim image at
    undistort(p_d). Image must be at the camera's native aspect (K scaled to its size)."""
    import cv2
    h, w = img_sim.shape[:2]
    mx, my = _distort_maps(name, w, h)
    return cv2.remap(img_sim, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


distort_from_sim = apply_real_distortion


def undistort_to_sim(img_real: np.ndarray, name: str) -> np.ndarray:
    """Real frame -> ideal pinhole image with the same K (directly comparable to a sim render)."""
    import cv2
    h, w = img_real.shape[:2]
    return cv2.undistort(img_real, camera_matrix(name, (w, h)), dist_coeffs(name))

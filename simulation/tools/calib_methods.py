"""Three ways to calibrate a pinhole camera from AprilGrid views, behind one interface.

  opencv  cv2.calibrateCamera on the tag corners (Zhang init + LM). model: 5 (k1 k2 p1 p2 k3) | 4 (k3 fixed) | 8 (rational)
  kalibr  Kalibr-STYLE re-implementation: AprilGrid + pinhole-radtan (k1 k2 r1 r2), closed-form focal init,
          then a batch bundle adjustment (intrinsics + all board poses) with a 3-sigma corner outlier pass.
          This is NOT the Kalibr binary (ROS1, needs a docker image); `export_kalibr` writes the files to run the real one.
  mrcal   the mrcal-calibrate-cameras CLI (the real mrcal). It wants a regular grid of points, so it is fed the
          gap-intersection points of the AprilGrid (each = mean of the diagonal tag-corner pairs facing it).
          Lens model LENSMODEL_OPENCV5 = same parameters as OpenCV's 5-term model.

Every result is stored in OpenCV convention (K, dist = k1 k2 p1 p2 [k3 ..]) and re-scored with the same
`evaluate()` (poses re-estimated by solvePnP on the tag corners), so RMS numbers are comparable across methods.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from aprilgrid import BoardSpec, Detection


@dataclass
class Obs:
    """One image's worth of board observations."""
    name: str
    det: Detection
    obj: np.ndarray    # (N,3) board points of every detected tag corner
    pix: np.ndarray    # (N,2) their pixels
    path: Path | None = None

    @classmethod
    def make(cls, name: str, det: Detection, spec: BoardSpec, path: Path | None = None) -> "Obs":
        obj, pix = det.correspondences(spec)
        return cls(name, det, obj, pix, path)


@dataclass
class CalibResult:
    method: str
    model: str
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]
    own_rms: float                           # RMS reported by the method itself on its own residuals
    seconds: float = 0.0
    n_images: int = 0
    std: dict[str, float] = field(default_factory=dict)   # 1-sigma of fx fy cx cy where the method provides it
    per_image: dict[str, float] = field(default_factory=dict)  # common evaluator, filled by evaluate()
    common_rms: float = float("nan")
    notes: list[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    @property
    def intrinsics(self) -> dict[str, float]:
        return {"fx": float(self.K[0, 0]), "fy": float(self.K[1, 1]), "cx": float(self.K[0, 2]), "cy": float(self.K[1, 2])}


def failed(method: str, model: str, size, err: str) -> CalibResult:
    return CalibResult(method, model, np.eye(3), np.zeros(5), size, float("nan"), ok=False, error=err)


# --------------------------------------------------------------------------- common evaluation
def pnp_pose(obj, pix, K, dist):
    ok, rvec, tvec = cv2.solvePnP(obj, pix, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, pix, K, dist, rvec, tvec)
    return rvec, tvec


def evaluate(res: CalibResult, obs: list[Obs]) -> CalibResult:
    """Per-image RMS reprojection error (px) of the tag corners, poses re-fitted with solvePnP."""
    sq, n = 0.0, 0
    res.per_image = {}
    for o in obs:
        if len(o.obj) < 4:
            continue
        pose = pnp_pose(o.obj, o.pix, res.K, res.dist)
        if pose is None:
            continue
        proj, _ = cv2.projectPoints(o.obj, pose[0], pose[1], res.K, res.dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - o.pix, axis=1)
        res.per_image[o.name] = float(np.sqrt((e ** 2).mean()))
        sq += float((e ** 2).sum()); n += len(e)
    res.common_rms = float(np.sqrt(sq / n)) if n else float("nan")
    return res


def usable(obs: list[Obs]) -> list[Obs]:
    return [o for o in obs if o.obj is not None and len(o.obj) >= 8]


# --------------------------------------------------------------------------- OpenCV
def calibrate_opencv(obs: list[Obs], size, model: str = "5") -> CalibResult:
    t0 = time.time()
    obs = usable(obs)
    name = f"radtan{model}" if model != "8" else "rational8"
    if len(obs) < 3:
        return failed("opencv", name, size, f"need >=3 usable images, have {len(obs)}")
    flags = {"5": 0, "4": cv2.CALIB_FIX_K3, "8": cv2.CALIB_RATIONAL_MODEL}[model]
    objs = [o.obj.astype(np.float32) for o in obs]
    pix = [o.pix.astype(np.float32) for o in obs]
    K0 = cv2.initCameraMatrix2D(objs, pix, size, 1.0)
    try:
        rms, K, dist, _, _, std_in, _, _ = cv2.calibrateCameraExtended(
            objs, pix, size, K0, None, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
    except cv2.error as e:
        return failed("opencv", name, size, str(e).splitlines()[-1])
    r = CalibResult("opencv", name, K, dist.ravel(), tuple(size), float(rms), time.time() - t0, len(obs))
    s = std_in.ravel()
    r.std = dict(zip(("fx", "fy", "cx", "cy"), map(float, s[:4])))
    return r


# --------------------------------------------------------------------------- Kalibr-style
def _radtan_project(P, intr, dist):
    x, y = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    k1, k2, r1, r2 = dist
    r2_ = x * x + y * y
    rad = 1 + k1 * r2_ + k2 * r2_ ** 2
    xd = x * rad + 2 * r1 * x * y + r2 * (r2_ + 2 * x * x)
    yd = y * rad + r1 * (r2_ + 2 * y * y) + 2 * r2 * x * y
    return np.stack([intr[0] * xd + intr[2], intr[1] * yd + intr[3]], 1)


def calibrate_kalibr_style(obs: list[Obs], size) -> CalibResult:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    t0 = time.time()
    obs = usable(obs)
    if len(obs) < 3:
        return failed("kalibr", "pinhole-radtan", size, f"need >=3 usable images, have {len(obs)}")
    objs = [o.obj.astype(np.float32) for o in obs]
    pix = [o.pix.astype(np.float32) for o in obs]
    K0 = cv2.initCameraMatrix2D(objs, pix, size, 1.0)          # closed-form focal init, principal point at the center
    intr0 = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]])
    poses = [pnp_pose(o.obj, o.pix, K0, np.zeros(5)) for o in obs]
    keep = [i for i, p in enumerate(poses) if p is not None]
    obs, poses = [obs[i] for i in keep], [poses[i] for i in keep]
    n_img = len(obs)
    counts = np.array([len(o.obj) for o in obs])
    offs = np.concatenate([[0], np.cumsum(counts)])
    all_obj, all_pix = np.vstack([o.obj for o in obs]), np.vstack([o.pix for o in obs])
    img_of = np.repeat(np.arange(n_img), counts)
    mask = np.ones(len(all_obj), bool)

    x0 = np.concatenate([intr0, np.zeros(4)] + [np.concatenate([p[0].ravel(), p[1].ravel()]) for p in poses])

    def residuals(x, m):
        intr, dist = x[:4], x[4:8]
        out = np.zeros((len(all_obj), 2))
        for i in range(n_img):
            sl = slice(offs[i], offs[i + 1])
            rv, tv = x[8 + 6 * i: 11 + 6 * i], x[11 + 6 * i: 14 + 6 * i]
            R, _ = cv2.Rodrigues(rv)
            out[sl] = _radtan_project(all_obj[sl] @ R.T + tv, intr, dist) - all_pix[sl]
        return out[m].ravel()

    def sparsity(m):
        idx = np.flatnonzero(m)
        S = lil_matrix((2 * len(idx), len(x0)), dtype=int)
        S[:, :8] = 1
        for row, k in enumerate(idx):
            c = 8 + 6 * img_of[k]
            S[2 * row: 2 * row + 2, c:c + 6] = 1
        return S

    def solve(x, m):
        return least_squares(residuals, x, args=(m,), jac_sparsity=sparsity(m), x_scale="jac", method="trf",
                             xtol=1e-9, ftol=1e-9, max_nfev=60)

    sol = solve(x0, mask)
    e = np.linalg.norm(residuals(sol.x, np.ones(len(all_obj), bool)).reshape(-1, 2), axis=1)
    sigma = float(np.sqrt((e[mask] ** 2).mean()))
    n_out = int((e > max(3 * sigma, 1.0)).sum())      # 3-sigma corner outlier pass, then re-solve
    if n_out:
        mask = e <= max(3 * sigma, 1.0)
        sol = solve(sol.x, mask)
    x = sol.x
    # parameter covariance from the Jacobian: (J^T J)^-1 * sigma^2 (intrinsics block)
    std = {}
    try:
        J = sol.jac.toarray() if hasattr(sol.jac, "toarray") else sol.jac
        cov = np.linalg.pinv(J.T @ J) * (2 * sol.fun.size / max(1, sol.fun.size - len(x))) * float((sol.fun ** 2).mean())
        std = dict(zip(("fx", "fy", "cx", "cy"), map(float, np.sqrt(np.diag(cov)[:4]))))
    except Exception:
        pass
    K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]])
    dist = np.array([x[4], x[5], x[6], x[7], 0.0])      # k1 k2 r1(p1) r2(p2); k3 = 0
    final = residuals(x, mask).reshape(-1, 2)
    r = CalibResult("kalibr", "pinhole-radtan (Kalibr-style, own implementation)", K, dist, tuple(size),
                    float(np.sqrt((final ** 2).sum(1).mean())), time.time() - t0, n_img, std)
    r.notes.append(f"{n_out} corner outliers (> 3 sigma) removed")
    return r


def export_kalibr(obs: list[Obs], spec: BoardSpec, size, out: Path, cam_name: str = "cam0") -> Path:
    """Write what the real Kalibr needs: images as <ns>.png in cam0/ and april.yaml, plus run_kalibr.sh (docker).
    Nothing is executed here."""
    out = Path(out)
    (out / cam_name).mkdir(parents=True, exist_ok=True)
    for i, o in enumerate(obs):
        src = getattr(o, "path", None)
        if src:
            cv2.imwrite(str(out / cam_name / f"{1_000_000_000 * (i + 1)}.png"), cv2.imread(str(src)))
    (out / "april.yaml").write_text(
        "target_type: 'aprilgrid'\n"
        f"tagCols: {spec.cols}\ntagRows: {spec.rows}\n"
        f"tagSize: {spec.tag_size}\ntagSpacing: {spec.spacing / spec.tag_size:.4f}\ncodeOffset: 0\n")
    sh = out / "run_kalibr.sh"
    sh.write_text(
        "#!/usr/bin/env bash\n"
        "# Runs the real Kalibr (ROS1) in docker. Pick an image YOU trust, e.g. build ethz-asl/kalibr's own Dockerfile_ros1_20_04.\n"
        "set -e\nIMG=${KALIBR_IMAGE:-kalibr}\nD=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "docker run --rm -v \"$D\":/data \"$IMG\" bash -c '\n"
        "  source /catkin_ws/devel/setup.bash && cd /data &&\n"
        "  rosrun kalibr kalibr_bagcreater --folder /data --output-bag /data/calib.bag &&\n"
        f"  rosrun kalibr kalibr_calibrate_cameras --bag /data/calib.bag --topics /{cam_name}/image_raw "
        "--models pinhole-radtan --target /data/april.yaml --dont-show-report'\n")
    sh.chmod(0o755)
    return out


# --------------------------------------------------------------------------- mrcal
MRCAL_MIN_POINTS = 8   # mrcal-calibrate-cameras insists on > 2*min(W,H) points per board (16 here), too strict for partial views
_MRCAL_RUNNER = """
import sys, runpy, functools, mrcal
exe, n = sys.argv[1], int(sys.argv[2])
mrcal.compute_chessboard_corners = functools.partial(mrcal.compute_chessboard_corners, Npoint_observations_per_board_min=n)
sys.argv = [exe] + sys.argv[3:]
runpy.run_path(exe, run_name="__main__")
"""


def calibrate_mrcal(obs: list[Obs], spec: BoardSpec, size, lensmodel: str = "LENSMODEL_OPENCV5",
                    skip_warp: bool = True) -> CalibResult:
    t0 = time.time()
    model = lensmodel.replace("LENSMODEL_", "")
    nx, ny = spec.cols - 1, spec.rows - 1
    pitch = spec.pitch
    rows, n_pts = [], {}
    for k, o in enumerate(obs):
        js = o.det.junctions(spec)
        n_pts[o.name] = len(js)
        fname = f"frame{k:03d}.png"
        for j in range(ny):               # mrcal grid: x fastest. (i, j) in id space; kept consistent for all frames
            for i in range(nx):
                p = js.get((i, j))
                rows.append(f"{fname} {p[0]:.4f} {p[1]:.4f} 1" if p is not None else f"{fname} - - -")
    used = [o for o in obs if n_pts[o.name] >= 6]
    if len(used) < 3:
        return failed("mrcal", model, size, f"need >=3 images with >=6 gap-intersection points, have {len(used)}")
    exe = Path(sys.executable).parent / "mrcal-calibrate-cameras"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "corners.vnl").write_text("# filename x y weight\n" + "\n".join(rows) + "\n")
        cmd = [sys.executable, "-c", _MRCAL_RUNNER, str(exe), str(MRCAL_MIN_POINTS), "--corners-cache", str(td / "corners.vnl"), "--corners-cache-has-weights",
               "--lensmodel", lensmodel, "--focal", "800", "--imagersize", str(size[0]), str(size[1]),
               "--object-spacing", f"{pitch:.6f}", "--object-width-n", str(nx), "--object-height-n", str(ny),
               "--outdir", str(td), "--jobs", "1"]
        if skip_warp:
            cmd.append("--skip-calobject-warp-solve")
        cmd.append("frame*.png")
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=td, timeout=300)
        log = p.stdout + p.stderr
        models = sorted(td.glob("*.cameramodel"))
        if p.returncode != 0 or not models:
            return failed("mrcal", model, size, log.strip().splitlines()[-1] if log.strip() else "mrcal failed")
        import mrcal
        m = mrcal.cameramodel(str(models[0]))
        _, par = m.intrinsics()
        par = np.asarray(par, float)
    K = np.array([[par[0], 0, par[2]], [0, par[1], par[3]], [0, 0, 1]])
    dist = par[4:]
    rms = float("nan")
    for line in log.splitlines():
        if line.startswith("RMS reprojection error:"):
            rms = float(line.split(":")[1].split()[0])
    r = CalibResult("mrcal", model, K, dist, tuple(size), rms, time.time() - t0, len(used))
    for line in log.splitlines():
        if line.startswith("Noutliers") or line.startswith("Worst reprojection"):
            r.notes.append(line.strip())
    r.notes.append("calibrated on gap-intersection points (fewer, regular-grid points), scored here on tag corners")
    return r


# --------------------------------------------------------------------------- dispatch
METHODS = {
    "opencv:5": "OpenCV (k1 k2 p1 p2 k3)",
    "opencv:4": "OpenCV (k3 fixed, 4 terms)",
    "opencv:8": "OpenCV rational (8 terms)",
    "kalibr:radtan4": "Kalibr-style pinhole-radtan (4 terms)",
    "mrcal:OPENCV5": "mrcal OPENCV5 (same 5 terms)",
    "mrcal:OPENCV8": "mrcal OPENCV8",
}


def run_method(spec_str: str, obs: list[Obs], spec: BoardSpec, size) -> CalibResult:
    kind, _, arg = spec_str.partition(":")
    try:
        if kind == "opencv":
            r = calibrate_opencv(obs, size, arg or "5")
        elif kind == "kalibr":
            r = calibrate_kalibr_style(obs, size)
        elif kind == "mrcal":
            r = calibrate_mrcal(obs, spec, size, "LENSMODEL_" + (arg or "OPENCV5"))
        else:
            raise ValueError(spec_str)
    except Exception as e:  # keep the other methods running
        return failed(kind, arg, size, f"{type(e).__name__}: {e}")
    r.method = spec_str
    return evaluate(r, obs) if r.ok else r

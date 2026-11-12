"""Camera calibration pipeline: find the board, weed out bad images, run several methods, compare.

    # analyse + weed + calibrate every session of a camera role, compare all methods
    python simulation/tools/calibrate.py wrist
    python simulation/tools/calibrate.py wrist --sessions 20261112_162124 --methods opencv:5 mrcal:OPENCV5
    python simulation/tools/calibrate.py wrist --move-bad          # move BAD images into <session>/_rejected/
    python simulation/tools/calibrate.py wrist --save opencv:5     # write calibration.yaml for camera_registry

An image is weeded out when (1) the board is not found well enough (too few tags, blurry, too small), or
(2) after a first calibration its reprojection error is an outlier (> max(k x median, 1 px)); repeated until stable.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

import board as ag
import calib_methods as cm
from board import BoardSpec, Detection, Quality, QualityThresholds, default_thresholds

CAM_ROOT = ag.CAM_ROOT
IMG_EXT = (".png", ".jpg", ".jpeg")
DEFAULT_METHODS = ["opencv:5", "opencv:4", "kalibr:radtan4", "mrcal:OPENCV5"]


@dataclass
class ImageReport:
    path: Path
    name: str
    det: Detection
    quality: Quality
    reproj: float | None = None   # RMS px under the weeding calibration
    outlier: bool = False

    @property
    def good(self) -> bool:
        return self.quality.ok and not self.outlier

    @property
    def verdict(self) -> str:
        return "GOOD" if self.good else "BAD"

    @property
    def reason(self) -> str:
        if self.outlier:
            return f"reprojection outlier ({self.reproj:.1f}px)"
        return "; ".join(self.quality.reasons)


# --------------------------------------------------------------------------- analysis
_CACHE: dict = {}


def analyze_image(path: Path, spec: BoardSpec, th: QualityThresholds) -> ImageReport:
    key = (str(path), path.stat().st_mtime_ns, spec, th)
    if key not in _CACHE:
        img = cv2.imread(str(path))
        det = ag.detect(img, spec)
        _CACHE[key] = ImageReport(path, f"{path.parent.name}/{path.stem}", det, ag.assess(det, spec, th))
    r = _CACHE[key]
    return ImageReport(r.path, r.name, r.det, r.quality)  # fresh copy: reproj/outlier are per-run


def list_images(session_dir: Path) -> list[Path]:
    return sorted(p for p in Path(session_dir).glob("*") if p.suffix.lower() in IMG_EXT)


def analyze(paths: list[Path], spec: BoardSpec, th: QualityThresholds) -> list[ImageReport]:
    return [analyze_image(p, spec, th) for p in paths]


def to_obs(reports: list[ImageReport], spec: BoardSpec) -> list[cm.Obs]:
    return [cm.Obs.make(r.name, r.det, spec, r.path) for r in reports]


def image_size(reports: list[ImageReport]) -> tuple[int, int]:
    return next((r.det.size for r in reports if r.det.n), (1280, 720))


# --------------------------------------------------------------------------- weeding
def weed(reports: list[ImageReport], spec: BoardSpec, k: float = 2.5, min_rms: float = 1.0,
         max_iters: int = 4, min_keep: int = 5) -> list[str]:
    """Mark reprojection outliers (in place). Returns a log. Quality-BAD images never enter the fits."""
    log = []
    for r in reports:
        r.outlier, r.reproj = False, None
    for it in range(max_iters):
        pool = [r for r in reports if r.good]
        if len(pool) < max(3, min_keep):
            log.append(f"pass {it + 1}: only {len(pool)} usable images, stopping")
            break
        size = image_size(reports)
        res = cm.run_method("opencv:4", to_obs(pool, spec), spec, size)  # 4-term: well conditioned on few views
        if not res.ok:
            log.append(f"pass {it + 1}: calibration failed ({res.error})")
            break
        for r in pool:
            r.reproj = res.per_image.get(r.name)
        errs = np.array([r.reproj for r in pool if r.reproj is not None])
        thr = max(k * float(np.median(errs)), min_rms)
        bad = [r for r in pool if r.reproj is not None and r.reproj > thr]
        log.append(f"pass {it + 1}: {len(pool)} images, RMS {res.common_rms:.2f}px, threshold {thr:.2f}px, "
                   f"dropping {len(bad)}")
        if not bad or len(pool) - len(bad) < min_keep:
            break
        for r in bad:
            r.outlier = True
    return log


# --------------------------------------------------------------------------- run
@dataclass
class CalibRun:
    spec: BoardSpec
    size: tuple[int, int]
    reports: list[ImageReport]
    kept: list[ImageReport]
    obs: list[cm.Obs]
    results: dict[str, cm.CalibResult]
    log: list[str] = field(default_factory=list)


def run(paths: list[Path], spec: BoardSpec, th: QualityThresholds | None = None, methods: list[str] | None = None,
        do_weed: bool = True, k: float = 2.5) -> CalibRun:
    th = th or default_thresholds(spec)
    methods = methods or DEFAULT_METHODS
    reports = analyze(paths, spec, th)
    log = [f"{len(reports)} images, {sum(r.quality.ok for r in reports)} pass the quality checks"]
    if do_weed:
        log += weed(reports, spec, k=k)
    kept = [r for r in reports if r.good]
    size = image_size(reports)
    log.append(f"calibrating with {len(kept)} images ({sum(r.det.n for r in kept)} {spec.unit})")
    obs = to_obs(kept, spec)
    results = {m: cm.run_method(m, obs, spec, size) for m in methods}
    for r in reports:  # per-image error under the primary method, for the table
        if r.reproj is None:
            ref = next((x for x in results.values() if x.ok), None)
            if ref is not None and r.quality.ok and r.det.n >= 4:
                o = cm.Obs.make(r.name, r.det, spec)
                pose = cm.pnp_pose(o.obj, o.pix, ref.K, ref.dist)
                if pose is not None:
                    proj, _ = cv2.projectPoints(o.obj, pose[0], pose[1], ref.K, ref.dist)
                    r.reproj = float(np.sqrt(((proj.reshape(-1, 2) - o.pix) ** 2).sum(1).mean()))
    return CalibRun(spec, size, reports, kept, obs, results, log)


def comparison_rows(run_: CalibRun) -> list[dict]:
    rows = []
    for m, r in run_.results.items():
        if not r.ok:
            rows.append({"method": m, "model": r.model, "status": f"FAILED: {r.error}"})
            continue
        d = {"method": m, "images": r.n_images, "own RMS px": round(r.own_rms, 3), "common RMS px": round(r.common_rms, 3)}
        d.update({k: round(v, 1) for k, v in r.intrinsics.items()})
        for i, n in enumerate(("k1", "k2", "p1", "p2", "k3")):
            d[n] = round(float(r.dist[i]), 4) if i < len(r.dist) else 0.0
        d["fx ±"] = round(r.std.get("fx", float("nan")), 2) if r.std else ""
        d["time s"] = round(r.seconds, 2)
        rows.append(d)
    return rows


# --------------------------------------------------------------------------- save / weed on disk
def save_calibration(run_: CalibRun, method: str, role_dir: Path) -> Path:
    r = run_.results[method]
    if not r.ok:
        raise ValueError(f"{method} failed: {r.error}")
    role_dir = Path(role_dir)
    (role_dir / "calibration").mkdir(parents=True, exist_ok=True)
    doc = {
        "method": method, "model": r.model,
        "image_size": list(run_.size),
        "camera_matrix": r.K.tolist(),
        "dist_coeffs": [float(x) for x in r.dist],       # OpenCV order: k1 k2 p1 p2 [k3 ...]
        "intrinsics": {k: round(v, 4) for k, v in r.intrinsics.items()},
        "rms_px": round(r.common_rms, 4), "n_images": r.n_images,
        "board": {**asdict(run_.spec), "description": run_.spec.describe()},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    text = yaml.safe_dump(doc, sort_keys=False)
    (role_dir / "calibration" / f"{method.replace(':', '_')}.yaml").write_text(text)
    out = role_dir / "calibration.yaml"
    out.write_text(text)
    (role_dir / "calibration" / "comparison.json").write_text(json.dumps(comparison_rows(run_), indent=1))
    return out


def move_bad(reports: list[ImageReport]) -> list[Path]:
    moved = []
    for r in reports:
        if not r.good and r.path.exists():
            dst = r.path.parent / "_rejected"
            dst.mkdir(exist_ok=True)
            moved.append(Path(shutil.move(str(r.path), str(dst / r.path.name))))
    return moved


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", help="camera role / folder under real_setup/cameras (or a path to a folder of sessions)")
    ap.add_argument("--root", type=Path, default=CAM_ROOT)
    ap.add_argument("--sessions", nargs="*", help="session folder names (default: all)")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS, help=f"any of {list(cm.METHODS)}")
    ap.add_argument("--min-tags", type=int, help="min tags (aprilgrid) / corners seen; default depends on the board kind")
    ap.add_argument("--min-sharpness", type=float, default=QualityThresholds.min_sharpness)
    ap.add_argument("--no-weed", action="store_true", help="skip reprojection-outlier rejection")
    ap.add_argument("--move-bad", action="store_true", help="move BAD images to <session>/_rejected/")
    ap.add_argument("--save", metavar="METHOD", help="write calibration.yaml from this method")
    ap.add_argument("--export-kalibr", metavar="DIR", type=Path,
                    help="write the kept images + april.yaml + run_kalibr.sh so the REAL Kalibr can be run (docker, not executed here)")
    a = ap.parse_args()

    role_dir = Path(a.role) if Path(a.role).exists() else a.root / a.role
    spec = ag.load_board(a.root)
    sessions = [role_dir / s for s in a.sessions] if a.sessions else sorted(p for p in role_dir.glob("*") if p.is_dir() and not p.name.startswith(("_", "calibration")))
    paths = [p for s in sessions for p in list_images(s)]
    th = QualityThresholds(a.min_tags or default_thresholds(spec).min_tags, a.min_sharpness)
    print(f"board: {spec.describe()}; {len(paths)} images")
    run_ = run(paths, spec, th, a.methods, not a.no_weed)
    print("\n".join(run_.log))
    print(f"\n{'image':34} {'verdict':7} {'tags':>4} {'sharp':>6} {'px err':>7}  reason")
    for r in run_.reports:
        err = "" if r.reproj is None else f"{r.reproj:7.2f}"
        print(f"{r.name:34} {r.verdict:7} {r.det.n:4d} {r.quality.sharpness:6.0f} {err:>7}  {r.reason}")
    print("\nMethod comparison (same images; 'common RMS' = identical scoring for all):")
    for row in comparison_rows(run_):
        print("  ", row)
    for m, r in run_.results.items():
        for n in r.notes:
            print(f"   [{m}] {n}")
    if a.move_bad:
        print("moved:", len(move_bad(run_.reports)), "images to _rejected/")
    if a.save:
        print("wrote", save_calibration(run_, a.save, role_dir))
    if a.export_kalibr:
        print("kalibr files in", cm.export_kalibr(run_.obs, spec, run_.size, a.export_kalibr), "-> bash run_kalibr.sh")


if __name__ == "__main__":
    main()

"""AprilGrid (Kalibr-style) target: board model, tag detection and per-image quality checks.

Tag id = row * cols + col (Kalibr numbering). The chart is verified against the detections: ids run
right-to-left / bottom-to-top as printed, so the board frame here is the chart rotated 180 deg: id 0 is
the top-right tag, x right, y up, z toward the viewer (right-handed). Each tag's 4 corners are in the
pattern's own order (top-left, top-right, bottom-right, bottom-left) = what OpenCV's aruco returns.

The board is described in real_setup/cameras/board.yaml (created with the measured defaults if missing).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

CAM_ROOT = Path(__file__).resolve().parents[1] / "real_setup" / "cameras"


@dataclass(frozen=True)
class BoardSpec:
    family: str = "36h11"
    cols: int = 12            # tags per row
    rows: int = 9             # tag rows
    tag_size: float = 0.016   # m, black square of one tag (measured ~16 mm)
    spacing: float = 0.005   # m, gap between tags (measured ~5 mm; Kalibr tagSpacing = spacing / tag_size)

    @property
    def n_tags(self) -> int:
        return self.cols * self.rows

    @property
    def pitch(self) -> float:
        return self.tag_size + self.spacing

    def tag_corners(self, tag_id: int) -> np.ndarray:
        """(4,3) corners of a tag in the board frame, order TL, TR, BR, BL of the tag pattern."""
        r, c = divmod(int(tag_id), self.cols)
        x0, y0, s = (self.cols - 1 - c) * self.pitch, (self.rows - 1 - r) * self.pitch, self.tag_size
        return np.array([[x0, y0 + s, 0], [x0 + s, y0 + s, 0], [x0 + s, y0, 0], [x0, y0, 0]], np.float64)

    def junction(self, i: int, j: int) -> np.ndarray:
        """Board-frame center (3,) of the gap intersection between tag cols i,i+1 and rows j,j+1 (id space)."""
        return np.array([(self.cols - 1 - i) * self.pitch - self.spacing / 2,
                         (self.rows - 1 - j) * self.pitch - self.spacing / 2, 0.0])


def board_path(root: Path | None = None) -> Path:
    return (root or CAM_ROOT) / "board.yaml"


def load_board(root: Path | None = None) -> BoardSpec:
    p = board_path(root)
    if not p.exists():
        spec = BoardSpec()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# AprilGrid calibration target: ids row-major (id = row*cols + col); sizes in metres, measured ~16 mm tag / ~5 mm gap\n"
                     + yaml.safe_dump(asdict(spec), sort_keys=False))
        return spec
    return BoardSpec(**(yaml.safe_load(p.read_text()) or {}))


# --------------------------------------------------------------------------- detection
_DICTS = {"36h11": cv2.aruco.DICT_APRILTAG_36h11, "25h9": cv2.aruco.DICT_APRILTAG_25h9,
          "16h5": cv2.aruco.DICT_APRILTAG_16h5}


@dataclass
class Detection:
    ids: np.ndarray                 # (n,) tag ids on this board
    corners: np.ndarray             # (n,4,2) pixel corners (sub-pixel)
    size: tuple[int, int]           # (w, h)
    gray: np.ndarray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    def correspondences(self, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) board points and (N,2) pixels for every detected corner."""
        if not self.n:
            return np.zeros((0, 3)), np.zeros((0, 2))
        return (np.concatenate([spec.tag_corners(i) for i in self.ids]), self.corners.reshape(-1, 2))

    def junctions(self, spec: BoardSpec) -> dict[tuple[int, int], np.ndarray]:
        """Pixel positions of gap intersections. Each is the mean of the diagonal tag-corner pairs that
        face it (tags TL/BR and TR/BL of the junction), so it needs both tags of at least one diagonal."""
        by_id = {int(i): c for i, c in zip(self.ids, self.corners)}
        out = {}
        for j in range(spec.rows - 1):
            for i in range(spec.cols - 1):
                a, b = by_id.get(j * spec.cols + i), by_id.get((j + 1) * spec.cols + i + 1)
                c, d = by_id.get(j * spec.cols + i + 1), by_id.get((j + 1) * spec.cols + i)
                est = []
                if a is not None and b is not None:  # a's bottom-left (3) meets b's top-right (1)
                    est.append((a[3] + b[1]) / 2)
                if c is not None and d is not None:  # c's bottom-right (2) meets d's top-left (0)
                    est.append((c[2] + d[0]) / 2)
                if est:
                    out[(i, j)] = np.mean(est, axis=0)
        return out


def make_detector(spec: BoardSpec) -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # (CORNER_REFINE_APRILTAG returns 0 detections in this OpenCV 5 build)
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(_DICTS[spec.family]), p)


_DET_CACHE: dict = {}


def detect(img: np.ndarray, spec: BoardSpec) -> Detection:
    key = (spec.family,)
    if key not in _DET_CACHE:
        _DET_CACHE[key] = make_detector(spec)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DET_CACHE[key].detectMarkers(gray)
    h, w = gray.shape
    if ids is None:
        return Detection(np.zeros(0, int), np.zeros((0, 4, 2)), (w, h), gray)
    ids = ids.ravel()
    corners = np.array(corners).reshape(-1, 4, 2)
    keep = ids < spec.n_tags  # drop false positives outside this board's id range
    _, first = np.unique(ids[keep], return_index=True)  # and duplicates
    idx = np.flatnonzero(keep)[first]
    return Detection(ids[idx], corners[idx], (w, h), gray)


# --------------------------------------------------------------------------- quality
@dataclass
class Quality:
    n_tags: int
    n_expected: int
    sharpness: float        # mean gradient energy inside the detected tags (higher = sharper)
    tag_px: float           # mean tag side length in px
    coverage: float         # fraction of the image covered by the detected tags' hull
    ok: bool
    reasons: list[str]

    @property
    def verdict(self) -> str:
        return "GOOD" if self.ok else "BAD"

    def text(self) -> str:
        why = "" if self.ok else ": " + "; ".join(self.reasons)
        return (f"{self.verdict}{why}  [{self.n_tags}/{self.n_expected} tags, sharp {self.sharpness:.0f}, "
                f"tag {self.tag_px:.0f}px, cover {100 * self.coverage:.0f}%]")


@dataclass(frozen=True)
class QualityThresholds:
    min_tags: int = 10
    min_sharpness: float = 25.0
    min_tag_px: float = 12.0


def sharpness_of(det: Detection) -> float:
    """Mean Sobel gradient magnitude inside the tag quads: blur or motion lowers it; independent of tag count."""
    if not det.n or det.gray is None:
        return 0.0
    g = det.gray
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mask = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(mask, det.corners.astype(np.int32), 1)
    return float(mag[mask > 0].mean()) if mask.any() else 0.0


def assess(det: Detection, spec: BoardSpec, th: QualityThresholds | None = None) -> Quality:
    th = th or QualityThresholds()
    w, h = det.size
    if not det.n:
        return Quality(0, spec.n_tags, 0.0, 0.0, 0.0, False, ["no board found"])
    side = np.linalg.norm(det.corners - np.roll(det.corners, 1, axis=1), axis=2).mean()
    hull = cv2.convexHull(det.corners.reshape(-1, 2).astype(np.float32))
    cover = float(cv2.contourArea(hull)) / (w * h)
    sharp = sharpness_of(det)
    reasons = []
    if det.n < th.min_tags:
        reasons.append(f"only {det.n} tags (need {th.min_tags})")
    if sharp < th.min_sharpness:
        reasons.append("blurry")
    if side < th.min_tag_px:
        reasons.append("board too far / small")
    return Quality(det.n, spec.n_tags, sharp, float(side), cover, not reasons, reasons)


def annotate(img: np.ndarray, det: Detection, q: Quality) -> np.ndarray:
    """Copy of the BGR image with detected tags drawn and a GOOD/BAD banner."""
    out = img.copy()
    if det.n:
        cv2.aruco.drawDetectedMarkers(out, [c.reshape(1, 4, 2).astype(np.float32) for c in det.corners],
                                      det.ids.reshape(-1, 1).astype(np.int32), (0, 255, 0))
    col = (0, 170, 0) if q.ok else (0, 0, 220)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), col, -1)
    cv2.putText(out, q.text(), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

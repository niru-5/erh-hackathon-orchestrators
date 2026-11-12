"""Calibration target models, detection and per-image quality checks.

Three board kinds share one interface (BoardSpec -> detect() -> Detection):

  aprilgrid     Kalibr's chart: cols x rows tags (default 12 x 9, family 36h11, ~16 mm tags, ~5 mm gaps).
                Elements are tags, 4 corners each. Tag id = row * cols + col. Verified on real images: the chart as
                printed has ids running right-to-left / bottom-to-top, so the board frame is the chart rotated
                180 deg (id 0 top-right, x right, y up, z toward the viewer).
  checkerboard  cols x rows SQUARES (so (cols-1) x (rows-1) inner corners), square side `tag_size`.
                Elements are inner corners (1 point each). Needs the whole board visible.
  charuco       checkerboard with ArUco markers in the white squares; cols x rows squares, square side `tag_size`,
                marker side = marker_ratio * square. Partial views work. Elements are interior corners.

The board is stored in real_setup/cameras/board.yaml (the UI edits it). `tag_size` is the tag side for aprilgrid and
the square side for the other kinds, in metres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

CAM_ROOT = Path(__file__).resolve().parents[1] / "real_setup" / "cameras"

KINDS = {"aprilgrid": "AprilGrid (Kalibr tag grid)", "checkerboard": "Checkerboard (plain squares)",
         "charuco": "ChArUco (checkerboard + markers)"}
FAMILIES = {"36h11": cv2.aruco.DICT_APRILTAG_36h11, "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "16h5": cv2.aruco.DICT_APRILTAG_16h5, "4x4_50": cv2.aruco.DICT_4X4_50, "5x5_100": cv2.aruco.DICT_5X5_100,
            "6x6_250": cv2.aruco.DICT_6X6_250, "7x7_250": cv2.aruco.DICT_7X7_250}


@dataclass(frozen=True)
class BoardSpec:
    kind: str = "aprilgrid"
    family: str = "36h11"      # aprilgrid / charuco marker dictionary
    cols: int = 12             # aprilgrid: tags across; checkerboard / charuco: SQUARES across
    rows: int = 9
    tag_size: float = 0.016    # m; aprilgrid tag side (~16 mm measured) | checkerboard / charuco square side
    spacing: float = 0.005     # m; aprilgrid gap between tags (~5 mm measured). Unused for the other kinds
    marker_ratio: float = 0.7  # charuco only: marker side / square side

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"board kind must be one of {list(KINDS)}")
        if self.kind in ("checkerboard", "charuco") and (self.cols < 3 or self.rows < 3):
            raise ValueError("need at least 3 x 3 squares")

    # ---- counts / geometry
    @property
    def unit(self) -> str:
        return "tags" if self.kind == "aprilgrid" else "corners"

    @property
    def n_tags(self) -> int:
        """Number of detectable elements (tags, or inner corners)."""
        return self.cols * self.rows if self.kind == "aprilgrid" else (self.cols - 1) * (self.rows - 1)

    @property
    def pitch(self) -> float:
        """Spacing of the regular point grid (tag pitch, or square side)."""
        return self.tag_size + self.spacing if self.kind == "aprilgrid" else self.tag_size

    @property
    def grid_dims(self) -> tuple[int, int]:
        """(nx, ny) of the regular grid mrcal is given: tag gap-intersections, or inner corners."""
        return self.cols - 1, self.rows - 1

    def extent(self) -> tuple[float, float]:
        """Physical (width, height) of the pattern in metres."""
        if self.kind == "aprilgrid":
            return (self.cols * self.tag_size + (self.cols - 1) * self.spacing,
                    self.rows * self.tag_size + (self.rows - 1) * self.spacing)
        return self.cols * self.tag_size, self.rows * self.tag_size

    def outline(self) -> np.ndarray:
        """(5,3) closed rectangle of the pattern in the board frame (for drawing)."""
        w, h = self.extent()
        x0 = y0 = 0.0 if self.kind == "aprilgrid" else -self.tag_size
        return np.array([[x0, y0, 0], [x0 + w, y0, 0], [x0 + w, y0 + h, 0], [x0, y0 + h, 0], [x0, y0, 0]], float)

    def describe(self) -> str:
        w, h = self.extent()
        if self.kind == "aprilgrid":
            return (f"AprilGrid {self.cols}x{self.rows} tags ({self.family}), tag {self.tag_size * 1000:g} mm, gap "
                    f"{self.spacing * 1000:g} mm -> {w * 1000:.0f} x {h * 1000:.0f} mm")
        nx, ny = self.grid_dims
        extra = f", markers {self.marker_ratio * self.tag_size * 1000:g} mm ({self.family})" if self.kind == "charuco" else ""
        return (f"{'ChArUco' if self.kind == 'charuco' else 'Checkerboard'} {self.cols}x{self.rows} squares of "
                f"{self.tag_size * 1000:g} mm{extra} -> {nx}x{ny} inner corners, {w * 1000:.0f} x {h * 1000:.0f} mm")

    # ---- model points
    def element_points(self, eid: int) -> np.ndarray:
        """(k,3) board-frame points of element `eid`: 4 corners (TL, TR, BR, BL of the tag pattern) or 1 corner."""
        if self.kind == "aprilgrid":
            return self.tag_corners(eid)
        nx = self.cols - 1
        j, i = divmod(int(eid), nx)
        return np.array([[i * self.tag_size, j * self.tag_size, 0.0]])

    def tag_corners(self, tag_id: int) -> np.ndarray:
        r, c = divmod(int(tag_id), self.cols)
        x0, y0, s = (self.cols - 1 - c) * self.pitch, (self.rows - 1 - r) * self.pitch, self.tag_size
        return np.array([[x0, y0 + s, 0], [x0 + s, y0 + s, 0], [x0 + s, y0, 0], [x0, y0, 0]], np.float64)

    def junction(self, i: int, j: int) -> np.ndarray:
        """aprilgrid: board-frame center (3,) of the gap intersection between tag cols i,i+1 and rows j,j+1 (id space)."""
        return np.array([(self.cols - 1 - i) * self.pitch - self.spacing / 2,
                         (self.rows - 1 - j) * self.pitch - self.spacing / 2, 0.0])


def board_path(root: Path | None = None) -> Path:
    return (root or CAM_ROOT) / "board.yaml"


_HEADER = ("# Calibration target. kind: aprilgrid | checkerboard | charuco. Sizes in metres.\n"
           "# cols x rows = tags (aprilgrid) or SQUARES (checkerboard / charuco); tag_size = tag or square side.\n")


def save_board(spec: BoardSpec, root: Path | None = None) -> Path:
    p = board_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_HEADER + yaml.safe_dump(asdict(spec), sort_keys=False))
    return p


def load_board(root: Path | None = None) -> BoardSpec:
    p = board_path(root)
    if not p.exists():
        spec = BoardSpec()
        save_board(spec, root)
        return spec
    return BoardSpec(**(yaml.safe_load(p.read_text()) or {}))


# --------------------------------------------------------------------------- detection
@dataclass
class Detection:
    ids: np.ndarray                 # (n,) element ids (tag ids or inner-corner ids)
    corners: np.ndarray             # (n,k,2) sub-pixel pixels; k = 4 for tags, 1 for corners
    size: tuple[int, int]           # (w, h)
    gray: np.ndarray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    def correspondences(self, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) board points and (N,2) pixels for every detected point."""
        if not self.n:
            return np.zeros((0, 3)), np.zeros((0, 2))
        return np.concatenate([spec.element_points(i) for i in self.ids]), self.corners.reshape(-1, 2)

    def grid_points(self, spec: BoardSpec) -> dict[tuple[int, int], np.ndarray]:
        """Pixels of the regular grid mrcal wants, keyed (i, j) in id space.

        aprilgrid: gap intersections; each is the mean of the diagonal tag-corner pairs that face it (needs both tags
        of a diagonal). checkerboard / charuco: the inner corners themselves."""
        if spec.kind != "aprilgrid":
            nx = spec.cols - 1
            return {(int(i) % nx, int(i) // nx): c[0] for i, c in zip(self.ids, self.corners)}
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

    junctions = grid_points  # old name


_DET_CACHE: dict = {}


def _aruco_params() -> "cv2.aruco.DetectorParameters":
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # (CORNER_REFINE_APRILTAG returns 0 detections in this OpenCV 5 build)
    return p


def _detector(spec: BoardSpec):
    if spec not in _DET_CACHE:
        dic = cv2.aruco.getPredefinedDictionary(FAMILIES[spec.family]) if spec.kind != "checkerboard" else None
        if spec.kind == "aprilgrid":
            _DET_CACHE[spec] = cv2.aruco.ArucoDetector(dic, _aruco_params())
        elif spec.kind == "charuco":
            board = cv2.aruco.CharucoBoard((spec.cols, spec.rows), spec.tag_size, spec.tag_size * spec.marker_ratio, dic)
            _DET_CACHE[spec] = cv2.aruco.CharucoDetector(board, detectorParams=_aruco_params())
        else:
            _DET_CACHE[spec] = None
    return _DET_CACHE[spec]


def detect(img: np.ndarray, spec: BoardSpec) -> Detection:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    empty = Detection(np.zeros(0, int), np.zeros((0, 1 if spec.kind != "aprilgrid" else 4, 2)), (w, h), gray)

    if spec.kind == "checkerboard":
        pattern = spec.grid_dims
        ok, c = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            return empty
        c = c.reshape(-1, 1, 2)
        return Detection(np.arange(len(c)), c, (w, h), gray)

    if spec.kind == "charuco":
        cc, ci, _, _ = _detector(spec).detectBoard(gray)
        if cc is None or ci is None or len(ci) == 0:
            return empty
        return Detection(ci.ravel().astype(int), np.asarray(cc).reshape(-1, 1, 2), (w, h), gray)

    corners, ids, _ = _detector(spec).detectMarkers(gray)
    if ids is None:
        return empty
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
    sharpness: float        # mean gradient energy around the detected elements (higher = sharper)
    tag_px: float           # tag side, or square side, in px
    coverage: float         # fraction of the image covered by the detections' hull
    ok: bool
    reasons: list[str]
    unit: str = "tags"

    @property
    def verdict(self) -> str:
        return "GOOD" if self.ok else "BAD"

    def text(self) -> str:
        why = "" if self.ok else ": " + "; ".join(self.reasons)
        return (f"{self.verdict}{why}  [{self.n_tags}/{self.n_expected} {self.unit}, sharp {self.sharpness:.0f}, "
                f"{'tag' if self.unit == 'tags' else 'square'} {self.tag_px:.0f}px, cover {100 * self.coverage:.0f}%]")


@dataclass(frozen=True)
class QualityThresholds:
    min_tags: int = 10
    min_sharpness: float = 60.0
    min_tag_px: float = 12.0


def default_thresholds(spec: BoardSpec) -> QualityThresholds:
    """Sensible defaults per board kind (a plain checkerboard is all-or-nothing, so a low floor is fine)."""
    return QualityThresholds(min_tags={"aprilgrid": 10, "charuco": 12, "checkerboard": 6}[spec.kind])


def _spacing_px(det: Detection) -> float:
    """Median nearest-neighbour distance between detected points (px): ~ the square side for corner boards."""
    c = det.corners.reshape(det.n, -1, 2).mean(1)
    if len(c) < 2:
        return 0.0
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def sharpness_of(det: Detection) -> float:
    """Mean Sobel gradient magnitude around the detections: blur or motion lowers it; independent of the count."""
    if not det.n or det.gray is None:
        return 0.0
    g = det.gray
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    mask = np.zeros(g.shape, np.uint8)
    if det.corners.shape[1] == 4:
        cv2.fillPoly(mask, det.corners.astype(np.int32), 1)
    else:
        r = max(3, int(0.35 * _spacing_px(det)))
        for x, y in det.corners.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), r, 1, -1)
    return float(mag[mask > 0].mean()) if mask.any() else 0.0


def assess(det: Detection, spec: BoardSpec, th: QualityThresholds | None = None) -> Quality:
    th = th or default_thresholds(spec)
    w, h = det.size
    if not det.n:
        return Quality(0, spec.n_tags, 0.0, 0.0, 0.0, False, ["no board found"], spec.unit)
    if det.corners.shape[1] == 4:
        side = float(np.linalg.norm(det.corners - np.roll(det.corners, 1, axis=1), axis=2).mean())
    else:
        side = _spacing_px(det)
    pts = det.corners.reshape(-1, 2).astype(np.float32)
    cover = float(cv2.contourArea(cv2.convexHull(pts))) / (w * h) if len(pts) >= 3 else 0.0
    sharp = sharpness_of(det)
    reasons = []
    if det.n < th.min_tags:
        reasons.append(f"only {det.n} {spec.unit} (need {th.min_tags})")
    if sharp < th.min_sharpness:
        reasons.append("blurry")
    if side < th.min_tag_px:
        reasons.append("board too far / small")
    return Quality(det.n, spec.n_tags, sharp, side, cover, not reasons, reasons, spec.unit)


def annotate(img: np.ndarray, det: Detection, q: Quality) -> np.ndarray:
    """Copy of the BGR image with the detections drawn and a GOOD/BAD banner."""
    out = img.copy()
    if det.n:
        if det.corners.shape[1] == 4:
            cv2.aruco.drawDetectedMarkers(out, [c.reshape(1, 4, 2).astype(np.float32) for c in det.corners],
                                          det.ids.reshape(-1, 1).astype(np.int32), (0, 255, 0))
        else:
            for x, y in det.corners.reshape(-1, 2):
                cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
    col = (0, 170, 0) if q.ok else (0, 0, 220)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), col, -1)
    cv2.putText(out, q.text(), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

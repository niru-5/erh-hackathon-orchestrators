"""Render checkerboard / ChArUco / AprilGrid images through a known camera, run the real detectors and calibrate."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import board as bd  # noqa: E402
import calib_methods as cm  # noqa: E402

SIZE = (1280, 720)
K_TRUE = np.array([[900.0, 0, 640], [0, 905.0, 360], [0, 0, 1]])
S = 60  # rendered pixels per square


def board_image(spec: bd.BoardSpec):
    """(image, M) with M mapping image pixels -> board-frame metres (origin = inner corner (0,0))."""
    sq, m = spec.tag_size, S
    w, h = spec.cols * S + 2 * m, spec.rows * S + 2 * m
    if spec.kind == "checkerboard":
        img = np.full((h, w), 255, np.uint8)
        for r in range(spec.rows):
            for c in range(spec.cols):
                if (r + c) % 2 == 0:
                    img[m + r * S: m + (r + 1) * S, m + c * S: m + (c + 1) * S] = 0
    else:
        dic = cv2.aruco.getPredefinedDictionary(bd.FAMILIES[spec.family])
        cb = cv2.aruco.CharucoBoard((spec.cols, spec.rows), sq, sq * spec.marker_ratio, dic)
        img = cb.generateImage((w, h), marginSize=m)
    ppm = S / sq
    M = np.array([[1 / ppm, 0, -(m + sq * ppm) / ppm], [0, 1 / ppm, -(m + sq * ppm) / ppm], [0, 0, 1]])
    return img, M


def render_views(spec, n=14, seed=0):
    img, M = board_image(spec)
    rng = np.random.default_rng(seed)
    w, h = spec.extent()
    ctr = np.array([w / 2 - spec.tag_size, h / 2 - spec.tag_size, 0])  # middle of the inner-corner grid
    out = []
    while len(out) < n:
        R, _ = cv2.Rodrigues(rng.uniform(-0.55, 0.55, 3) * [1, 1, 1.5])
        t = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(0.22, 0.34)]) - R @ ctr
        H = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t])
        view = cv2.warpPerspective(img, H @ M, SIZE, borderValue=128)
        out.append(cv2.GaussianBlur(view, (3, 3), 0.8))
    return out


@pytest.mark.parametrize("spec", [
    bd.BoardSpec(kind="checkerboard", cols=10, rows=10, tag_size=0.008),   # the user's 10x10, 8 mm board
    bd.BoardSpec(kind="charuco", family="4x4_50", cols=7, rows=5, tag_size=0.012, marker_ratio=0.7),
], ids=["checkerboard-10x10-8mm", "charuco-7x5-12mm"])
def test_detect_and_calibrate(spec):
    views = render_views(spec)
    obs = []
    for i, v in enumerate(views):
        det = bd.detect(v, spec)
        if det.n >= 12:
            obs.append(cm.Obs.make(f"v{i}", det, spec))
    assert len(obs) >= 10, f"only {len(obs)} of {len(views)} views detected"
    if spec.kind == "checkerboard":
        assert all(o.det.n == spec.n_tags for o in obs)  # whole board, every time
    for method in ("opencv:4", "mrcal:OPENCV5"):
        r = cm.run_method(method, obs, spec, SIZE)
        assert r.ok, (method, r.error)
        assert abs(r.K[0, 0] - 900) / 900 < 0.03, (method, r.intrinsics)
        assert abs(r.K[0, 2] - 640) < 10 and abs(r.K[1, 2] - 360) < 10, (method, r.intrinsics)
        assert r.common_rms < 1.0, (method, r.common_rms)


def test_describe_and_counts():
    s = bd.BoardSpec(kind="checkerboard", cols=10, rows=10, tag_size=0.008)
    assert s.grid_dims == (9, 9) and s.n_tags == 81 and s.unit == "corners"
    assert abs(s.extent()[0] - 0.08) < 1e-9
    assert "9x9 inner corners" in s.describe() and "80 x 80 mm" in s.describe()
    assert bd.BoardSpec().n_tags == 108  # the AprilGrid default is unchanged


def test_board_yaml_roundtrip(tmp_path):
    s = bd.BoardSpec(kind="checkerboard", cols=10, rows=10, tag_size=0.008)
    bd.save_board(s, tmp_path)
    assert bd.load_board(tmp_path) == s
    (tmp_path / "board.yaml").write_text("family: 36h11\ncols: 12\nrows: 9\ntag_size: 0.016\nspacing: 0.005\n")
    assert bd.load_board(tmp_path).kind == "aprilgrid"  # old files without `kind` still load


def test_blurred_board_is_flagged():
    spec = bd.BoardSpec(kind="checkerboard", cols=10, rows=10, tag_size=0.008)
    sharp = render_views(spec, 1, seed=3)[0]
    blurred = cv2.GaussianBlur(sharp, (0, 0), 9)
    assert bd.assess(bd.detect(sharp, spec), spec).ok
    q = bd.assess(bd.detect(blurred, spec), spec)
    assert not q.ok and "blurry" in q.reasons
    assert not bd.assess(bd.detect(np.full((720, 1280), 120, np.uint8), spec), spec).ok  # blank wall

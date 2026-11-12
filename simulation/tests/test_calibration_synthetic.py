"""Ground-truth test: render synthetic AprilGrid views with a known camera, recover it with every method.

Validates the board model, the junction (gap-intersection) logic used for mrcal, and the three solvers.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import board as ag  # noqa: E402
import calib_methods as cm  # noqa: E402

SIZE = (1280, 720)
K_TRUE = np.array([[900.0, 0, 640], [0, 905.0, 360], [0, 0, 1]])
DIST_TRUE = np.array([-0.25, 0.08, 0.001, -0.0005, 0.0])
SPEC = ag.BoardSpec()


def synth(n_views=25, keep=0.5, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    center = np.array([SPEC.cols * SPEC.pitch / 2, SPEC.rows * SPEC.pitch / 2, 0])
    obs = []
    while len(obs) < n_views:
        rvec = rng.uniform(-0.7, 0.7, 3) * [1, 1, 2]
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), rng.uniform(0.35, 0.6)]) - R @ center
        ids = [i for i in range(SPEC.n_tags) if rng.random() < keep]
        pts = np.concatenate([SPEC.tag_corners(i) for i in ids])
        cam = pts @ R.T + t
        if (cam[:, 2] < 0.1).any() or R[2, 2] < 0.45:  # in front of the camera, not too oblique
            continue
        proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(R)[0], t, K_TRUE, DIST_TRUE)
        proj = proj.reshape(-1, 4, 2) + rng.normal(0, noise, (len(ids), 4, 2))
        inside = ((proj[..., 0] > 0) & (proj[..., 0] < SIZE[0]) & (proj[..., 1] > 0) & (proj[..., 1] < SIZE[1])).all(1)
        if inside.sum() < 12:
            continue
        det = ag.Detection(np.array(ids)[inside], proj[inside], SIZE)
        obs.append(cm.Obs.make(f"v{len(obs):02d}", det, SPEC))
    return obs


@pytest.fixture(scope="module")
def obs():
    return synth()


def close(r, tol_f=0.01):
    assert r.ok, r.error
    assert abs(r.K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < tol_f
    assert abs(r.K[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1] < tol_f
    assert abs(r.K[0, 2] - K_TRUE[0, 2]) < 4 and abs(r.K[1, 2] - K_TRUE[1, 2]) < 4
    assert r.common_rms < 0.6


def test_opencv(obs):
    close(cm.run_method("opencv:5", obs, SPEC, SIZE))


def test_kalibr_style(obs):
    r = cm.run_method("kalibr:radtan4", obs, SPEC, SIZE)
    close(r, 0.015)  # 4-term model: k3 absent in the ground truth anyway


def test_mrcal(obs):
    close(cm.run_method("mrcal:OPENCV5", obs, SPEC, SIZE), 0.015)


def test_junction_geometry(obs):
    """Junctions derived from detected tag corners must land where the board model says they are."""
    o = obs[0]
    js = o.det.junctions(SPEC)
    assert js
    for (i, j), p in js.items():
        proj, _ = cv2.projectPoints(SPEC.junction(i, j)[None], *cm.pnp_pose(o.obj, o.pix, K_TRUE, DIST_TRUE), K_TRUE, DIST_TRUE)
        assert np.linalg.norm(proj.ravel() - p) < 2.0


def test_quality_and_weeding(obs):
    """A corrupted view is flagged as a reprojection outlier; an under-seen view fails the quality gate."""
    import calibrate as cal
    TH = ag.QualityThresholds(min_sharpness=0)  # synthetic views carry no pixels, so no blur measure
    reports = []
    for o in obs[:12]:
        det = ag.Detection(o.det.ids.copy(), o.det.corners.copy(), SIZE)
        reports.append(cal.ImageReport(Path(o.name), o.name, det, ag.assess(det, SPEC, TH)))
    bad = reports[3].det
    bad.corners += np.random.default_rng(1).normal(0, 12, bad.corners.shape)  # smeared / mis-detected corners
    reports[3].quality = ag.assess(bad, SPEC, TH)
    few = reports[5].det
    few.ids, few.corners = few.ids[:3], few.corners[:3]
    reports[5].quality = ag.assess(few, SPEC, TH)
    assert not reports[5].quality.ok and "only 3 tags" in reports[5].quality.reasons[0]
    log = cal.weed(reports, SPEC)
    assert reports[3].outlier and not reports[3].good, log
    assert sum(r.good for r in reports) >= 9

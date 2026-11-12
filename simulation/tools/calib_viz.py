"""Matplotlib figures describing a CalibRun (calibrate.py). Every function returns a Figure (Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import calib_methods as cm  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _ok(run):
    return {m: r for m, r in run.results.items() if r.ok}


def _empty(msg: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


def coverage(run):
    """Where in the image the board corners were seen, with each method's principal point."""
    w, h = run.size
    if not run.obs:
        return _empty("no usable images")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pts = np.vstack([o.pix for o in run.obs])
    gx, gy = 8, 5
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[gx, gy], range=[[0, w], [0, h]])
    ax.imshow(H.T, extent=[0, w, h, 0], cmap="Greens", alpha=0.55, aspect="equal")
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c="k", alpha=0.5)
    ax.plot([w / 2], [h / 2], "k+", ms=14, label="image center")
    for c, (m, r) in zip(COLORS, _ok(run).items()):
        ax.plot(r.K[0, 2], r.K[1, 2], "o", color=c, ms=7, mfc="none", mew=2, label=f"{m} principal pt")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title(f"Corner coverage: {int((H > 0).sum())}/{gx * gy} image cells hit ({len(run.obs)} images)")
    ax.legend(fontsize=7, loc="lower right")
    return fig


def errors(run):
    """Per-image RMS reprojection error, same scoring for every method."""
    ok = _ok(run)
    names = [o.name for o in run.obs]
    if not ok or not names:
        return _empty("no successful calibration")
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(names) + 2), 4))
    wd = 0.8 / len(ok)
    for k, (c, (m, r)) in enumerate(zip(COLORS, ok.items())):
        ax.bar(np.arange(len(names)) + k * wd, [r.per_image.get(n, np.nan) for n in names], wd, color=c,
               label=f"{m} (RMS {r.common_rms:.2f})")
    ax.set_xticks(np.arange(len(names)) + 0.4 - wd / 2, [n.split("/")[-1].replace("img_", "#") + "\n" + n.split("/")[0][-6:]
                                                          for n in names], fontsize=7)
    ax.set_ylabel("RMS reprojection error (px)")
    ax.set_title("Per-image error (poses re-fitted by solvePnP, tag corners)")
    ax.legend(fontsize=7)
    return fig


def distortion(run):
    """Radial displacement the lens model applies (px) along the image diagonal vs the pinhole model."""
    ok = _ok(run)
    if not ok:
        return _empty("no successful calibration")
    w, h = run.size
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, (m, r) in zip(COLORS, ok.items()):
        cx, cy = r.K[0, 2], r.K[1, 2]
        ang = np.arctan2(h - cy, w - cx)
        rr = np.linspace(0, np.hypot(w - cx, h - cy), 80)
        xn = np.stack([rr * np.cos(ang) / r.K[0, 0], rr * np.sin(ang) / r.K[1, 1], np.ones_like(rr)], 1)
        d, _ = cv2.projectPoints(xn.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), r.K, r.dist)
        u, _ = cv2.projectPoints(xn.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), r.K, np.zeros(5))
        c0 = np.array([cx, cy])
        shift = np.linalg.norm(d.reshape(-1, 2) - c0, axis=1) - np.linalg.norm(u.reshape(-1, 2) - c0, axis=1)
        ax.plot(rr, shift, color=c, label=m)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("distance from principal point (px)")
    ax.set_ylabel("radial shift due to distortion (px)\n(+ outward / - inward)")
    ax.set_title("Lens distortion per method")
    ax.legend(fontsize=7)
    return fig


def parameters(run):
    """fx fy cx cy per method (1-sigma bars where available) next to the image center."""
    ok = _ok(run)
    if not ok:
        return _empty("no successful calibration")
    w, h = run.size
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, keys, ref in ((axes[0], ("fx", "fy"), None), (axes[1], ("cx", "cy"), {"cx": w / 2, "cy": h / 2})):
        x = np.arange(len(ok))
        for j, key in enumerate(keys):
            vals = [r.intrinsics[key] for r in ok.values()]
            errs = [r.std.get(key, 0) for r in ok.values()]
            ax.bar(x + j * 0.38, vals, 0.38, yerr=errs, capsize=3, label=key)
            if ref:
                ax.axhline(ref[key], color=COLORS[j], ls="--", lw=0.8)
        ax.set_xticks(x + 0.19, list(ok), fontsize=7, rotation=15)
        ax.legend(fontsize=7)
        ax.set_title("focal length (px)" if ref is None else "principal point (px), dashed = image center")
    return fig


def poses(run, method: str | None = None):
    """Board positions in the camera frame (camera at the origin looking along +z)."""
    ok = _ok(run)
    if not ok:
        return _empty("no successful calibration")
    method = method if method in ok else next(iter(ok))
    r = ok[method]
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    quad = run.spec.outline()
    for k, o in enumerate(run.obs):
        pose = cm.pnp_pose(o.obj, o.pix, r.K, r.dist)
        if pose is None:
            continue
        R, _ = cv2.Rodrigues(pose[0])
        q = quad @ R.T + pose[1].ravel()
        ax.plot(q[:, 0], q[:, 2], -q[:, 1], color=plt.cm.viridis(k / max(1, len(run.obs) - 1)), lw=1.4)
    ax.scatter([0], [0], [0], c="r", s=40)
    ax.text(0, 0, 0, " camera", color="r")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z / depth (m)"); ax.set_zlabel("-y (m)")
    ax.set_title(f"Board poses seen by the camera ({method})")
    return fig


def residuals(run, method: str | None = None, gain: float = 15.0):
    """Reprojection residual vectors (magnified) over the image. Structure here = model/board error, not noise."""
    ok = _ok(run)
    if not ok:
        return _empty("no successful calibration")
    method = method if method in ok else next(iter(ok))
    r = ok[method]
    w, h = run.size
    P, V = [], []
    for o in run.obs:
        pose = cm.pnp_pose(o.obj, o.pix, r.K, r.dist)
        if pose is None:
            continue
        proj, _ = cv2.projectPoints(o.obj, pose[0], pose[1], r.K, r.dist)
        P.append(o.pix); V.append(o.pix - proj.reshape(-1, 2))
    if not P:
        return _empty("no residuals")
    P, V = np.vstack(P), np.vstack(V)
    mag = np.linalg.norm(V, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    q = ax.quiver(P[:, 0], P[:, 1], V[:, 0] * gain, -V[:, 1] * gain, mag, angles="xy", scale_units="xy", scale=1,
                  cmap="plasma", width=0.0025)
    fig.colorbar(q, ax=ax, label="residual (px)")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title(f"Reprojection residuals x{gain:g} ({method}); mean {mag.mean():.2f}px")
    return fig


def undistorted(run, method: str | None = None):
    """Best good image before / after undistortion (straight board edges are the visual check)."""
    ok = _ok(run)
    if not ok or not run.kept:
        return _empty("no successful calibration")
    method = method if method in ok else next(iter(ok))
    r = ok[method]
    best = max(run.kept, key=lambda x: x.det.n)
    img = cv2.cvtColor(cv2.imread(str(best.path)), cv2.COLOR_BGR2RGB)
    und = cv2.undistort(img, r.K, r.dist)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].imshow(img); axes[0].set_title(f"original ({best.name})")
    axes[1].imshow(und); axes[1].set_title(f"undistorted with {method}")
    for a in axes:
        a.axis("off")
    return fig


def all_figures(run, method: str | None = None) -> dict:
    return {"coverage": coverage(run), "errors": errors(run), "parameters": parameters(run),
            "distortion": distortion(run), "poses": poses(run, method), "residuals": residuals(run, method),
            "undistorted": undistorted(run, method)}

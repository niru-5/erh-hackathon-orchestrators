"""Solve the real side camera pose from a frame with the board in view (cv2.solvePnP on the 7x7 inner corners).

Usage (repo root): python simulation/tools/solve_side_pose.py
Prints candidate poses for the 4 symmetric grid labelings; the one matching the measured camera position is kept.
The result is stored by hand in simulation/config/cameras_sim.yaml (side.orientation.R_cv, side.position_m).
"""
import sys, cv2, numpy as np, itertools
sys.path.insert(0, "simulation/mujoco_env")  # run from the repo root
import cameras
K = cameras.camera_matrix("side"); D = cameras.dist_coeffs("side")
im = cv2.imread("simulation/real_setup/cameras/side/20261112_205932/img_001.png")
g = cv2.cvtColor(cv2.undistort(im, K, D, None, K), cv2.COLOR_BGR2GRAY)
ok, c = cv2.findChessboardCornersSB(g, (7, 7), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
c = c.reshape(7, 7, 2)
pitch = 0.144600 / 8; x0, y0, z = 0.1027, 0.1981, 0.0221
def world(rows, cols):  # rows index x (rank), cols index y (file)
    return np.array([[x0 + (r + 1) * pitch, y0 + (cc + 1) * pitch, z] for r in range(rows) for cc in range(cols)])
W = world(7, 7).reshape(7, 7, 3)
res = []
for k in range(4):
    for flip in (False, True):
        img = c.copy()
        if flip: img = img[:, ::-1]
        img = np.rot90(img, k)
        pts = img.reshape(-1, 2).astype(np.float64)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(W.reshape(-1, 3), pts, K, None, flags=cv2.SOLVEPNP_IPPE)
        for rvec, tvec in zip(rvecs, tvecs):
            pr, _ = cv2.projectPoints(W.reshape(-1, 3), rvec, tvec, K, None)
            err = np.sqrt(((pr.reshape(-1, 2) - pts) ** 2).sum(1).mean())
            R, _ = cv2.Rodrigues(rvec); pos = (-R.T @ tvec).ravel()
            res.append((err, k, flip, pos, R))
for err, k, flip, pos, R in sorted(res, key=lambda r: r[0])[:10]:
    print(f"rms {err:6.2f}px rot{k} flip{flip} cam pos mm {np.round(pos*1000,0)}")

best = [r for r in res if r[1] == 0 and not r[2]][0] if False else min([r for r in res if r[3][2] > 0 and abs(r[3][0] - 0.5) < 0.2 and r[3][1] > 0.1], key=lambda r: r[0])
err, k, flip, pos, R = best
Rcw = R.T  # columns = camera axes (right, down, forward) in world
print("SOLVED pos_m", np.round(pos, 4).tolist(), "rms", round(err, 2))
print("R_cv", np.round(Rcw, 6).tolist())

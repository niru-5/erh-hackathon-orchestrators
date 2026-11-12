import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mujoco_env import cameras as C  # noqa: E402


def test_xml_well_formed():
    for s, name in ((C.side_camera_xml(), "side"), (C.wrist_camera_xml(), "wrist")):
        e = ET.fromstring(s)
        assert e.tag == "camera" and e.get("name") == name
    for s in (C.wrist_mount_assets_xml(), C.wrist_mount_visual_xml()):
        if s:
            ET.fromstring(f"<r>{s}</r>")


def test_render_size_and_matrices():
    assert C.render_size("side") == (1280, 720)
    assert C.render_size("wrist") == (1280, 720)
    K = C.camera_matrix("side")
    assert K[0, 0] == pytest.approx(803.3372, abs=1e-3)
    K2 = C.camera_matrix("side", (640, 360))
    assert K2[0, 0] == pytest.approx(K[0, 0] / 2) and K2[0, 2] == pytest.approx(K[0, 2] / 2)
    # 16:9 -> 4:3 centre crop keeps focal length scale = 480/720, cx shifts by the cropped margin
    K43 = C.camera_matrix("side", (640, 480), "crop")
    s = 480 / 720
    assert K43[0, 0] == pytest.approx(K[0, 0] * s)
    assert K43[0, 2] == pytest.approx((K[0, 2] - (1280 - 960) / 2) * s)
    with pytest.raises(ValueError):
        C.rescale_intrinsics(K, (1280, 720), (640, 480), "scale")


def _scene(name):
    wrist = name == "wrist"
    cam = C.wrist_camera_xml() if wrist else C.side_camera_xml()
    body_open = '<body name="gripper" pos="0.3 0.3 0.3" euler="10 20 30">' if wrist else "<body>"
    return f"""<mujoco><visual><global offwidth="1280" offheight="720"/></visual><worldbody>
    {body_open}{cam if wrist else ""}</body>{"" if wrist else cam}
    <body mocap="true" pos="0 0 0"><geom type="sphere" size="0.004" rgba="1 0 0 1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>"""


@pytest.mark.parametrize("name", ["side", "wrist"])
def test_projection_matches_opencv(name):
    xml = _scene(name)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cid = m.cam(name).id
    Rcv = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])  # camera axes in world, OpenCV convention
    t = d.cam_xpos[cid]
    # target point: 0.4 m in front of the camera, offset right/down by 0.05/0.03 m
    P = t + Rcv @ np.array([0.05, 0.03, 0.4])
    d.mocap_pos[0] = P
    mujoco.mj_forward(m, d)
    K = C.camera_matrix(name)
    rvec, _ = cv2.Rodrigues(Rcv.T)
    uv_cv, _ = cv2.projectPoints(P.reshape(1, 1, 3), rvec, -Rcv.T @ t, K, None)
    uv_cv = uv_cv.ravel()
    # analytic
    pc = Rcv.T @ (P - t)
    uv_an = np.array([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])
    assert np.abs(uv_cv - uv_an).max() < 1e-6
    # rendered
    w, h = C.render_size(name)
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, name)
    im = r.render().astype(int)
    ys, xs = np.where((im[..., 0] - im[..., 1] > 60) & (im[..., 0] - im[..., 2] > 60))
    assert len(xs) > 5
    uv_r = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    assert np.abs(uv_r - uv_cv).max() < 0.5, (uv_r, uv_cv)


def test_side_camera_pose_is_a_valid_view_of_the_board():
    pos, R = C.side_pose_cv()
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-4) and np.linalg.det(R) > 0
    assert R[2, 1] < 0  # image 'down' points downwards in the world
    # the board playing-area corners (solved pose: rms 1.6 px) must fall inside the 1280x720 image
    K = C.camera_matrix("side")
    corners = np.array([[x, y, 0.0221] for x in (0.1027, 0.2473) for y in (0.1981, 0.3427)])
    cam = (corners - pos) @ R  # camera coordinates (x right, y down, z forward)
    assert np.all(cam[:, 2] > 0)
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    assert np.all((uv[:, 0] > 0) & (uv[:, 0] < 1280) & (uv[:, 1] > 0) & (uv[:, 1] < 720))


def test_distortion_roundtrip():
    img = np.zeros((720, 1280, 3), np.uint8)
    cv2.rectangle(img, (300, 200), (900, 500), (255, 255, 255), 3)
    back = C.undistort_to_sim(C.apply_real_distortion(img, "side"), "side")
    assert cv2.absdiff(img[150:550, 250:950], back[150:550, 250:950]).mean() < 3

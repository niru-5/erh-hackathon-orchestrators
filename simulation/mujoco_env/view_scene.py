"""View / render the chess scene.

  python simulation/mujoco_env/view_scene.py --layout start --viewer
  python simulation/mujoco_env/view_scene.py --layout start --render out/top.png --camera top
  python simulation/mujoco_env/view_scene.py --layout start --render out/sheet.png --camera sheet   # top, front, 3/4, side, wrist
"""
import argparse
import os
import sys
from pathlib import Path

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scene import load_model, reset_rest  # noqa: E402
from world import load_world, mm  # noqa: E402

# free cameras: lookat (m), distance (m), azimuth, elevation (deg). Board centre is (0.175, 0.2675).
FREE_CAMS = {
    "top": dict(lookat=[0.175, 0.2675, 0.02], distance=0.62, azimuth=90, elevation=-90),
    "front": dict(lookat=[0.175, 0.2675, 0.05], distance=0.75, azimuth=90, elevation=-18),  # from the robot side, looking +y
    "three_quarter": dict(lookat=[0.175, 0.2675, 0.04], distance=0.8, azimuth=125, elevation=-35),
    "free": dict(lookat=[0.175, 0.25, 0.05], distance=1.1, azimuth=135, elevation=-30),
}


def free_camera(name: str) -> mujoco.MjvCamera:
    p = FREE_CAMS[name]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = p["lookat"]
    cam.distance, cam.azimuth, cam.elevation = p["distance"], p["azimuth"], p["elevation"]
    return cam


def render_view(model, data, camera: str, size=(960, 720)) -> np.ndarray:
    w, h = size
    if camera in FREE_CAMS:
        cam = free_camera(camera)
    else:
        cam = camera
        try:
            import cameras

            w, h = cameras.render_size(camera)
        except Exception:
            w, h = 640, 480
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.update_scene(data, camera=cam)
        return r.render().copy()


def contact_sheet(model, data, names, size=(640, 480)) -> np.ndarray:
    import cv2

    tiles = []
    for n in names:
        try:
            img = render_view(model, data, n, size)
        except Exception as e:  # camera missing
            img = np.zeros((size[1], size[0], 3), np.uint8)
            print(f"camera {n} failed: {e}")
        img = cv2.resize(img, size)
        cv2.putText(img, n, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="start", choices=["start", "single", "empty", "photo_210615"])
    ap.add_argument("--square", default=None, help="square for --layout single, e.g. e4")
    ap.add_argument("--piece", default=None, help="piece type for --layout single")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--render", metavar="PNG")
    ap.add_argument("--camera", default="three_quarter", help="free|top|front|three_quarter|side|wrist|sheet")
    ap.add_argument("--settle", type=float, default=0.0, help="simulate this many seconds before rendering")
    a = ap.parse_args()
    kw = {}
    if a.square:
        kw["single_square"] = a.square
    if a.piece:
        kw["single_type"] = a.piece
    model = load_model(a.layout, **kw)
    data = reset_rest(model)
    for _ in range(int(a.settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if a.render:
        import cv2

        Path(a.render).parent.mkdir(parents=True, exist_ok=True)
        if a.camera == "sheet":
            img = contact_sheet(model, data, ["top", "front", "three_quarter", "side", "wrist", "free"])
        else:
            img = render_view(model, data, a.camera)
        cv2.imwrite(a.render, img[:, :, ::-1])
        print("wrote", a.render)
    if a.viewer:
        import mujoco.viewer

        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

"""A1 smoke test: load a tiny scene, step physics, render one frame.

Run from the repo root:  python simulation/mujoco_env/smoke_test.py [--out frame.png]
Headless: MUJOCO_GL=egl (default here when no DISPLAY).
"""

import argparse
import os

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <camera name="side" pos="0.6 -0.6 0.4" xyaxes="0.707 0.707 0 -0.3 0.3 0.9"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.02" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="save the rendered frame as PNG")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    for _ in range(1000):  # 2 s
        mujoco.mj_step(model, data)
    z = data.qpos[2]
    assert abs(z - 0.02) < 5e-3, f"box should rest on the floor, z={z}"

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="side")
        frame = renderer.render()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    assert frame.std() > 1, "frame looks blank"

    if args.out:
        import cv2

        cv2.imwrite(args.out, frame[:, :, ::-1])
    print(f"ok: mujoco {mujoco.__version__}, GL={os.environ.get('MUJOCO_GL', 'default')}, box z={z:.4f}, frame {frame.shape}")


if __name__ == "__main__":
    main()

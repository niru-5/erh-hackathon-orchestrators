"""Standalone stability check + lineup render for the chess pieces.

    simulation/.venv-sim/bin/python simulation/assets/chess/check_pieces.py [--no-render]

Builds a plane + all 12 pieces (6 types x white/black), steps 10 s at dt=0.002, asserts no piece drifts or
tilts (> 0.1 mm / 0.5 deg), and writes simulation/out/pieces_lineup.png.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "simulation"))
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from mujoco_env import pieces as P

DT = 0.002
SECONDS = 10.0


def scene_xml(extra_opt=""):
    bodies = []
    for ci, col in enumerate(("white", "black")):
        for i, t in enumerate(P.PIECE_TYPES):
            bodies.append(P.piece_body_xml(t, col, f"{col}_{t}", (0.03 * i - 0.075, 0.035 * (1 - 2 * ci), 0.0)))
    return f"""<mujoco>
  <option timestep="{DT}" {extra_opt}/>
  <visual><global offwidth="1600" offheight="800"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/></visual>
  <asset>{P.piece_assets_xml()}
    <texture type="skybox" builtin="gradient" rgb1=".6 .7 .8" rgb2=".2 .25 .3" width="64" height="64"/>
    <material name="floor" rgba=".55 .5 .45 1"/></asset>
  <default>{P.default_defaults_xml()}</default>
  <worldbody>
    <light pos="0 -0.1 0.2" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="1 1 .1" material="floor" friction="0.8 0.01 0.001" condim="4"/>
    {chr(10).join(bodies)}
  </worldbody>
</mujoco>"""


def run_stability(opt=""):
    m = mujoco.MjModel.from_xml_string(scene_xml(opt))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p0 = {}
    for col in ("white", "black"):
        for t in P.PIECE_TYPES:
            b = m.body(f"{col}_{t}")
            p0[b.name] = d.xpos[b.id].copy()
    maxdrift, maxtilt = {}, {}
    for _ in range(int(SECONDS / DT)):
        mujoco.mj_step(m, d)
    worst = {}
    for n, p in p0.items():
        b = m.body(n)
        drift = np.abs(d.xpos[b.id] - p).max()
        zaxis = d.xmat[b.id].reshape(3, 3)[:, 2]
        tilt = np.degrees(np.arccos(np.clip(zaxis[2], -1, 1)))
        worst[n] = (drift, tilt)
    return m, d, worst


def render(m, d, path):
    r = mujoco.Renderer(m, 800, 1600)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, 0.013)
    cam.distance, cam.azimuth, cam.elevation = 0.17, 90, -28
    r.update_scene(d, cam)
    import cv2
    img = r.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r.close()


def main():
    m, d, worst = run_stability()
    ok = True
    for n, (drift, tilt) in worst.items():
        flag = drift < 1e-4 and tilt < 0.5
        ok &= flag
        print(f"{n:14s} drift={drift*1e3:.4f} mm tilt={tilt:.3f} deg {'OK' if flag else 'FAIL'}")
    if "--no-render" not in sys.argv:
        render(m, d, REPO / "simulation" / "out" / "pieces_lineup.png")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

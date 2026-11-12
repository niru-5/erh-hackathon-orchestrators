"""A2: inspect the SO-101 MJCF.

  python simulation/mujoco_env/view_robot.py                  # print joint/actuator table
  python simulation/mujoco_env/view_robot.py --render out/    # render zero pose + each joint at its limits (headless OK)
  python simulation/mujoco_env/view_robot.py --viewer         # interactive viewer: drag joints with the sliders (Control panel)
"""

import argparse
import os
from pathlib import Path

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL))


def print_table(model: mujoco.MjModel) -> None:
    print(f"model: {MODEL.name}  bodies={model.nbody} joints={model.njnt} actuators={model.nu} total mass={model.body_mass.sum():.3f} kg")
    print(f"{'joint':14} {'lo deg':>8} {'hi deg':>8} {'act ctrl lo..hi (deg)':>24} {'kp':>8} {'kv':>6} {'force':>6} {'damp':>5} {'fric':>6} {'arm':>6}")
    for name in JOINTS:
        j = model.joint(name)
        a = model.actuator(name)
        lo, hi = np.degrees(j.range)
        clo, chi = np.degrees(a.ctrlrange)
        dof = j.dofadr[0]
        print(
            f"{name:14} {lo:8.1f} {hi:8.1f} {clo:11.1f}..{chi:<11.1f} {a.gainprm[0]:8.1f} {-a.biasprm[2]:6.2f} "
            f"{a.forcerange[1]:6.2f} {model.dof_damping[dof]:5.2f} {model.dof_frictionloss[dof]:6.3f} {model.dof_armature[dof]:6.3f}"
        )


def set_pose(model, data, q: dict[str, float]) -> None:
    data.qpos[:] = 0
    for name, v in q.items():
        data.qpos[model.joint(name).qposadr[0]] = v
    mujoco.mj_forward(model, data)


def render(model, out: Path) -> None:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = 0.5  # the bare robot MJCF has no lights of its own
    model.vis.headlight.diffuse[:] = 0.9
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -20

    def shot(q: dict[str, float], label: str) -> np.ndarray:
        set_pose(model, data, q)
        with mujoco.Renderer(model, height=360, width=480) as r:
            r.update_scene(data, camera=cam)
            img = r.render().copy()
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    rest = shot({}, "all joints at 0 (middle of range)")
    cv2.imwrite(str(out / "zero_pose.png"), rest[:, :, ::-1])
    rows = []
    for name in JOINTS:
        lo, hi = model.joint(name).range
        rows.append(np.hstack([shot({name: lo}, f"{name} = {np.degrees(lo):.0f} deg"), rest, shot({name: hi}, f"{name} = {np.degrees(hi):.0f} deg")]))
    cv2.imwrite(str(out / "joint_limits.png"), np.vstack(rows)[:, :, ::-1])
    print(f"wrote {out}/zero_pose.png and {out}/joint_limits.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", metavar="DIR", help="save zero pose and joint-limit renders here")
    ap.add_argument("--viewer", action="store_true", help="open the interactive viewer")
    args = ap.parse_args()
    model = load()
    print_table(model)
    if args.render:
        render(model, Path(args.render))
    if args.viewer:
        import mujoco.viewer

        mujoco.viewer.launch(model)


if __name__ == "__main__":
    main()

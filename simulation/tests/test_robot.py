from pathlib import Path

import mujoco
import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL))


def test_joint_and_actuator_order(model):
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)] == JOINTS
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)] == JOINTS


def test_actuator_ranges_match_joint_limits(model):
    for name in JOINTS:
        assert np.allclose(model.actuator(name).ctrlrange, model.joint(name).range, atol=1e-4)


def test_total_mass_plausible(model):
    assert 0.4 < model.body_mass.sum() < 1.0  # kg


def test_zero_pose_is_stable_when_held(model):
    data = mujoco.MjData(model)
    for _ in range(1000):  # 2 s with ctrl = 0 (hold the middle of every range)
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.abs(data.qpos).max() < 0.2  # rad: held near zero against gravity

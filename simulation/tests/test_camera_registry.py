import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import camera_registry as reg  # noqa: E402

CAM_ID = "046d:08e5@1-5"


def test_roles_and_controls_do_not_clobber_each_other(tmp_path):
    reg.save_role("side", CAM_ID, tmp_path)
    reg.save_controls(CAM_ID, {"focus_absolute": 20, "focus_automatic_continuous": 0}, tmp_path)
    reg.save_role("wrist", "05a3:9230@1-3", tmp_path)   # re-writing roles must keep the controls
    doc = yaml.safe_load((tmp_path / "cameras.yaml").read_text())
    assert doc["roles"] == {"side": CAM_ID, "wrist": "05a3:9230@1-3"}
    assert reg.load_controls(CAM_ID, tmp_path) == {"focus_automatic_continuous": 0, "focus_absolute": 20}
    assert list(doc["controls"][CAM_ID]) == ["focus_automatic_continuous", "focus_absolute"]  # mode switch first
    assert reg.load_controls("nope", tmp_path) == {}
    with pytest.raises(ValueError):
        reg.save_controls(CAM_ID, {"brightness": 3}, tmp_path)


class FakeCam:
    """Mimics v4l2-ctl: focus_absolute is only writable while autofocus is off; first write can fail with EIO."""

    def __init__(self, flaky=0):
        self.state = {"focus_automatic_continuous": 1, "focus_absolute": 0}
        self.writes, self.flaky = [], flaky

    def __call__(self, device, *args):
        if args[0] == "--get-ctrl":
            n = args[1]
            return subprocess.CompletedProcess(args, 0, f"{n}: {self.state[n]}\n", "") if n in self.state \
                else subprocess.CompletedProcess(args, 1, "", "unknown control")
        name, val = args[1].split("=")
        if name == "focus_absolute" and self.state["focus_automatic_continuous"]:
            return subprocess.CompletedProcess(args, 255, "", "Permission denied")
        if self.flaky:
            self.flaky -= 1
            return subprocess.CompletedProcess(args, 255, "", "Input/output error")
        self.state[name] = int(val)
        self.writes.append((name, int(val)))
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(reg.time, "sleep", lambda s: None)


def test_apply_orders_writes_and_skips_noops(monkeypatch, fast):
    cam = FakeCam()
    monkeypatch.setattr(reg, "_v4l2", cam)
    # given in the "wrong" order: the autofocus switch must still be written first
    assert reg.apply_controls("/dev/video0", {"focus_absolute": 20, "focus_automatic_continuous": 0}) == []
    assert cam.writes == [("focus_automatic_continuous", 0), ("focus_absolute", 20)]
    assert reg.apply_controls("/dev/video0", {"focus_absolute": 20, "focus_automatic_continuous": 0}) == []
    assert len(cam.writes) == 2  # already set: nothing rewritten


def test_apply_retries_transient_io_error(monkeypatch, fast):
    cam = FakeCam(flaky=2)
    monkeypatch.setattr(reg, "_v4l2", cam)
    assert reg.apply_controls("/dev/video0", {"focus_automatic_continuous": 0, "focus_absolute": 20}) == []
    assert cam.state == {"focus_automatic_continuous": 0, "focus_absolute": 20}


def test_apply_reports_persistent_failure(monkeypatch, fast):
    cam = FakeCam(flaky=99)
    monkeypatch.setattr(reg, "_v4l2", cam)
    problems = reg.apply_controls("/dev/video0", {"focus_automatic_continuous": 0})
    assert problems and "Input/output error" in problems[0]
    assert reg.apply_controls(0, {"focus_absolute": 3})[0].startswith("controls need a /dev/videoN")


def test_describe():
    assert reg.describe_controls({}) == "none saved"
    assert reg.describe_controls({"focus_automatic_continuous": 0, "focus_absolute": 20}) == "manual focus @20"
    assert reg.describe_controls({"focus_automatic_continuous": 1}) == "autofocus on"


def test_calibration_records_controls(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate as cal
    import calib_methods as cm
    import test_calibration_synthetic as T
    obs = T.synth(12, 0.5, 0.3, seed=2)
    run = cal.CalibRun(T.SPEC, T.SIZE, [], [], obs, {"opencv:4": cm.run_method("opencv:4", obs, T.SPEC, T.SIZE)})
    out = cal.save_calibration(run, "opencv:4", tmp_path / "side", controls={"focus_automatic_continuous": 0, "focus_absolute": 20})
    assert yaml.safe_load(out.read_text())["camera_controls"] == {"focus_automatic_continuous": 0, "focus_absolute": 20}
    assert reg.load_calibration("side", tmp_path)["camera_controls"]["focus_absolute"] == 20

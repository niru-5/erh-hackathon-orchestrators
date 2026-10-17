import numpy as np
import pytest

from rollout.codec import decode_plan, decode_window, encode_plan, encode_window
from rollout.contract import Observation, Plan, PolicySpec
from rollout.history import History

SPEC = PolicySpec(name="test")


def obs(value: float, hw=(480, 640)) -> Observation:
    images = {k: np.full((*hw, 3), int(value) % 256, np.uint8) for k in ("scene", "wrist", "extra")}
    return Observation(state=np.full(6, value, np.float32), images=images)


def test_first_tick_fills_window_with_state_as_command():
    history = History(SPEC)
    history.append(obs(1), None)
    window = history.window("task")
    assert window.states.shape == window.commands.shape == (8, 6)
    np.testing.assert_array_equal(window.states, 1)
    np.testing.assert_array_equal(window.commands, 1)  # no command yet: the measured state
    assert set(window.images) == {"scene", "wrist"}  # extra cameras dropped
    assert set(window.images["scene"]) == {0, 7}
    assert window.images["scene"][0].shape == (256, 256, 3)  # resized to the model input


def test_later_ticks_shift_and_record_previous_command():
    history = History(SPEC)
    history.append(obs(0), None)
    for t in range(1, 10):
        history.append(obs(t), np.full(6, 100 + t, np.float32))
    window = history.window("task")
    np.testing.assert_array_equal(window.states[:, 0], np.arange(2, 10))
    np.testing.assert_array_equal(window.commands[:, 0], 100 + np.arange(2, 10))
    assert window.images["wrist"][0][0, 0, 0] == 2 and window.images["wrist"][7][0, 0, 0] == 9
    history.reset()
    with pytest.raises(RuntimeError):
        history.window("task")


def test_missing_camera_is_rejected():
    history = History(SPEC)
    with pytest.raises(KeyError):
        history.append(Observation(np.zeros(6, np.float32), {"scene": np.zeros((4, 4, 3), np.uint8)}), None)


def test_window_and_plan_roundtrip():
    history = History(SPEC)
    rng = np.random.default_rng(0)
    smooth = np.repeat(np.repeat(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), 32, 0), 32, 1)
    history.append(Observation(np.arange(6, dtype=np.float32), {"scene": smooth, "wrist": smooth}), None)
    window = history.window("pick it up")
    decoded = decode_window(encode_window(window))
    assert decoded.task == "pick it up"
    np.testing.assert_array_equal(decoded.states, window.states)
    np.testing.assert_array_equal(decoded.commands, window.commands)
    error = np.abs(decoded.images["scene"][7].astype(int) - window.images["scene"][7].astype(int)).mean()
    assert error < 3  # JPEG q90

    plan = Plan(np.linspace(0, 1, 32 * 6, dtype=np.float32).reshape(32, 6), {"inference_ms": 5.0}, "m")
    back = decode_plan(encode_plan(plan))
    np.testing.assert_allclose(back.actions, plan.actions)
    assert back.timing == {"inference_ms": 5.0} and back.model == "m"
    with pytest.raises(ValueError):
        decode_plan({"actions": [[float("nan")] * 6]})

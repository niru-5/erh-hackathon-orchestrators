"""Mock model: every plan repeats the last issued command, so the arm holds still. No GPU needed."""

import time

import numpy as np

from ..contract import Plan, PolicySpec, Window


class HoldModel:
    def __init__(self, spec: PolicySpec | None = None, delay_s: float = 0.0):
        self._spec = spec or PolicySpec(name="hold")
        self.delay_s = delay_s  # simulate inference time

    def spec(self) -> PolicySpec:
        return self._spec

    def predict(self, window: Window) -> Plan:
        started = time.perf_counter()
        if self.delay_s:
            time.sleep(self.delay_s)
        actions = np.repeat(window.commands[-1][None], self._spec.n_action_steps, axis=0).astype(np.float32)
        debug = {"chunk": actions} if window.debug else {}
        return Plan(actions, {"inference_ms": (time.perf_counter() - started) * 1e3}, self._spec.name, debug)

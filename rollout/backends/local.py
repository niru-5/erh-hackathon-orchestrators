"""Call a model in the same process (a GPU machine, or the mock model)."""

import time

from ..contract import Plan, PolicySpec, Window
from ..models import Model


class LocalBackend:
    def __init__(self, model: Model):
        self.model = model

    def spec(self) -> PolicySpec:
        return self.model.spec()

    def predict(self, window: Window) -> Plan:
        started = time.perf_counter()
        plan = self.model.predict(window)
        plan.timing["client_total_ms"] = (time.perf_counter() - started) * 1e3
        return plan

    def close(self) -> None:
        pass
